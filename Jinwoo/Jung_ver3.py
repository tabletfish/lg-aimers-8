import os
import torch
import shutil
import json
import torch.nn as nn
from datasets import load_dataset
# 최신 trl에서는 SFTConfig를 가져와야 합니다.
from trl import SFTTrainer, SFTConfig 
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from google.colab import drive, files

# 0. 드라이브 마운트
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

MODEL_ID = "/content/drive/MyDrive/Colab/base_model"
OUT_DIR  = "./model"
DATASET_ID = "LGAI-EXAONE/MANTA-1M"

# ==========================================
# 1. 모델 로드 & Pruning
# ==========================================
print("[INFO] 1. 모델 로드 및 Pruning...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    attn_implementation="flash_attention_2"
)

# Pruning (안전한 5개 삭제)
layers_to_drop = {9, 11, 13, 17, 19}
if hasattr(model, "model") and hasattr(model.model, "layers"):
    old_layers = model.model.layers
else:
    old_layers = model.model.layers # Fallback

new_layers = nn.ModuleList()
for i, layer in enumerate(old_layers):
    if i not in layers_to_drop:
        new_layers.append(layer)
    else:
        del layer
        
model.model.layers = new_layers
model.config.num_hidden_layers = len(new_layers)
print(f"   -> Pruning 완료: {len(old_layers)} -> {len(new_layers)}")

# ==========================================
# 2. LoRA 파인 튜닝 (최신 trl 문법 적용)
# ==========================================
print("[INFO] 2. LoRA 파인 튜닝...")

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# 데이터 포맷팅 (리스트 에러 방지용 수동 전처리)
def format_data(batch):
    formatted = []
    for convo in batch['conversations']:
        text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
        formatted.append(text)
    return {"text": formatted}

train_dataset = load_dataset(DATASET_ID, split="train")
train_dataset = train_dataset.shuffle(seed=42).select(range(2000))
train_dataset = train_dataset.map(format_data, batched=True, remove_columns=train_dataset.column_names)

# [핵심] SFTConfig 사용 (최신 버전은 모든 설정을 여기 넣어야 함)
sft_config = SFTConfig(
    output_dir="./lora_output",
    dataset_text_field="text",   # 데이터 컬럼명
    max_seq_length=2048,         # 시퀀스 길이
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    max_steps=100,
    fp16=False,
    bf16=True,
    optim="adamw_torch",
    save_strategy="no",
    report_to="none",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    args=sft_config,       # Config 객체 전달
    peft_config=peft_config,
    # 여기에 dataset_text_field 같은 거 넣으면 에러남 (Config에 넣었으니 제거)
)

trainer.train()
print("[INFO] LoRA 학습 완료 및 병합...")
model = trainer.model.merge_and_unload()

# ==========================================
# 3. GPTQ 양자화
# ==========================================
print("[INFO] 3. GPTQ 양자화...")
calib_dataset = load_dataset(DATASET_ID, split="train").shuffle(seed=42).select(range(2000, 2512))

def preprocess_calib(example):
    return {"text": tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)}
ds_calib = calib_dataset.map(preprocess_calib)

recipe = [
    GPTQModifier(scheme="W4A16", targets=["Linear"], ignore=["embed_tokens", "lm_head"], dampening_frac=0.01)
]

oneshot(
    model=model, dataset=ds_calib, recipe=recipe,
    max_seq_length=2048, num_calibration_samples=512
)

# ==========================================
# 4. 저장 및 제출
# ==========================================
print("[INFO] 4. 저장 및 압축...")
if os.path.exists(OUT_DIR): shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

model.save_pretrained(OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(OUT_DIR)

# Config 수정
config_path = os.path.join(OUT_DIR, "config.json")
with open(config_path, "r") as f: config = json.load(f)
config["num_hidden_layers"] = len(new_layers)

# 🔥 layer_types도 같이 잘라주기
if "layer_types" in config:
    config["layer_types"] = config["layer_types"][:len(new_layers)]

if "architectures" not in config: config["architectures"] = ["ExaoneForCausalLM"]
with open(config_path, "w") as f: json.dump(config, f, indent=2)

shutil.make_archive("final_submission_v5", "zip", root_dir=".", base_dir="model")
try: files.download("final_submission_v5.zip")
except: print("다운로드 실패 시 수동 다운로드")
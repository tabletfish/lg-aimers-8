import os
import torch
import shutil
from google.colab import drive
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# ---------------------------------------------------------------------------
# 0. 구글 드라이브 마운트 (이미 되어 있다면 패스)
# ---------------------------------------------------------------------------
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# ---------------------------------------------------------------------------
# 1. 설정 (Configuration)
# ---------------------------------------------------------------------------
# [입력 경로] 구글 드라이브에 있는 베이스 모델 경로
MODEL_ID = "/content/drive/MyDrive/Colab/base_model" 

# [임시 저장 경로] Colab 내부 저장소 (속도 향상을 위해 필수!)
# 드라이브에 직접 저장하면 파일 쓰기 속도가 매우 느립니다.
LOCAL_OUT_DIR = "/content/temp_model_fp8"

# [최종 저장 경로] 작업 완료 후 압축 파일이 저장될 구글 드라이브 위치
DRIVE_SAVE_DIR = "/content/drive/MyDrive/Colab/model"

DATASET_ID = "LGAI-EXAONE/MANTA-1M"
DATASET_SPLIT = "train"
NUM_CALIBRATION_SAMPLES = 1024  # 1024는 너무 많을 수 있어 512 권장
MAX_SEQUENCE_LENGTH = 2048

# FP8 Quantization Settings
SCHEME = "FP8"
TARGETS = ["Linear"]
IGNORE = ["lm_head"]

# 레이어 스킵 로직 (모델 config에서 레이어 수 자동 확인)
try:
    # config를 먼저 로드해서 레이어 수를 확인
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    TOTAL_LAYERS = config.num_hidden_layers
    print(f"[INFO] 감지된 총 레이어 수: {TOTAL_LAYERS}")
except:
    TOTAL_LAYERS = 32 # 실패 시 기본값 (모델에 맞게 수정 필요)
    print(f"[WARNING] config 로드 실패. 기본값 {TOTAL_LAYERS} 사용")

SKIP_COUNT = 1
front_layers = list(range(SKIP_COUNT))
back_layers = list(range(TOTAL_LAYERS - SKIP_COUNT, TOTAL_LAYERS))
layers_to_ignore = front_layers + back_layers

for layer_idx in layers_to_ignore:
    IGNORE.append(f"model.layers.{layer_idx}")

print(f"[INFO] 설정 완료: Scheme={SCHEME}, Ignore Layers={layers_to_ignore}")

# ---------------------------------------------------------------------------
# 2. 모델 및 토크나이저 로드
# ---------------------------------------------------------------------------
print(f"[INFO] 모델 로딩 중... (Path: {MODEL_ID})")

if not os.path.exists(MODEL_ID):
    raise FileNotFoundError(f"구글 드라이브 경로에 모델이 없습니다: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

# ---------------------------------------------------------------------------
# 3. 데이터셋 로드 및 전처리
# ---------------------------------------------------------------------------
print("[INFO] 캘리브레이션 데이터 로딩 중...")
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]", trust_remote_code=True)

def preprocess(example): #공백 예외처리
    try:
        text = ""
        if "conversations" in example: #대화형태로 받는다면, 템플릿에 맞게 text를 저장
            text = tokenizer.apply_chat_template(example["conversations"], add_generation_prompt=True, tokenize=False)
        elif "text" in example: #공백을 받는 다면, 
            text = example["text"]
        
        return tokenizer(text, truncation=True, max_length=MAX_SEQUENCE_LENGTH)
    except:
        return tokenizer("", truncation=True, max_length=MAX_SEQUENCE_LENGTH)

ds = ds.map(preprocess, remove_columns=ds.column_names)
ds = ds.filter(lambda x: len(x["input_ids"]) > 0)
print(f"[INFO] 데이터 전처리 완료 (샘플 수: {len(ds)})")

# ---------------------------------------------------------------------------
# 4. 양자화 실행
# ---------------------------------------------------------------------------
print(f"[INFO] 양자화 시작...")
recipe = [QuantizationModifier(scheme=SCHEME, targets=TARGETS, ignore=IGNORE)]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# ---------------------------------------------------------------------------
# 5. 모델 저장 (Colab 로컬 -> 압축 -> 드라이브 이동)
# ---------------------------------------------------------------------------
# 1) Colab 로컬에 저장 (빠름)
print(f"[INFO] 임시 저장 중 (Local): {LOCAL_OUT_DIR}")
if os.path.exists(LOCAL_OUT_DIR): shutil.rmtree(LOCAL_OUT_DIR)
model.save_pretrained(LOCAL_OUT_DIR, save_compressed=True)
tokenizer.save_pretrained(LOCAL_OUT_DIR)

# 2) 압축 (Zip)
zip_name = "fp8_quantized_model"
print(f"[INFO] 압축 중: {zip_name}.zip")
shutil.make_archive(zip_name, 'zip', LOCAL_OUT_DIR)

# 3) 구글 드라이브로 이동
print(f"[INFO] 구글 드라이브로 복사 중: {DRIVE_SAVE_DIR}")
if not os.path.exists(DRIVE_SAVE_DIR):
    os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)

# shutil.move는 덮어쓰기 에러가 날 수 있어 copy 사용
target_zip_path = os.path.join(DRIVE_SAVE_DIR, f"{zip_name}.zip")
shutil.copy(f"{zip_name}.zip", target_zip_path)

print(f"\n[SUCCESS] 모든 작업 완료!")
print(f"구글 드라이브 위치: {target_zip_path}")
#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"
SPLIT="${SPLIT:-both}"
OUTPUT_ROOT="${OUTPUT_ROOT:-p3_decoding_results}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
TORCH_PACKAGES="${TORCH_PACKAGES:-torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  echo "PyTorch를 설치합니다: $TORCH_INDEX_URL"
  pip install --timeout 120 --retries 10 --no-cache-dir $TORCH_PACKAGES --index-url "$TORCH_INDEX_URL"
fi

pip install -r requirements.txt
python download_checkpoint.py

GPU_FLAG=""
if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  GPU_FLAG="--use-gpu"
  echo "CUDA가 감지되었습니다. GPU로 실행합니다."
else
  echo "CUDA가 감지되지 않았습니다. CPU로 실행하며 매우 오래 걸릴 수 있습니다."
fi

python run_p3_decoding_experiment.py \
  --experiment cmu_balanced_n30 \
  --checkpoint checkpoints/p3_selected_checkpoint.pt \
  --output-root "$OUTPUT_ROOT" \
  --split "$SPLIT" \
  $GPU_FLAG

python run_p3_decoding_experiment.py \
  --experiment rerank_balanced_n30 \
  --checkpoint checkpoints/p3_selected_checkpoint.pt \
  --output-root "$OUTPUT_ROOT" \
  --split "$SPLIT" \
  $GPU_FLAG

mkdir -p predictions
for EXP in cmu_balanced_n30 rerank_balanced_n30; do
  mkdir -p "predictions/$EXP"
  cp "$OUTPUT_ROOT/$EXP"/generated_*.txt "predictions/$EXP"/
  cp "$OUTPUT_ROOT/$EXP"/metrics.json "predictions/$EXP"/
  cp "$OUTPUT_ROOT/$EXP"/generation.log "predictions/$EXP"/
  if [ -f "$OUTPUT_ROOT/$EXP/ridge_model.json" ]; then
    cp "$OUTPUT_ROOT/$EXP/ridge_model.json" "predictions/$EXP"/
  fi
done

if [ "$SPLIT" = "dev" ] || [ "$SPLIT" = "both" ]; then
  python evaluate_p3_outputs.py --split dev
fi
if [ "$SPLIT" = "test" ] || [ "$SPLIT" = "both" ]; then
  python evaluate_p3_outputs.py --split test
fi

echo "완료되었습니다. 결과는 $OUTPUT_ROOT 및 predictions/ 아래에 저장되었습니다."

# GPT2-XL P3 Prefix + CMU/Ridge Decoding 제출 코드

이 폴더는 과제 필수 파일과 Part-II 소넷 생성 실험 코드를 함께 포함한다. 최종 실험은 P3 prefix-tuning checkpoint를 기반으로 하며, 다음 두 방법을 재현할 수 있다.

- CMU 발음 사전을 이용한 rhyme-aware decoding, `num_candidates=30`
- CMU 후보 생성 후 Ridge regression reranker 적용, `num_candidates=30`, `rerank_candidates=3`
- 여기서, team-9의 최종 모델은 GPT2-XL P3 Prefix + CMU(num_candidates=30) + Ridge regression reranker 이다.

## 포함된 파일

필수 파일:

- `models/base_gpt.py`
- `models/gpt2.py`
- `modules/attention.py`
- `modules/gpt2_layer.py`
- `config.py`
- `sanity_check.py`
- `optimizer.py`
- `optimizer_test.py`
- `optimizer_test.npy`
- `classifier.py`
- `evaluation.py`
- `utils.py`
- `datasets.py`
- `data/`
- `predictions/`

Part-II 소넷 생성 및 평가 파일:

- `sonnet_generation.py`
- `rhyme_utils.py`
- `rhyme_decoding.py`
- `ridge_reranker.py`
- `run_p3_decoding_experiment.py`
- `evaluate_p3_outputs.py`
- `download_checkpoint.py`
- `run_final_experiments.sh`
- `run_final_experiments.ps1`

## Checkpoint 준비

P3 prefix checkpoint는 약 300MB라 GitHub repository에 직접 포함하지 않는다. 실행 스크립트에 현재 자동으로 다운로드 받아지도록 되어있지만, 작동하지 않을 경우를 대비한 실제 링크이다.

```text
https://drive.google.com/file/d/1aNs-Y8EqPyKzIx-KhoddaYlMZKFwD50g/view?usp=sharing
```

다운로드된 checkpoint는 아래 위치에 저장되어야 한다.

```text
checkpoints/p3_selected_checkpoint.pt
```

사용한 checkpoint는 다음 prefix-tuning 실험의 selected checkpoint이다.

```text
P3_prefix20_hidden512_lr2e-4
```

## 한 번에 실행하기

Linux 또는 클라우드 환경:

```bash
bash run_final_experiments.sh
```

Windows PowerShell:

```powershell
.\run_final_experiments.ps1
```

위 스크립트는 다음 과정을 자동으로 수행한다.

1. `.venv` 가상환경 생성
2. PyTorch가 없으면 CUDA 12.4용 PyTorch 설치
3. `requirements.txt`의 나머지 라이브러리 설치
4. Google Drive에서 checkpoint 다운로드
5. `cmu_balanced_n30` 생성 실행
6. `rerank_balanced_n30` 생성 실행 및 Ridge reranker 학습
7. 결과를 `predictions/` 아래로 복사
8. `predictions/dev_metrics.csv`, `predictions/test_metrics.csv` 생성

기본값은 dev와 test를 모두 실행한다. test만 실행하려면 다음처럼 실행한다.

```bash
SPLIT=test bash run_final_experiments.sh
```

PowerShell에서는 다음처럼 실행한다.

```powershell
$env:SPLIT="test"
.\run_final_experiments.ps1
```

## CUDA 버전이 맞지 않을 때

기본 PyTorch wheel은 CUDA 12.4 기준이다. CUDA 12.1 환경이라면 다음처럼 실행한다.

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 \
TORCH_PACKAGES="torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1" \
bash run_final_experiments.sh
```

PowerShell에서는 다음처럼 설정한다.

```powershell
$env:TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
$env:TORCH_PACKAGES="torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
.\run_final_experiments.ps1
```

CPU 환경에서도 실행은 가능하지만 GPT2-XL 생성이 매우 느리므로 GPU 사용을 권장한다.

## 개별 실행 방법

CMU n=30만 실행:

```bash
python run_p3_decoding_experiment.py \
  --experiment cmu_balanced_n30 \
  --checkpoint checkpoints/p3_selected_checkpoint.pt \
  --use-gpu
```

CMU n=30 + Ridge reranker 실행:

```bash
python run_p3_decoding_experiment.py \
  --experiment rerank_balanced_n30 \
  --checkpoint checkpoints/p3_selected_checkpoint.pt \
  --use-gpu
```

출력은 기본적으로 다음 위치에 저장된다.

```text
p3_decoding_results/<experiment_name>/
```

각 실험 폴더에는 다음 파일이 생성된다.

- `generated_dev.txt`
- `generated_test.txt`
- `metrics.json`
- `generation.log`
- `ridge_model.json`, Ridge 실험에서만 생성

## 평가 실행

`predictions/` 아래의 생성 결과를 평가하려면 다음을 실행한다.

```bash
python evaluate_p3_outputs.py --split dev
python evaluate_p3_outputs.py --split test
```

생성되는 파일:

```text
predictions/dev_metrics.csv
predictions/test_metrics.csv
```

평가 지표:

- `chrF`
- `Rhyme Acc`
- `Repetition`
- `Gen Time`, 로그나 metrics 파일에 시간이 남아 있는 경우

최종 보고서에는 G-Eval 평가지표도 있으나, API 토큰 비용 관계상 생략

## 최종 실험 설정

```text
model_size = gpt2-xl
temperature = 0.7
top_p = 0.85
num_candidates = 30
rerank_candidates = 1, CMU-only
rerank_candidates = 3, CMU + Ridge
ridge_train_candidates = 3, CMU + Ridge
max_length = 180
target_lines = 14
seed = 11711
```
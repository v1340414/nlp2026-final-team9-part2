$ErrorActionPreference = "Stop"

$PythonBin = if ($env:PYTHON_BIN) { $env:PYTHON_BIN } else { "python" }
$VenvDir = if ($env:VENV_DIR) { $env:VENV_DIR } else { ".venv" }
$Split = if ($env:SPLIT) { $env:SPLIT } else { "both" }
$OutputRoot = if ($env:OUTPUT_ROOT) { $env:OUTPUT_ROOT } else { "p3_decoding_results" }
$TorchIndexUrl = if ($env:TORCH_INDEX_URL) { $env:TORCH_INDEX_URL } else { "https://download.pytorch.org/whl/cu124" }
$TorchPackages = if ($env:TORCH_PACKAGES) { $env:TORCH_PACKAGES } else { "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0" }

if (-not (Test-Path $VenvDir)) {
  & $PythonBin -m venv $VenvDir
}

& ".\$VenvDir\Scripts\Activate.ps1"
python -m pip install --upgrade pip

$torchInstalled = $true
python -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('torch') else 1)"
if ($LASTEXITCODE -ne 0) {
  $torchInstalled = $false
}
if (-not $torchInstalled) {
  Write-Host "PyTorch를 설치합니다: $TorchIndexUrl"
  $TorchPackageArgs = $TorchPackages -split " "
  pip install --timeout 120 --retries 10 --no-cache-dir @TorchPackageArgs --index-url $TorchIndexUrl
}

pip install -r requirements.txt
python download_checkpoint.py

$gpuFlag = @()
python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -eq 0) {
  $gpuFlag = @("--use-gpu")
  Write-Host "CUDA가 감지되었습니다. GPU로 실행합니다."
} else {
  Write-Host "CUDA가 감지되지 않았습니다. CPU로 실행하며 매우 오래 걸릴 수 있습니다."
}

python run_p3_decoding_experiment.py `
  --experiment cmu_balanced_n30 `
  --checkpoint checkpoints/p3_selected_checkpoint.pt `
  --output-root $OutputRoot `
  --split $Split `
  @gpuFlag

python run_p3_decoding_experiment.py `
  --experiment rerank_balanced_n30 `
  --checkpoint checkpoints/p3_selected_checkpoint.pt `
  --output-root $OutputRoot `
  --split $Split `
  @gpuFlag

New-Item -ItemType Directory -Force -Path predictions | Out-Null
foreach ($Exp in @("cmu_balanced_n30", "rerank_balanced_n30")) {
  New-Item -ItemType Directory -Force -Path "predictions\$Exp" | Out-Null
  Copy-Item "$OutputRoot\$Exp\generated_*.txt" "predictions\$Exp\" -Force
  Copy-Item "$OutputRoot\$Exp\metrics.json" "predictions\$Exp\" -Force
  Copy-Item "$OutputRoot\$Exp\generation.log" "predictions\$Exp\" -Force
  if (Test-Path "$OutputRoot\$Exp\ridge_model.json") {
    Copy-Item "$OutputRoot\$Exp\ridge_model.json" "predictions\$Exp\" -Force
  }
}

if ($Split -eq "dev" -or $Split -eq "both") {
  python evaluate_p3_outputs.py --split dev
}
if ($Split -eq "test" -or $Split -eq "both") {
  python evaluate_p3_outputs.py --split test
}

Write-Host "완료되었습니다. 결과는 $OutputRoot 및 predictions/ 아래에 저장되었습니다."

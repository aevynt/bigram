param(
    [string]$Config = "configs\tensor1_48gb.json",
    [string]$OutDir = "checkpoints\tensor1_48gb",
    [string]$TrainData = "data\train",
    [string]$ValData = "data\val",
    [string]$Tokenizer = "data\tokenizer.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

.\.venv\Scripts\Activate.ps1
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:TOKENIZERS_PARALLELISM = "false"
$env:CUDA_LAUNCH_BLOCKING = "0"

$cuda = python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($cuda -ne "1") {
    throw "CUDA không khả dụng. Cài PyTorch CUDA build trước khi train Tensor 1."
}

New-Item -ItemType Directory -Force -Path logs | Out-Null
try {
    python scripts\train.py --train-data $TrainData --val-data $ValData --tokenizer $Tokenizer --config $Config --out-dir $OutDir 2>&1 |
        Tee-Object -FilePath logs\train_tensor1_48gb.log
} catch {
    Write-Warning "Training lỗi. Nếu là CUDA OOM: dùng configs\tensor1_48gb_safe.json, giảm max_seq_len, giảm mean_recurrence, tăng grad_accum hoặc giữ compile_model=false."
    throw
}

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

python --version
$versionText = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$version = [version]$versionText
if (($version -lt [version]"3.10") -or ($version -ge [version]"3.12")) {
    throw "Python 3.10 hoặc 3.11 là bắt buộc. Đang có: $versionText"
}

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
if (Test-Path "requirements-server.txt") {
    pip install -r requirements-server.txt
}

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
$cuda = python -c "import torch; print('1' if torch.cuda.is_available() else '0')"
if ($cuda -ne "1") {
    Write-Warning "Torch CUDA chưa khả dụng. Cài PyTorch CUDA build từ pytorch.org khớp NVIDIA driver trước khi train Tensor 1."
}

New-Item -ItemType Directory -Force -Path data, data\rag_sources, data\rag_index, checkpoints, logs | Out-Null

Write-Host "Setup xong."
Write-Host "Next:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\prepare_tensor1.ps1"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\smoke_test.ps1"

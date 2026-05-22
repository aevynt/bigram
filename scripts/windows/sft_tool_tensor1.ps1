param(
    [string]$Data = "data\tool_sft.jsonl",
    [string]$ValData = "data\tool_sft_val.jsonl",
    [string]$Tokenizer = "data\tokenizer.json",
    [string]$Init = "checkpoints\tensor1_48gb\ckpt_final.pt",
    [string]$Config = "configs\tensor1_48gb.json",
    [string]$OutDir = "checkpoints\tensor1_tool_sft"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

.\.venv\Scripts\Activate.ps1
foreach ($path in @($Data, $Tokenizer, $Init, $Config)) {
    if (-not (Test-Path $path)) {
        throw "Thiếu file bắt buộc: $path"
    }
}
New-Item -ItemType Directory -Force -Path logs | Out-Null
python scripts\train_tool_sft.py --data $Data --val-data $ValData --tokenizer $Tokenizer --init $Init --config $Config --out-dir $OutDir 2>&1 |
    Tee-Object -FilePath logs\sft_tool_tensor1.log

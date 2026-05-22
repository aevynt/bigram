param(
    [string]$InputDir = "data\rag_sources",
    [string]$Output = "data\rag_index.jsonl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

.\.venv\Scripts\Activate.ps1
python scripts\build_rag_index.py --input-dir $InputDir --output $Output
if (Test-Path $Output) {
    python scripts\rag_search.py --index $Output --query "Điều lệ Đảng" --top-k 1
}

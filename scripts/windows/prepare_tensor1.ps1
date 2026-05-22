Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
} else {
    Write-Warning "Không thấy .venv. Chạy scripts\windows\setup_server.ps1 trước."
}

if ((-not (Test-Path "data\tokenizer.json")) -and (Test-Path "data\corpus.txt")) {
    python scripts\train_tokenizer.py --input data\corpus.txt --output data\tokenizer.json --vocab-size 64000 --min-frequency 2
}

if (-not (Test-Path "data\tokenizer.json")) {
    Write-Warning "Thiếu data\tokenizer.json. Bỏ qua prepare train/val."
} else {
    if (Test-Path "data\train.txt") {
        python scripts\prepare_data.py --tokenizer data\tokenizer.json --input data\train.txt --output-prefix data\train
    } else {
        Write-Warning "Thiếu data\train.txt"
    }
    if (Test-Path "data\val.txt") {
        python scripts\prepare_data.py --tokenizer data\tokenizer.json --input data\val.txt --output-prefix data\val
    } else {
        Write-Warning "Thiếu data\val.txt"
    }
}

$ragFiles = @()
if (Test-Path "data\rag_sources") {
    $ragFiles = Get-ChildItem -Path "data\rag_sources" -File -Recurse -Include *.txt,*.md,*.jsonl
}
if ($ragFiles.Count -gt 0) {
    python scripts\build_rag_index.py --input-dir data\rag_sources --output data\rag_index.jsonl
} else {
    Write-Warning "data\rag_sources chưa có .txt/.md/.jsonl nên bỏ qua RAG index."
}

param(
    [int]$Port = 8000,
    [string]$HostAddress = "0.0.0.0",
    [string]$Config = "configs\tensor1_inference.json",
    [string]$Tokenizer = "data\tokenizer.json",
    [string]$Checkpoint = "checkpoints\tensor1_tool_sft\ckpt_final.pt",
    [string]$RagIndex = "data\rag_index.jsonl"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

.\.venv\Scripts\Activate.ps1
$env:BIGRAM_CONFIG = $Config
$env:BIGRAM_TOKENIZER = $Tokenizer
$env:BIGRAM_CHECKPOINT = $Checkpoint
$env:BIGRAM_RAG_INDEX = $RagIndex

uvicorn server.app:app --host $HostAddress --port $Port

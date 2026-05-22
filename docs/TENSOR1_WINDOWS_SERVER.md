# Bigram Tensor 1 on Windows Server + RTX A6000/Blackwell 48GB

## 1. Requirements

- Windows Server
- NVIDIA driver installed
- Python 3.10 hoặc 3.11
- Git
- PowerShell 7 recommended
- PyTorch CUDA build
- 1 GPU 48GB VRAM, ví dụ RTX A6000 hoặc Blackwell-class 48GB

Không cần Docker. Không cần WSL.

## 2. Clone

```powershell
git clone https://github.com/aevynt/bigram
cd bigram
```

## 3. Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\setup_server.ps1
```

Nếu script báo CUDA chưa khả dụng, cài PyTorch CUDA wheel từ https://pytorch.org/get-started/locally/ khớp driver NVIDIA rồi chạy lại smoke test.

## 4. Prepare Data

Expected files:

- `data\corpus.txt` optional, dùng train tokenizer
- `data\train.txt`
- `data\val.txt`
- `data\tool_sft.jsonl` optional
- `data\tool_sft_val.jsonl` optional
- `data\rag_sources\*.txt` hoặc `*.md`

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\prepare_tensor1.ps1
```

## 5. Train 1B on 48GB

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\train_tensor1_48gb.ps1
```

Nếu OOM:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\train_tensor1_48gb.ps1 -Config configs\tensor1_48gb_safe.json
```

Config 48GB dùng `batch_size=1`, `grad_accum_steps=128`, mixed precision, gradient checkpointing, MoE chỉ ở recurrent core, `max_seq_len=4096`, `mean_recurrence=24`, `backprop_depth=4`.

## 6. Tool SFT

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\sft_tool_tensor1.ps1
```

Tool-SFT JSONL dùng conversation có assistant tool call:

```json
{"messages":[{"role":"user","content":"Tra cứu X"},{"role":"assistant","content":"<tool_call>{\"tool\":\"rag.search\",\"args\":{\"query\":\"X\"}}</tool_call>"},{"role":"tool","content":"<tool_result>...</tool_result>"},{"role":"assistant","content":"Câu trả lời có nguồn."}]}
```

## 7. Build RAG

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\build_rag.ps1
```

CLI trực tiếp:

```powershell
python scripts\build_rag_index.py --input-dir data\rag_sources --output data\rag_index.jsonl
python scripts\rag_search.py --index data\rag_index.jsonl --query "Điều lệ Đảng" --top-k 5
```

## 8. Run Server

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\run_server.ps1
```

Server vẫn start nếu thiếu checkpoint/tokenizer. `/health` vẫn hoạt động, còn `/generate` và `/agent/chat` trả lỗi rõ `model not loaded`.

## 9. API Examples

```powershell
Invoke-RestMethod http://localhost:8000/health
```

```powershell
Invoke-RestMethod http://localhost:8000/generate `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"prompt":"Thủ đô Việt Nam là","max_new_tokens":64,"recurrence":32}'
```

```powershell
Invoke-RestMethod http://localhost:8000/agent/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"messages":[{"role":"user","content":"Tóm tắt tài liệu trong RAG"}],"max_tool_turns":6,"recurrence":64,"max_new_tokens":512}'
```

```powershell
Invoke-RestMethod http://localhost:8000/rag/search `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"query":"Điều lệ Đảng","top_k":5}'
```

## 10. Notes

- No Docker.
- No WSL required.
- Fact about ĐCS/politics should use RAG/verifier.
- 1x 48GB đủ cho experiments và serious SFT/continued pretrain, không thực tế cho full 4T-token pretrain từ đầu.
- Route tốt nhất: 100B-300B clean continued pretrain, 1B-5B tool SFT, calibration, rồi RAG.
- `terminal.run` chạy PowerShell với block list nguy hiểm, nhưng không phải sandbox bảo mật hoàn chỉnh. Không expose endpoint agent ra internet nếu chưa có auth/rate limit.

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (Test-Path ".venv\Scripts\Activate.ps1") {
    .\.venv\Scripts\Activate.ps1
}

python -c "import bigram; cfg=bigram.tensor1_config(); print('tensor1', cfg.model.hidden_size, cfg.model.moe_scope)"
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
python -c "from bigram.tools import ToolRegistry, CalculatorTool; r=ToolRegistry(); r.register(CalculatorTool()); out=r.run('calculator', {'expression':'2+3*4'}); assert out.ok and out.output=='14'; print('calculator ok')"
python -c "from bigram.agent.schema import render_tool_call, parse_tool_call; s=render_tool_call('rag.search', {'query':'x'}); c=parse_tool_call(s); assert c and c.tool=='rag.search'; print('schema ok')"
python -c "import tempfile, pathlib; from bigram.rag import build_index, lexical_search; d=tempfile.TemporaryDirectory(); p=pathlib.Path(d.name); (p/'a.md').write_text('Điều lệ Đảng quy định nguyên tắc tổ chức.', encoding='utf-8'); out=p/'idx.jsonl'; build_index(p,out); hits=lexical_search(out,'Điều lệ Đảng',1); assert hits; print('rag ok')"
python -c "import server.app; print('server import ok')"

try {
    $health = Invoke-WebRequest http://localhost:8000/health -UseBasicParsing -TimeoutSec 2
    Write-Host $health.Content
} catch {
    Write-Warning "Server localhost:8000 chưa chạy, bỏ qua health HTTP."
}

"""Local tools implemented with Windows-safe defaults."""

import ast
import operator
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from .base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    name = "calculator"
    _ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(self, node):
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.operand))
        raise ValueError("unsupported expression")

    def run(self, args: dict) -> ToolResult:
        expr = str(args.get("expression", args.get("query", "")))
        tree = ast.parse(expr, mode="eval")
        return ToolResult(ok=True, output=str(self._eval(tree)))


class MarkdownCheckTool(BaseTool):
    name = "markdown.check"

    def run(self, args: dict) -> ToolResult:
        text = str(args.get("text", ""))
        issues = []
        if text.count("```") % 2:
            issues.append("unclosed code fence")
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") and not stripped.startswith("# "):
                issues.append(f"line {lineno}: suspicious heading spacing")
            if "|" in line and not stripped.startswith("|") and line.count("|") == 1:
                issues.append(f"line {lineno}: suspicious table pipe count")
        return ToolResult(ok=not issues, output="\n".join(issues) if issues else "ok")


class PythonRunTool(BaseTool):
    name = "python.run"

    def run(self, args: dict) -> ToolResult:
        code = str(args.get("code", ""))
        timeout = float(args.get("timeout", 20))
        cwd = args.get("cwd") or os.getcwd()
        with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as f:
            f.write(code)
            path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, path],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output = (proc.stdout or "") + (proc.stderr or "")
            return ToolResult(
                ok=proc.returncode == 0,
                output=output,
                error=None if proc.returncode == 0 else output,
                metadata={"exit_code": proc.returncode},
            )
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


class TerminalRunTool(BaseTool):
    name = "terminal.run"
    blocked = [
        "Remove-Item -Recurse -Force C:\\",
        "del /s /q C:\\",
        "format",
        "shutdown",
        "Restart-Computer",
        "Stop-Computer",
        "diskpart",
        "bcdedit",
        "cipher /w",
        "rm -rf",
        "mkfs",
        "dd if=",
        "sudo",
        ":(){ :|:& };:",
    ]

    def __init__(self, safe_mode=True):
        self.safe_mode = safe_mode

    def _is_blocked(self, command: str) -> bool:
        lowered = command.lower()
        return any(pattern.lower() in lowered for pattern in self.blocked)

    def run(self, args: dict) -> ToolResult:
        command = str(args.get("command", ""))
        timeout = float(args.get("timeout", 30))
        cwd = args.get("cwd") or os.getcwd()
        if self.safe_mode and self._is_blocked(command):
            return ToolResult(ok=False, output="", error="blocked dangerous command")
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        return ToolResult(
            ok=proc.returncode == 0,
            output=output,
            error=None if proc.returncode == 0 else output,
            metadata={"exit_code": proc.returncode},
        )


class CodeSearchTool(BaseTool):
    name = "code.search"

    def run(self, args: dict) -> ToolResult:
        query = str(args.get("query", ""))
        root = Path(args.get("root", ".")).resolve()
        max_results = int(args.get("max_results", 20))
        if not query:
            return ToolResult(ok=False, output="", error="query is required")
        if shutil.which("rg"):
            proc = subprocess.run(
                ["rg", "--line-number", "--fixed-strings", query, str(root)],
                capture_output=True,
                text=True,
                timeout=20,
            )
            lines = (proc.stdout or "").splitlines()[:max_results]
            return ToolResult(ok=bool(lines), output="\n".join(lines), metadata={"engine": "rg"})

        hits = []
        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                path = Path(dirpath) / filename
                try:
                    for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                        if query in line:
                            hits.append(f"{path}:{lineno}:{line}")
                            if len(hits) >= max_results:
                                return ToolResult(ok=True, output="\n".join(hits), metadata={"engine": "python"})
                except OSError:
                    continue
        return ToolResult(ok=bool(hits), output="\n".join(hits), metadata={"engine": "python"})

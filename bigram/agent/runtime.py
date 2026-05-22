"""Minimal tool loop runtime for Bigram Tensor 1."""

from .schema import Message, parse_tool_call, render_tool_result, is_abstain


class AgentRuntime:
    def __init__(self, model, tokenizer, tool_registry, device, max_tool_turns=6):
        self.model = model
        self.tokenizer = tokenizer
        self.tool_registry = tool_registry
        self.device = device
        self.max_tool_turns = max_tool_turns

    def _render_messages(self, messages):
        rendered = []
        for msg in messages:
            role = msg.role if isinstance(msg, Message) else msg.get("role", "user")
            content = msg.content if isinstance(msg, Message) else msg.get("content", "")
            rendered.append(f"<{role}>{content}</{role}>")
        rendered.append("<assistant>")
        return "\n".join(rendered)

    def _generate_text(self, prompt, generation_kwargs):
        if hasattr(self.model, "generate_text"):
            return self.model.generate_text(prompt, **generation_kwargs)
        tok, tone = self.tokenizer.encode(prompt, add_special=True)
        import torch

        token_ids = torch.tensor([tok], dtype=torch.long, device=self.device)
        tone_ids = torch.tensor([tone], dtype=torch.long, device=self.device)
        out_ids, out_tones, _ = self.model.generate(
            token_ids,
            tone_ids,
            max_new_tokens=generation_kwargs.get("max_new_tokens", 256),
            num_recurrence=generation_kwargs.get("recurrence"),
            temperature=generation_kwargs.get("temperature", 0.7),
            top_k=generation_kwargs.get("top_k", 40),
            top_p=generation_kwargs.get("top_p", 0.9),
            repetition_penalty=generation_kwargs.get("repetition_penalty", 1.1),
            abstention_threshold=generation_kwargs.get("abstention_threshold"),
        )
        return self.tokenizer.decode(out_ids[0].tolist(), out_tones[0].tolist())

    def chat(self, messages, generation_kwargs=None):
        generation_kwargs = generation_kwargs or {}
        working = list(messages)
        trace = []
        max_turns = int(generation_kwargs.pop("max_tool_turns", self.max_tool_turns))

        for _ in range(max_turns + 1):
            text = self._generate_text(self._render_messages(working), generation_kwargs)
            call = parse_tool_call(text)
            if call is None:
                return {"answer": text, "tool_trace": trace, "abstained": is_abstain(text)}

            result = self.tool_registry.run(call.tool, call.args)
            trace.append({
                "tool": call.tool,
                "args": call.args,
                "ok": result.ok,
                "error": result.error,
                "metadata": result.metadata,
            })
            working.append({"role": "assistant", "content": text})
            working.append({"role": "tool", "content": render_tool_result(result.output if result.ok else result.error or "")})

        answer = "Không đủ căn cứ: đã vượt quá số lượt gọi tool cho phép."
        return {"answer": answer, "tool_trace": trace, "abstained": True}

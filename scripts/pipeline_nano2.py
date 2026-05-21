"""
pipeline_nano2.py
=================
Pipeline đầy đủ: Model A (router + reasoner) -> tool -> Model B (synthesizer)

Cách dùng:
    python scripts/pipeline_nano2.py
    python scripts/pipeline_nano2.py --model-a <path> --model-b <path>
"""

import argparse
import json
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bigram import BigramModel, BigramTokenizer
from bigram.config import ModelConfig
from scripts.postprocess import fix_output


def load_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"]["model"])
    model = BigramModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens=80,
    temperature=0.3,
    top_k=10,
    recurrence=4,
    device="cpu",
) -> str:
    tok, tone = tokenizer.encode(prompt, add_special=False)
    bos = tokenizer.token_to_id("<bos>")
    tok = [bos] + tok
    tone = [0] + tone

    prompt_len = len(tok)
    token_ids = torch.tensor([tok], dtype=torch.long, device=device)
    tone_ids = torch.tensor([tone], dtype=torch.long, device=device)

    out_ids, out_tones, _ = model.generate(
        token_ids,
        tone_ids,
        max_new_tokens=max_new_tokens,
        num_recurrence=recurrence,
        temperature=temperature,
        top_k=top_k,
    )

    new_ids = out_ids[0, prompt_len:].tolist()
    new_tones = out_tones[0, prompt_len:].tolist()

    eos = tokenizer.token_to_id("<eos>")
    pad = tokenizer.token_to_id("<pad>")
    special = {tokenizer.token_to_id("<bos>"), eos, pad}

    if eos in new_ids:
        cut = new_ids.index(eos)
        new_ids = new_ids[:cut]
        new_tones = new_tones[:cut]

    filtered_ids, filtered_tones = [], []
    for tid, tn in zip(new_ids, new_tones):
        if tid not in special:
            filtered_ids.append(tid)
            filtered_tones.append(tn)

    if not filtered_ids:
        return ""
    return tokenizer.decode(filtered_ids, filtered_tones).strip()


def parse_model_a_output(raw: str):
    """
    Trả về dict:
      think      : nội dung <think> nếu có, None nếu không
      tool_call  : dict JSON nếu có <tool_call>, None nếu không
      response   : phần text sau think/tool_call nếu trả lời thẳng
    """
    text = fix_output(raw)

    think = None
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()

    tool_call = None
    tc_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if tc_match:
        try:
            tool_call = json.loads(tc_match.group(1))
        except json.JSONDecodeError:
            tool_call = None

    response = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    response = re.sub(r"<tool_call>.*?</tool_call>", "", response, flags=re.DOTALL)
    response = response.strip()

    return {"think": think, "tool_call": tool_call, "response": response}


FAKE_RESULTS = {
    "search": lambda q: f"Kết quả tìm kiếm cho '{q}': đây là thông tin giả lập.",
    "calculate": {
        "quadratic": lambda e: {"x1": 2.0, "x2": 3.0},
        "multiply": lambda e: 42.0,
        "linear_system": lambda e: {"x": 2.0, "y": 1.0},
        "word_problem": lambda e: {"result": "6", "steps": ["bước 1", "bước 2"]},
    },
}


def run_tool(tool_call: dict) -> str:
    name = tool_call.get("name", "")
    args = tool_call.get("arguments", {})

    if name == "search":
        query = args.get("query", "")
        result = FAKE_RESULTS["search"](query)
        return json.dumps(
            {"name": "search", "query": query, "result": result},
            ensure_ascii=False,
        )

    if name == "calculate":
        calc_type = args.get("type", "multiply")
        expr = args.get("expression", "")
        handler = FAKE_RESULTS["calculate"].get(
            calc_type,
            FAKE_RESULTS["calculate"]["multiply"],
        )
        result = handler(expr)
        return json.dumps(
            {"name": "calculate", "type": calc_type, "expression": expr, "result": result},
            ensure_ascii=False,
        )

    return json.dumps({"error": f"Tool '{name}' không được hỗ trợ"}, ensure_ascii=False)


def run_pipeline(user_input: str, model_a, model_b, tokenizer, device="cpu") -> dict:
    raw_a = generate(model_a, tokenizer, user_input, device=device)
    parsed = parse_model_a_output(raw_a)

    result = {"think": parsed["think"], "action": None, "response": None}

    if parsed["tool_call"]:
        result["action"] = parsed["tool_call"]
        tool_result = run_tool(parsed["tool_call"])

        prompt_b = (
            f"<tool_result>\n{tool_result}\n</tool_result>\n\n"
            f"Câu hỏi gốc: {user_input}"
        )
        result["response"] = generate(model_b, tokenizer, prompt_b, device=device)
    else:
        result["action"] = "direct"
        result["response"] = parsed["response"] or raw_a

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-a", default="nano2/model_a/sft/ckpt_final.pt")
    parser.add_argument("--model-b", default="nano2/model_b/sft/ckpt_final.pt")
    parser.add_argument("--tokenizer", default="nano2/tokenizer.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Đang nạp Model A...")
    model_a = load_model(args.model_a, device)
    print("Đang nạp Model B...")
    model_b = load_model(args.model_b, device)
    tokenizer = BigramTokenizer.load(args.tokenizer)

    print("\n" + "=" * 50)
    print("  Bigram Nano 2 Pipeline")
    print("  Gõ 'thoát' để dừng")
    print("=" * 50 + "\n")

    while True:
        try:
            user = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            break

        if not user:
            continue
        if user.lower() in ("thoát", "quit", "exit"):
            print("Nano 2: Tạm biệt!")
            break

        result = run_pipeline(user, model_a, model_b, tokenizer, device)

        if result["think"]:
            print(f"  [think] {result['think']}")
        if result["action"] and result["action"] != "direct":
            print(f"  [tool]  {result['action']}")
        print(f"Nano 2: {result['response']}\n")


if __name__ == "__main__":
    main()

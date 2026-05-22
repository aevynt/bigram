"""Tool-SFT dataset for Bigram Tensor 1."""

import json
import re

import torch
from torch.utils.data import Dataset

from bigram.model.tooling import TOOL_REGISTRY_DEFAULT
from bigram.agent.schema import parse_tool_call


ROLE_ORDER = ("system", "user", "assistant", "tool")
TOOL_TO_ID = {name: idx for idx, name in TOOL_REGISTRY_DEFAULT.items()}


class ToolSFTDataset(Dataset):
    """JSONL conversation dataset with coarse per-token tool targets."""

    def __init__(self, jsonl_path: str, tokenizer, block_size: int, ignore_index: int = -100):
        self.samples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.ignore_index = ignore_index
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(obj["messages"])

    def __len__(self):
        return len(self.samples)

    def _encode_messages(self, messages):
        token_ids = [self.tokenizer.token_to_id("<bos>")]
        tone_ids = [0]
        assistant_starts = []
        lm_mask = [False]
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            segment = f"<{role}>{content}</{role}>\n"
            seg_tok, seg_tone = self.tokenizer.encode(segment, add_special=False)
            start = len(token_ids)
            token_ids.extend(seg_tok)
            tone_ids.extend(seg_tone)
            lm_mask.extend([False] * len(seg_tok))
            if role == "assistant":
                # Target tại vị trí start-1 dự đoán token đầu tiên của assistant.
                assistant_starts.append((max(0, start - 1), content))
                for pos in range(max(0, start - 1), max(0, len(token_ids) - 1)):
                    if pos < len(lm_mask):
                        lm_mask[pos] = True
        token_ids.append(self.tokenizer.token_to_id("<eos>"))
        tone_ids.append(0)
        lm_mask.append(False)
        return token_ids, tone_ids, lm_mask, assistant_starts

    def __getitem__(self, idx):
        tok, tone, lm_mask, assistant_starts = self._encode_messages(self.samples[idx])
        pad_id = self.tokenizer.token_to_id("<pad>")
        need = self.block_size + 1
        if len(tok) > need:
            tok = tok[:need]
            tone = tone[:need]
            lm_mask = lm_mask[:need]
        else:
            pad_n = need - len(tok)
            tok = tok + [pad_id] * pad_n
            tone = tone + [0] * pad_n
            lm_mask = lm_mask + [False] * pad_n

        token_tensor = torch.tensor(tok, dtype=torch.int64)
        tone_tensor = torch.tensor(tone, dtype=torch.int64)
        mask_tensor = torch.tensor(lm_mask, dtype=torch.bool)
        x = token_tensor[:-1]
        t = tone_tensor[:-1]
        y = token_tensor[1:].clone()
        tone_y = tone_tensor[1:].clone()
        y[~mask_tensor[:-1]] = self.ignore_index
        tone_y[~mask_tensor[:-1]] = self.ignore_index
        y[x == pad_id] = self.ignore_index
        tone_y[x == pad_id] = self.ignore_index

        router_targets = torch.full((self.block_size,), self.ignore_index, dtype=torch.int64)
        name_targets = torch.full((self.block_size,), self.ignore_index, dtype=torch.int64)
        abstention_targets = torch.zeros(self.block_size, dtype=torch.float32)
        abstention_mask = torch.zeros(self.block_size, dtype=torch.float32)
        verifier_targets = torch.full((self.block_size,), self.ignore_index, dtype=torch.float32)

        # Alignment thô: gắn nhãn tool/verifier tại token đầu assistant turn.
        for pos, content in assistant_starts:
            if pos >= self.block_size:
                continue
            call = parse_tool_call(content)
            if call is not None:
                router_targets[pos] = 1
                name_targets[pos] = TOOL_TO_ID.get(call.tool, 0)
            elif re.search(r"không đủ căn cứ|khong du can cu|tôi không chắc", content, re.I):
                router_targets[pos] = 2
                abstention_targets[pos] = 1.0
                abstention_mask[pos] = 1.0
                verifier_targets[pos] = 0.0
            else:
                router_targets[pos] = 0
                verifier_targets[pos] = 1.0

        return {
            "token_ids": x,
            "tone_ids": t,
            "targets": y,
            "tone_targets": tone_y,
            "tool_router_targets": router_targets,
            "tool_name_targets": name_targets,
            "abstention_targets": abstention_targets,
            "abstention_mask": abstention_mask,
            "verifier_targets": verifier_targets,
        }

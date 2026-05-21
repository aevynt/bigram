"""
dataset.py
==========
Nạp dữ liệu huấn luyện cho Bigram.

Quy ước định dạng dữ liệu:
Sau khi tokenizer mã hóa toàn bộ corpus, ta lưu thành 2 file nhị phân song song:
  - <tên>.tok.bin  : mảng uint16, các token id nối liền nhau.
  - <tên>.tone.bin : mảng uint8,  các tone id tương ứng (cùng độ dài).

Việc lưu nhị phân (thay vì text/json) giúp nạp cực nhanh bằng numpy.memmap —
không cần đọc cả file vào RAM, hệ điều hành tự nạp phần cần thiết.

Lớp `PackedDataset` cắt corpus thành các đoạn (block) độ dài cố định để train
mô hình ngôn ngữ kiểu next-token-prediction.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset


class PackedDataset(Dataset):
    """
    Dataset cho pretraining: corpus được "đóng gói" liền mạch rồi cắt thành
    các block độ dài `block_size`.

    Mỗi mẫu trả về:
      token_ids : (block_size,)   — token đầu vào.
      tone_ids  : (block_size,)   — tone đầu vào.
      targets   : (block_size,)   — token đích = token_ids dịch trái 1 vị trí.
    """

    def __init__(self, token_bin: str, tone_bin: str, block_size: int):
        """
        token_bin : đường dẫn file .tok.bin (uint16).
        tone_bin  : đường dẫn file .tone.bin (uint8). Có thể None nếu không
                    dùng thông tin thanh điệu.
        block_size: độ dài mỗi đoạn (= max_seq_len khi train).
        """
        assert os.path.exists(token_bin), f"Không tìm thấy file: {token_bin}"
        self.block_size = block_size

        # memmap: ánh xạ file vào bộ nhớ ảo, không nạp hết vào RAM.
        self.tokens = np.memmap(token_bin, dtype=np.uint16, mode="r")

        self.tones = None
        if tone_bin is not None and os.path.exists(tone_bin):
            self.tones = np.memmap(tone_bin, dtype=np.uint8, mode="r")
            assert len(self.tones) == len(self.tokens), \
                "File token và tone phải có cùng độ dài"

        # Số block trọn vẹn có thể cắt được. Cần +1 token cho target -> -1.
        self.n_blocks = (len(self.tokens) - 1) // block_size
        assert self.n_blocks > 0, \
            "Dữ liệu quá ngắn so với block_size — cần thêm dữ liệu"

    def __len__(self):
        return self.n_blocks

    def __getitem__(self, idx):
        # Vị trí bắt đầu của block thứ idx.
        start = idx * self.block_size
        end = start + self.block_size

        # Lấy block + 1 token (token thừa dùng làm target cuối).
        # .astype(np.int64): PyTorch cần int64 cho chỉ số embedding.
        chunk = self.tokens[start:end + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1].copy())   # đầu vào.
        y = torch.from_numpy(chunk[1:].copy())    # đích (dịch trái 1).

        if self.tones is not None:
            # Lấy block + 1 tone, giống token: phần đầu là input, dịch trái là đích.
            tone_chunk = self.tones[start:end + 1].astype(np.int64)
            t = torch.from_numpy(tone_chunk[:-1].copy())       # tone đầu vào.
            tone_y = torch.from_numpy(tone_chunk[1:].copy())   # tone đích.
        else:
            # Không có tone -> trả về dãy 0 (tất cả <none>).
            t = torch.zeros(self.block_size, dtype=torch.int64)
            tone_y = torch.zeros(self.block_size, dtype=torch.int64)

        return {"token_ids": x, "tone_ids": t, "targets": y,
                "tone_targets": tone_y}


class JsonlSFTDataset(Dataset):
    """
    Dataset cho giai đoạn SFT (Supervised Fine-Tuning).

    Đọc file .jsonl, mỗi dòng là một object JSON:
        {"prompt": "<câu hỏi>", "response": "<câu trả lời tốt>"}

    Mã hóa thành chuỗi (prompt + response) và tạo target sao cho loss CHỈ tính
    trên phần response (phần prompt được gán nhãn ignore_index = -100).
    Như vậy model học cách TRẢ LỜI, không học cách lặp lại câu hỏi.
    """

    def __init__(self, jsonl_path: str, tokenizer, block_size: int,
                 ignore_index: int = -100):
        import json
        self.samples = []
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append((obj["prompt"], obj["response"]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]

        # Mã hóa riêng prompt và response để biết ranh giới.
        p_tok, p_tone = self.tokenizer.encode(prompt, add_special=False)
        r_tok, r_tone = self.tokenizer.encode(response, add_special=False)

        # Ghép: <bos> prompt response <eos>
        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        tok = [bos] + p_tok + r_tok + [eos]
        tone = [0] + p_tone + r_tone + [0]

        # Cắt / đệm cho đủ block_size + 1 (cần token thừa cho target).
        need = self.block_size + 1
        pad_id = self.tokenizer.token_to_id("<pad>")
        if len(tok) > need:
            tok = tok[:need]
            tone = tone[:need]
        else:
            pad_n = need - len(tok)
            tok = tok + [pad_id] * pad_n
            tone = tone + [0] * pad_n

        tok = torch.tensor(tok, dtype=torch.int64)
        tone = torch.tensor(tone, dtype=torch.int64)

        x = tok[:-1]
        t = tone[:-1]
        y = tok[1:].clone()
        tone_y = tone[1:].clone()  # tone đích — dịch trái 1, song song với y.

        # Mask phần prompt: target ở các vị trí prompt -> ignore_index.
        # Số vị trí prompt trong dãy target (đã dịch trái 1): len(bos+prompt)-1.
        prompt_len = 1 + len(p_tok)
        y[:prompt_len - 1] = self.ignore_index
        tone_y[:prompt_len - 1] = self.ignore_index
        # Các vị trí padding cũng bỏ qua.
        y[x == pad_id] = self.ignore_index
        tone_y[x == pad_id] = self.ignore_index

        return {"token_ids": x, "tone_ids": t, "targets": y,
                "tone_targets": tone_y}


class PreferenceDataset(Dataset):
    """
    Dataset cho giai đoạn DPO (Direct Preference Optimization).

    Đọc file .jsonl, mỗi dòng là một object JSON:
        {"prompt": "...", "chosen": "<câu trả lời tốt>", "rejected": "<câu kém>"}

    Mỗi mẫu trả về SÁU tensor — phiên bản đã token hóa của (prompt + chosen)
    và (prompt + rejected), kèm target đã mask phần prompt:
        chosen_token_ids,  chosen_tone_ids,  chosen_targets
        rejected_token_ids, rejected_tone_ids, rejected_targets

    DPO so sánh hai câu trả lời cho cùng một prompt nên cần cả hai cùng lúc.
    """

    def __init__(self, jsonl_path, tokenizer, block_size, ignore_index=-100):
        import json
        self.samples = []
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append(
                    (obj["prompt"], obj["chosen"], obj["rejected"]))

    def __len__(self):
        return len(self.samples)

    def _encode_pair(self, prompt, response):
        """Mã hóa (prompt + response), mask phần prompt trong target.

        Logic giống JsonlSFTDataset._encode: loss chỉ tính trên phần response.
        """
        p_tok, p_tone = self.tokenizer.encode(prompt, add_special=False)
        r_tok, r_tone = self.tokenizer.encode(response, add_special=False)

        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        pad_id = self.tokenizer.token_to_id("<pad>")

        tok = [bos] + p_tok + r_tok + [eos]
        tone = [0] + p_tone + r_tone + [0]

        # Cắt / đệm cho đủ block_size + 1.
        need = self.block_size + 1
        if len(tok) > need:
            tok = tok[:need]
            tone = tone[:need]
        else:
            pad_n = need - len(tok)
            tok = tok + [pad_id] * pad_n
            tone = tone + [0] * pad_n

        tok = torch.tensor(tok, dtype=torch.int64)
        tone = torch.tensor(tone, dtype=torch.int64)

        x = tok[:-1]
        t = tone[:-1]
        y = tok[1:].clone()

        # Mask phần prompt và padding khỏi loss.
        prompt_len = 1 + len(p_tok)
        y[:prompt_len - 1] = self.ignore_index
        y[x == pad_id] = self.ignore_index

        return x, t, y

    def __getitem__(self, idx):
        prompt, chosen, rejected = self.samples[idx]
        c_x, c_t, c_y = self._encode_pair(prompt, chosen)
        r_x, r_t, r_y = self._encode_pair(prompt, rejected)
        return {
            "chosen_token_ids": c_x,
            "chosen_tone_ids": c_t,
            "chosen_targets": c_y,
            "rejected_token_ids": r_x,
            "rejected_tone_ids": r_t,
            "rejected_targets": r_y,
        }


class CalibrationDataset(Dataset):
    """
    Dataset cho giai đoạn 4 — calibration (huấn luyện abstention head).

    Đọc file .jsonl, mỗi dòng:
        {"prompt": "...", "response": "...", "should_abstain": 0 hoặc 1}

    Mỗi mẫu trả về:
        token_ids, tone_ids, targets   — như SFT (để giữ năng lực ngôn ngữ).
        abstention_targets             — nhãn 0/1 cho TỪNG vị trí token.
        abstention_mask                — 1 ở vị trí cần tính abstention loss.

    Nhãn should_abstain được "trải" ra cho mọi vị trí của phần response: nếu
    câu hỏi nên bị từ chối thì mọi token response đều mang nhãn 1.
    """

    def __init__(self, jsonl_path, tokenizer, block_size, ignore_index=-100):
        import json
        self.samples = []
        self.block_size = block_size
        self.ignore_index = ignore_index
        self.tokenizer = tokenizer

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                self.samples.append((
                    obj["prompt"], obj["response"],
                    float(obj.get("should_abstain", 0)),
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response, should_abstain = self.samples[idx]

        p_tok, p_tone = self.tokenizer.encode(prompt, add_special=False)
        r_tok, r_tone = self.tokenizer.encode(response, add_special=False)

        bos = self.tokenizer.token_to_id("<bos>")
        eos = self.tokenizer.token_to_id("<eos>")
        pad_id = self.tokenizer.token_to_id("<pad>")

        tok = [bos] + p_tok + r_tok + [eos]
        tone = [0] + p_tone + r_tone + [0]

        need = self.block_size + 1
        if len(tok) > need:
            tok = tok[:need]
            tone = tone[:need]
        else:
            pad_n = need - len(tok)
            tok = tok + [pad_id] * pad_n
            tone = tone + [0] * pad_n

        tok = torch.tensor(tok, dtype=torch.int64)
        tone = torch.tensor(tone, dtype=torch.int64)

        x = tok[:-1]
        t = tone[:-1]
        y = tok[1:].clone()

        prompt_len = 1 + len(p_tok)
        y[:prompt_len - 1] = self.ignore_index
        y[x == pad_id] = self.ignore_index

        # Nhãn abstention: trải should_abstain ra mọi vị trí response.
        abst_tgt = torch.zeros(self.block_size, dtype=torch.float32)
        abst_mask = torch.zeros(self.block_size, dtype=torch.float32)
        # Vùng response trong dãy đầu vào x (đã dịch): từ prompt_len-1 trở đi.
        resp_start = prompt_len - 1
        for i in range(resp_start, self.block_size):
            if x[i] == pad_id:
                break
            abst_tgt[i] = should_abstain
            abst_mask[i] = 1.0

        return {
            "token_ids": x,
            "tone_ids": t,
            "targets": y,
            "abstention_targets": abst_tgt,
            "abstention_mask": abst_mask,
        }

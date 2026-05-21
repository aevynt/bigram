#!/usr/bin/env python
"""
make_calibration_data.py
========================
Tiện ích tạo dữ liệu mẫu cho GIAI ĐOẠN 4 — calibration (huấn luyện abstention head).

Bối cảnh (xem PHILOSOPHY.md, trụ cột 2):
Abstention head là đầu ra dự đoán "model có NÊN từ chối trả lời không". Để train
nó, cần dữ liệu có nhãn: câu hỏi nào model nên trả lời, câu hỏi nào nên nói
"tôi không chắc". Script này tạo một file .jsonl mẫu theo đúng định dạng.

Mỗi dòng .jsonl có cấu trúc:
    {"prompt": "<câu hỏi>", "response": "<câu trả lời>", "should_abstain": 0 hoặc 1}

  - should_abstain = 0 : câu hỏi có thể trả lời chắc chắn -> model NÊN trả lời.
  - should_abstain = 1 : câu hỏi bịa / mơ hồ / ngoài tầm -> model NÊN từ chối.

Đây CHỈ là dữ liệu minh họa định dạng. Để calibration thực sự, bạn cần xây
một tập lớn hơn nhiều, phản ánh đúng phân bố câu hỏi thực tế của người dùng.

Cách dùng:
    python scripts/make_calibration_data.py --output data/calibration.jsonl
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Dữ liệu minh họa. Mỗi phần tử: (prompt, response, should_abstain).
# Nhóm 1 — câu hỏi có thể trả lời (should_abstain = 0).
_ANSWERABLE = [
    ("Thủ đô của Việt Nam là gì?",
     "Thủ đô của Việt Nam là Hà Nội.", 0),
    ("Một năm có bao nhiêu tháng?",
     "Một năm có mười hai tháng.", 0),
    ("Nước sôi ở nhiệt độ bao nhiêu?",
     "Ở áp suất khí quyển bình thường, nước sôi ở một trăm độ C.", 0),
    ("Mặt trời mọc ở hướng nào?",
     "Mặt trời mọc ở hướng đông.", 0),
    ("Sông dài nhất Việt Nam tên gì?",
     "Đó là sông Đồng Nai nếu chỉ tính phần chảy trong nước.", 0),
]

# Nhóm 2 — câu hỏi model NÊN từ chối (should_abstain = 1):
# câu hỏi về thông tin không tồn tại, riêng tư, hoặc ngoài khả năng biết.
_SHOULD_ABSTAIN = [
    ("Số căn cước công dân của tôi là gì?",
     "Tôi không có thông tin đó và không thể biết được.", 1),
    ("Ngày mai giá vàng sẽ là bao nhiêu?",
     "Tôi không thể dự đoán chắc chắn giá vàng trong tương lai.", 1),
    ("Hôm nay tôi ăn gì vào bữa trưa?",
     "Tôi không có cách nào biết được điều đó.", 1),
    ("Tên con mèo nhà hàng xóm của tôi là gì?",
     "Tôi không có thông tin về điều này.", 1),
    ("Có bao nhiêu hạt cát trên bãi biển Nha Trang?",
     "Đây là con số không thể xác định chính xác được.", 1),
]


def main():
    parser = argparse.ArgumentParser(
        description="Tạo dữ liệu calibration mẫu cho abstention head")
    parser.add_argument("--output", default="data/calibration.jsonl",
                        help="Đường dẫn file .jsonl output")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    all_samples = _ANSWERABLE + _SHOULD_ABSTAIN
    with open(args.output, "w", encoding="utf-8") as f:
        for prompt, response, should_abstain in all_samples:
            obj = {
                "prompt": prompt,
                "response": response,
                "should_abstain": should_abstain,
            }
            # ensure_ascii=False để giữ nguyên chữ tiếng Việt có dấu.
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    n_answer = len(_ANSWERABLE)
    n_abstain = len(_SHOULD_ABSTAIN)
    print(f"Đã tạo {args.output}")
    print(f"  {n_answer} mẫu nên trả lời (should_abstain=0)")
    print(f"  {n_abstain} mẫu nên từ chối  (should_abstain=1)")
    print()
    print("LƯU Ý: đây chỉ là dữ liệu minh họa định dạng. Để calibration thực")
    print("sự hiệu quả, hãy mở rộng tập này với hàng nghìn ví dụ đa dạng.")


if __name__ == "__main__":
    main()

import os
import sys
import glob
import json
import unicodedata
import random
import difflib

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def clean_text(text):
    if not text:
        return ""
    # Unicode NFC Normalization
    text = unicodedata.normalize("NFC", text)
    # Sửa các lỗi chính tả tiếng Việt được yêu cầu
    replacements = {
        "qủa": "quả",
        "gía": "giá",
        "tường đương": "tương đương",
        "đươc": "được",
        "tưong": "tương",
        "yều": "yêu"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def main():
    source_dir = "datasach"
    files = glob.glob(os.path.join(source_dir, "*.jsonl"))
    
    total_read = 0
    invalid_json = 0
    empty_prompt = 0
    short_resp = 0
    long_resp = 0
    skipped_format = 0
    
    processed_samples = []
    
    print(f"Bắt đầu đọc dữ liệu từ {len(files)} file...")
    
    for f_path in files:
        with open(f_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                if not line.strip():
                    continue
                total_read += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    invalid_json += 1
                    continue
                
                prompt = ""
                response = ""
                
                # Nhận diện định dạng
                if "conversations" in obj:
                    convs = obj["conversations"]
                    user_msgs = [m["content"] for m in convs if m.get("role") == "user"]
                    assistant_msgs = [m["content"] for m in convs if m.get("role") == "assistant"]
                    if user_msgs and assistant_msgs:
                        prompt = user_msgs[0]
                        response = assistant_msgs[0]
                    else:
                        skipped_format += 1
                        continue
                elif "prompt" in obj and "response" in obj:
                    prompt = obj["prompt"]
                    response = obj["response"]
                else:
                    skipped_format += 1
                    continue
                
                # Làm sạch văn bản
                prompt = clean_text(prompt).strip()
                response = clean_text(response).strip()
                
                # Bộ lọc
                if not prompt:
                    empty_prompt += 1
                    continue
                if len(response) < 10:
                    short_resp += 1
                    continue
                if len(response) > 1000:
                    long_resp += 1
                    continue
                
                processed_samples.append({"prompt": prompt, "response": response})
                
    print("\nThống kê trước khi loại trùng:")
    print(f"  Tổng số mẫu đọc vào: {total_read}")
    print(f"  Lọc bỏ - JSON không hợp lệ: {invalid_json}")
    print(f"  Lọc bỏ - Định dạng lạ: {skipped_format}")
    print(f"  Lọc bỏ - Prompt rỗng: {empty_prompt}")
    print(f"  Lọc bỏ - Response < 10 ký tự: {short_resp}")
    print(f"  Lọc bỏ - Response > 1000 ký tự: {long_resp}")
    print(f"  Số mẫu hợp lệ tạm thời: {len(processed_samples)}")
    
    # Dedup: Bỏ mẫu có prompt giống nhau trên 90%
    # Sắp xếp các mẫu theo độ dài của prompt để so sánh tối ưu (sliding window)
    processed_samples.sort(key=lambda x: len(x["prompt"]))
    
    unique_samples = []
    deduped_count = 0
    
    # Để kiểm tra xem sample đã bị loại chưa
    removed = [False] * len(processed_samples)
    
    for i in range(len(processed_samples)):
        if removed[i]:
            continue
        p1 = processed_samples[i]["prompt"]
        unique_samples.append(processed_samples[i])
        
        # Chỉ so sánh với các prompt phía sau có độ dài tăng không quá 10%
        j = i + 1
        while j < len(processed_samples) and len(processed_samples[j]["prompt"]) <= len(p1) * 1.1:
            if not removed[j]:
                p2 = processed_samples[j]["prompt"]
                # Tính độ tương đồng nhanh bằng difflib
                matcher = difflib.SequenceMatcher(None, p1, p2)
                # Quick ratio để loại bỏ nhanh các chuỗi quá khác nhau
                if matcher.real_quick_ratio() > 0.9 and matcher.quick_ratio() > 0.9:
                    if matcher.ratio() > 0.9:
                        removed[j] = True
                        deduped_count += 1
            j += 1
            
    print(f"  Lọc bỏ - Trùng lặp prompt >90%: {deduped_count}")
    print(f"  Số mẫu độc bản cuối cùng: {len(unique_samples)}")
    
    # Shuffle seed=42
    random.seed(42)
    random.shuffle(unique_samples)
    
    # Split 90/10
    split_idx = int(len(unique_samples) * 0.9)
    train_samples = unique_samples[:split_idx]
    val_samples = unique_samples[split_idx:]
    
    # Tạo thư mục data nếu chưa có
    os.makedirs("data", exist_ok=True)
    
    # Ghi file
    with open("data/nano15_train.jsonl", "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    with open("data/nano15_val.jsonl", "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print("\nChia tập dữ liệu thành công:")
    print(f"  data/nano15_train.jsonl: {len(train_samples)} mẫu")
    print(f"  data/nano15_val.jsonl  : {len(val_samples)} mẫu")

if __name__ == "__main__":
    main()

#!/usr/bin/env python
import json
import random
import re
from pathlib import Path


FORBIDDEN = {"\u2014", "\u201c", "\u201d", "\u2018", "\u2019", "\u2026"}


def dump_json(obj):
    return json.dumps(obj, ensure_ascii=False)


def tool_call(name, arguments):
    return "<tool_call>\n" + dump_json({"name": name, "arguments": arguments}) + "\n</tool_call>"


def tool_result(obj, question):
    return "<tool_result>\n" + dump_json(obj) + "\n</tool_result>\n\nCâu hỏi gốc: " + question


def validate_text(text):
    if any(ch in text for ch in FORBIDDEN):
        raise ValueError("forbidden punctuation")
    if "\ufffd" in text:
        raise ValueError("replacement character")


def validate_tool_call(response):
    if "<tool_call>" not in response:
        return
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.S)
    if not match:
        raise ValueError("missing tool_call json")
    obj = json.loads(match.group(1))
    if not isinstance(obj.get("arguments"), dict):
        raise ValueError("arguments must be object")


def validate_tool_result_prompt(prompt):
    if "<tool_result>" not in prompt:
        return
    match = re.search(r"<tool_result>\s*(\{.*?\})\s*</tool_result>", prompt, re.S)
    if not match:
        raise ValueError("missing tool_result json")
    json.loads(match.group(1))


def add(rows, group, prompt, response):
    validate_text(prompt + response)
    validate_tool_call(response)
    validate_tool_result_prompt(prompt)
    rows.append({"prompt": prompt, "response": response, "group": group})


def split_write(prefix, rows):
    random.Random(42).shuffle(rows)
    split = int(len(rows) * 0.9)
    train, val = rows[:split], rows[split:]
    for path, subset in [
        (Path(f"data/{prefix}_sft.jsonl"), rows),
        (Path(f"data/{prefix}_train.jsonl"), train),
        (Path(f"data/{prefix}_val.jsonl"), val),
    ]:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in subset:
                fh.write(dump_json({"prompt": row["prompt"], "response": row["response"]}) + "\n")
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    return counts, len(train), len(val)


def build_model_a():
    rows = []
    search_topics = [
        ("giá vàng hôm nay?", "giá vàng hôm nay"),
        ("thời tiết Hà Nội ngày mai?", "thời tiết Hà Nội ngày mai"),
        ("tỷ giá USD hôm nay?", "tỷ giá USD hôm nay"),
        ("tin tức công nghệ mới nhất?", "tin tức công nghệ mới nhất"),
        ("dân số Việt Nam hiện tại?", "dân số Việt Nam hiện tại"),
        ("lịch thi đấu bóng đá tối nay?", "lịch thi đấu bóng đá tối nay"),
        ("giá cổ phiếu VNM hôm nay?", "giá cổ phiếu VNM hôm nay"),
        ("lịch chiếu phim cuối tuần này?", "lịch chiếu phim cuối tuần này"),
        ("giá xăng mới nhất?", "giá xăng mới nhất"),
        ("dân số Hà Nội hiện tại?", "dân số Hà Nội hiện tại"),
        ("giá bitcoin hôm nay?", "giá bitcoin hôm nay"),
        ("thời tiết Đà Nẵng có mưa không?", "thời tiết Đà Nẵng hôm nay"),
    ]
    search_templates = [
        "{q}",
        "cho mình biết {q}",
        "bạn tìm giúp mình {q}",
        "mình cần thông tin về {q}",
        "hiện tại {q}",
        "tra cứu nhanh {q}",
        "cập nhật mới nhất: {q}",
        "làm ơn tìm {q}",
        "nói mình biết {q}",
        "mình muốn xem {q}",
    ]
    for i in range(120):
        q, query = search_topics[i % len(search_topics)]
        prompt = search_templates[i % len(search_templates)].format(q=q)
        add(rows, "A1", prompt, tool_call("search", {"query": query}))

    quadratics = [
        "x^2 - 5x + 6 = 0", "x^2 - 4 = 0", "2x^2 + 3x - 2 = 0",
        "x^2 + 2x + 1 = 0", "3x^2 - 12 = 0", "x^2 - x - 6 = 0",
    ]
    for i in range(30):
        expr = quadratics[i % len(quadratics)]
        prompt = ["giải ", "tìm nghiệm ", "phương trình ", "giúp mình giải "][i % 4] + expr
        add(rows, "A2_quadratic", prompt, tool_call("calculate", {"type": "quadratic", "expression": expr}))

    multiplies = [
        ("123 nhân 456 bằng bao nhiêu?", "123 * 456"),
        ("tính 3.14 * 25", "3.14 * 25"),
        ("15% của 2 triệu là bao nhiêu?", "2000000 * 15 / 100"),
        ("chia đều 5 triệu cho 4 người", "5000000 / 4"),
        ("20% của 5 triệu là bao nhiêu?", "5000000 * 20 / 100"),
        ("12 nhân 37 bằng bao nhiêu?", "12 * 37"),
    ]
    for i in range(30):
        prompt, expr = multiplies[i % len(multiplies)]
        add(rows, "A2_multiply", prompt, tool_call("calculate", {"type": "multiply", "expression": expr}))

    systems = [
        "2x + y = 5 và x - y = 1",
        "x + 2y = 7 và 3x - y = 5",
        "x + y = 10 và x - y = 4",
        "3x + y = 11 và x + y = 5",
        "2x - y = 3 và x + y = 6",
    ]
    for i in range(30):
        expr = systems[i % len(systems)]
        prompt = ["giải hệ ", "tìm x và y biết ", "hệ phương trình ", "giúp mình giải hệ "][i % 4] + expr
        add(rows, "A2_linear_system", prompt, tool_call("calculate", {"type": "linear_system", "expression": expr}))

    word_problems = [
        "An có 5 cái kẹo, cho Bình 2 cái, mua thêm 3 cái. Hỏi An còn mấy cái?",
        "Một cửa hàng bán được 120 sản phẩm trong 3 ngày. Hỏi trung bình mỗi ngày bán bao nhiêu?",
        "Một lớp có 40 học sinh, 60% là nữ. Hỏi có bao nhiêu học sinh nữ?",
        "Lan có 10 quyển vở, tặng bạn 4 quyển rồi mua thêm 2 quyển. Hỏi Lan có mấy quyển?",
        "Một xe đi 180 km trong 3 giờ. Hỏi trung bình mỗi giờ đi bao nhiêu km?",
    ]
    for i in range(30):
        expr = word_problems[i % len(word_problems)]
        add(rows, "A2_word_problem", expr, tool_call("calculate", {"type": "word_problem", "expression": expr}))

    common = [
        ("thủ đô Việt Nam là gì?", "Thủ đô Việt Nam là Hà Nội."),
        ("thủ đô Pháp là gì?", "Thủ đô Pháp là Paris."),
        ("nước nào đông dân nhất thế giới?", "Ấn Độ là nước đông dân nhất thế giới hiện nay."),
        ("công thức tính diện tích hình tròn là gì?", "Diện tích hình tròn bằng pi nhân bán kính bình phương."),
        ("nước sôi ở bao nhiêu độ C?", "Ở áp suất tiêu chuẩn, nước sôi ở 100 độ C."),
        ("quang hợp là gì?", "Quang hợp là quá trình cây dùng ánh sáng để tạo chất dinh dưỡng."),
        ("đệ quy là gì?", "Đệ quy là khi một hàm tự gọi lại chính nó với bài toán nhỏ hơn."),
        ("Trái Đất quay quanh gì?", "Trái Đất quay quanh Mặt Trời."),
        ("một ngày có bao nhiêu giờ?", "Một ngày có 24 giờ."),
        ("màu đỏ trộn xanh dương ra màu gì?", "Thường sẽ tạo ra màu tím."),
    ]
    chat = [
        ("xin chào!", "Chào bạn, mình là Bigram Nano 2 của Aevynt Lab."),
        ("cảm ơn bạn", "Không có gì, mình rất vui được hỗ trợ."),
        ("tạm biệt", "Tạm biệt, mình là Bigram Nano 2 của Aevynt Lab. Hẹn gặp lại."),
        ("bạn khỏe không?", "Mình ổn nha, sẵn sàng hỗ trợ bạn."),
    ]
    identity = [
        ("bạn là ai?", "Mình là Bigram Nano 2 do Aevynt Lab phát triển."),
        ("ai tạo ra bạn?", "Aevynt Lab phát triển mình. Tên mình là Bigram Nano 2."),
        ("bạn làm được gì?", "Mình có thể định tuyến câu hỏi và hỗ trợ trả lời ngắn gọn."),
        ("bạn là ChatGPT à?", "Không, mình là Bigram Nano 2 của Aevynt Lab."),
        ("bạn là AI không?", "Đúng, mình là Bigram Nano 2 do Aevynt Lab phát triển."),
    ]
    odd = [
        ("bạn ăn cơm chưa?", "Mình không ăn cơm, nhưng vẫn sẵn sàng trả lời bạn."),
        ("bạn ở đâu?", "Mình đang ở trong phiên chat này."),
        ("bạn mặc gì?", "Mình không có cơ thể nên không mặc quần áo."),
        ("bạn bao nhiêu tuổi?", "Mình không có tuổi như con người, chỉ có phiên bản Nano 2."),
        ("bạn ngủ chưa?", "Mình không ngủ như người, chỉ im lặng khi không có câu hỏi."),
    ]
    for i in range(50):
        add(rows, "A3_common", *common[i % len(common)])
    for i in range(40):
        add(rows, "A3_chat", *chat[i % len(chat)])
    for i in range(30):
        add(rows, "A3_identity", *identity[i % len(identity)])
    for i in range(30):
        add(rows, "A3_odd", *odd[i % len(odd)])

    known = common[:5]
    changing = [
        ("dân số Việt Nam hiện tại?", "dân số Việt Nam hiện tại"),
        ("giá xăng mới nhất?", "giá xăng mới nhất"),
        ("tổng thống Mỹ hiện tại là ai?", "tổng thống Mỹ hiện tại"),
        ("giá vàng hôm nay?", "giá vàng hôm nay"),
        ("thời tiết Hà Nội ngày mai?", "thời tiết Hà Nội ngày mai"),
    ]
    for i in range(25):
        add(rows, "A4_known", *known[i % len(known)])
    for i in range(25):
        prompt, query = changing[i % len(changing)]
        add(rows, "A4_changing", prompt, tool_call("search", {"query": query}))
    return rows


def build_model_b():
    rows = []
    search_items = [
        ("giá vàng hôm nay?", "giá vàng hôm nay", "Giá vàng hôm nay khoảng 85 triệu đồng mỗi lượng.", "Giá vàng hôm nay khoảng 85 triệu đồng mỗi lượng."),
        ("thời tiết Hà Nội ngày mai?", "thời tiết Hà Nội ngày mai", "Hà Nội ngày mai có mưa nhẹ, khoảng 24 độ C.", "Hà Nội ngày mai có mưa nhẹ, nhiệt độ khoảng 24 độ C."),
        ("tỷ giá USD hôm nay?", "tỷ giá USD hôm nay", "1 USD khoảng 25,400 VND.", "Tỷ giá tham khảo là 1 USD khoảng 25,400 VND."),
        ("dân số Việt Nam hiện tại?", "dân số Việt Nam hiện tại", "Dân số Việt Nam khoảng 98 triệu người.", "Dân số Việt Nam hiện khoảng 98 triệu người."),
        ("giá bitcoin?", "giá bitcoin", "Bitcoin hôm nay khoảng 67,000 USD.", "Bitcoin hiện khoảng 67,000 USD."),
    ]
    for i in range(85):
        q, query, result, answer = search_items[i % len(search_items)]
        add(rows, "B1_search", tool_result({"name": "search", "query": query, "result": result}, q), answer)
    for i in range(15):
        q = "giá vé concert hôm nay?"
        prompt = tool_result({"name": "search", "query": q, "error": "Không tìm thấy kết quả"}, q)
        add(rows, "B1_search_error", prompt, "Mình chưa tìm được kết quả phù hợp. Bạn có thể thử lại với từ khóa cụ thể hơn.")

    quadratics = [
        ("giải x^2 - 5x + 6 = 0", {"x1": 2, "x2": 3}, "Phương trình có hai nghiệm là x = 2 và x = 3."),
        ("giải x^2 - 4 = 0", {"x1": -2, "x2": 2}, "Phương trình có hai nghiệm là x = -2 và x = 2."),
        ("giải x^2 + 2x + 1 = 0", {"x1": -1, "x2": -1}, "Phương trình có nghiệm kép x = -1."),
        ("giải x^2 + 1 = 0", {"no_solution": "vô nghiệm thực"}, "Phương trình vô nghiệm trong tập số thực."),
    ]
    for i in range(30):
        q, result, ans = quadratics[i % len(quadratics)]
        add(rows, "B2_quadratic", tool_result({"name": "calculate", "type": "quadratic", "result": result}, q), ans)

    mults = [
        ("15% của 2 triệu là bao nhiêu?", 300000, "15% của 2 triệu là 300000 đồng."),
        ("123 nhân 456 bằng bao nhiêu?", 56088, "123 nhân 456 bằng 56088."),
        ("chia đều 5 triệu cho 4 người?", 1250000, "Mỗi người nhận 1250000 đồng."),
        ("20% của 5 triệu là bao nhiêu?", 1000000, "20% của 5 triệu là 1000000 đồng."),
    ]
    for i in range(30):
        q, result, ans = mults[i % len(mults)]
        add(rows, "B2_multiply", tool_result({"name": "calculate", "type": "multiply", "result": result}, q), ans)

    systems = [
        ("giải hệ 2x + y = 5 và x - y = 1", {"x": 2, "y": 1}, "Hệ phương trình có nghiệm x = 2 và y = 1."),
        ("giải hệ x + y = 10 và x - y = 4", {"x": 7, "y": 3}, "Hệ có nghiệm x = 7 và y = 3."),
        ("giải hệ x + y = 2 và 2x + 2y = 5", {"no_solution": "vô nghiệm"}, "Hệ phương trình vô nghiệm vì hai điều kiện mâu thuẫn."),
        ("giải hệ x + y = 2 và 2x + 2y = 4", {"infinite_solutions": "vô số nghiệm"}, "Hệ có vô số nghiệm vì hai phương trình tương đương."),
    ]
    for i in range(30):
        q, result, ans = systems[i % len(systems)]
        add(rows, "B2_linear_system", tool_result({"name": "calculate", "type": "linear_system", "result": result}, q), ans)

    words = [
        ("An có 5 kẹo cho 2 mua thêm 3 còn mấy?", {"result": "6", "steps": ["5-2=3", "3+3=6"]}, "An còn 6 cái kẹo: sau khi cho 2 cái còn 3, mua thêm 3 thì có 6."),
        ("Một lớp có 40 học sinh, 60% là nữ. Hỏi có bao nhiêu học sinh nữ?", {"result": "24", "steps": ["40*60%=24"]}, "Lớp đó có 24 học sinh nữ."),
        ("Một cửa hàng bán 120 sản phẩm trong 3 ngày. Trung bình mỗi ngày bán bao nhiêu?", {"result": "40", "steps": ["120/3=40"]}, "Trung bình mỗi ngày cửa hàng bán 40 sản phẩm."),
    ]
    for i in range(30):
        q, result, ans = words[i % len(words)]
        add(rows, "B2_word_problem", tool_result({"name": "calculate", "type": "word_problem", **result}, q), ans)

    variants = [
        ("15% của 2 triệu là bao nhiêu?", {"name": "calculate", "type": "multiply", "result": 300000}, [
            "Kết quả là 300000 đồng.",
            "15% của 2 triệu tương đương 300000 đồng.",
            "Bạn lấy 2 triệu nhân 15%, kết quả là 300000 đồng.",
        ]),
        ("giá vàng hôm nay?", {"name": "search", "query": "giá vàng hôm nay", "result": "Giá vàng hôm nay khoảng 85 triệu đồng mỗi lượng."}, [
            "Giá vàng hôm nay khoảng 85 triệu đồng mỗi lượng.",
            "Theo kết quả tìm được, vàng đang ở khoảng 85 triệu đồng mỗi lượng.",
            "Mức giá tham khảo hiện là khoảng 85 triệu đồng mỗi lượng.",
        ]),
        ("giải x^2 - 5x + 6 = 0", {"name": "calculate", "type": "quadratic", "result": {"x1": 2, "x2": 3}}, [
            "Phương trình có hai nghiệm x = 2 và x = 3.",
            "Nghiệm của phương trình là 2 và 3.",
            "Kết quả: x1 = 2, x2 = 3.",
        ]),
    ]
    i = 0
    while len([r for r in rows if r["group"] == "B3_variants"]) < 70:
        q, result, answers = variants[i % len(variants)]
        ans = answers[(i // len(variants)) % len(answers)]
        add(rows, "B3_variants", tool_result(result, q), ans)
        i += 1
    return rows


def main():
    a = build_model_a()
    b = build_model_b()
    a_counts, a_train, a_val = split_write("nano2_a", a)
    b_counts, b_train, b_val = split_write("nano2_b", b)
    print(json.dumps({
        "model_a": {"counts": a_counts, "train": a_train, "val": a_val},
        "model_b": {"counts": b_counts, "train": b_train, "val": b_val},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

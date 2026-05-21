#!/usr/bin/env python
import json
import random
import re
from pathlib import Path


FORBIDDEN_CHARS = {"—", "“", "”", "‘", "’", "…"}
BAD_ACCENTS = [
    "qủa", "Qủa", "gía", "Gía", "tường đương", "Tường đương", "tưong", "Tưong",
    "đươc", "Đươc", "đươợc", "hỏi đap", "kết qủa", "Kết qủa", "số liêu",
    "thưc tế", "Thưc tế", "chinh xác", "Chinh xác", "câu hoi", "Câu hoi",
]


def dump(obj):
    return json.dumps(obj, ensure_ascii=False)


def think(text):
    return f"<think>\n{text}\n</think>\n"


def tool_call(name, arguments):
    return "<tool_call>\n" + dump({"name": name, "arguments": arguments}) + "\n</tool_call>"


def tool_result(obj, question):
    return "<tool_result>\n" + dump(obj) + "\n</tool_result>\n\nCâu hỏi gốc: " + question


def between(text, start, end):
    if start not in text:
        return None
    a = text.index(start) + len(start)
    b = text.index(end, a)
    return text[a:b].strip()


def validate_text(text):
    if any(ch in text for ch in FORBIDDEN_CHARS):
        raise ValueError("forbidden punctuation")
    for bad in BAD_ACCENTS:
        if bad in text:
            raise ValueError(f"bad accent: {bad}")
    if "�" in text:
        raise ValueError("replacement character")


def validate_tool_call(response):
    payload = between(response, "<tool_call>", "</tool_call>")
    if payload is None:
        return
    obj = json.loads(payload)
    if not isinstance(obj, dict) or not isinstance(obj.get("arguments"), dict):
        raise ValueError("invalid tool_call object")


def validate_tool_result(prompt):
    payload = between(prompt, "<tool_result>", "</tool_result>")
    if payload is None:
        return
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("invalid tool_result object")


def add(rows, group, prompt, response):
    validate_text(prompt)
    validate_text(response)
    validate_tool_call(response)
    validate_tool_result(prompt)
    if "<think>" in response and "<tool_call>" in response:
        body = between(response, "<think>", "</think>")
        if "không cần" in body and "<tool_call>" in response:
            raise ValueError("think conflicts with tool call")
    rows.append({"prompt": prompt, "response": response, "group": group})


def split_write(prefix, rows):
    random.Random(42).shuffle(rows)
    split = int(len(rows) * 0.9)
    train, val = rows[:split], rows[split:]
    for path, data in [
        (Path(f"data/{prefix}_sft.jsonl"), rows),
        (Path(f"data/{prefix}_train.jsonl"), train),
        (Path(f"data/{prefix}_val.jsonl"), val),
    ]:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in data:
                out = {"prompt": row["prompt"], "response": row["response"]}
                line = dump(out)
                json.loads(line)
                fh.write(line + "\n")
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    return counts, len(train), len(val)


def build_model_a():
    rows = []
    search_items = [
        ("giá vàng hôm nay thay đổi thế nào so với tuần trước?", "giá vàng hôm nay so với tuần trước"),
        ("thời tiết Hà Nội ngày mai có mưa không?", "thời tiết Hà Nội ngày mai"),
        ("tỷ giá USD hôm nay tại ngân hàng là bao nhiêu?", "tỷ giá USD hôm nay ngân hàng"),
        ("tin tức mới nhất về thị trường chứng khoán Việt Nam là gì?", "tin tức chứng khoán Việt Nam mới nhất"),
        ("dân số Việt Nam hiện tại khoảng bao nhiêu?", "dân số Việt Nam hiện tại"),
        ("lịch thi đấu bóng đá tối nay có trận nào đáng chú ý?", "lịch thi đấu bóng đá tối nay"),
        ("giá cổ phiếu VNM hiện tại là bao nhiêu?", "giá cổ phiếu VNM hiện tại"),
        ("lịch chiếu phim cuối tuần ở Hà Nội có gì mới?", "lịch chiếu phim Hà Nội cuối tuần"),
        ("giá xăng mới nhất đang là bao nhiêu?", "giá xăng mới nhất"),
        ("giá bitcoin hôm nay tăng hay giảm?", "giá bitcoin hôm nay"),
        ("dự báo thời tiết Đà Nẵng cuối tuần này thế nào?", "dự báo thời tiết Đà Nẵng cuối tuần"),
        ("tổng thống Mỹ hiện tại là ai?", "tổng thống Mỹ hiện tại"),
        ("dân số Hà Nội hiện tại là bao nhiêu?", "dân số Hà Nội hiện tại"),
        ("giá vé máy bay đi Đà Nẵng hôm nay khoảng bao nhiêu?", "giá vé máy bay đi Đà Nẵng hôm nay"),
        ("tin mới nhất về công nghệ trí tuệ nhân tạo là gì?", "tin mới nhất trí tuệ nhân tạo"),
        ("lãi suất tiết kiệm ngân hàng hiện nay ra sao?", "lãi suất tiết kiệm ngân hàng hiện nay"),
    ]
    complex_prefixes = ["cho mình biết", "bạn kiểm tra giúp", "mình cần biết", "tra giúp mình", "xem giúp"]
    for i in range(80):
        base, query = search_items[i % len(search_items)]
        prompt = f"{complex_prefixes[i % len(complex_prefixes)]} {base}"
        reason = "Câu hỏi này phụ thuộc vào thông tin mới hoặc thay đổi theo thời gian. Cần dùng search để lấy dữ liệu mới nhất trước khi trả lời."
        add(rows, "A1_search_reasoning", prompt, think(reason) + tool_call("search", {"query": query}))

    simple_search = [
        ("giá vàng mới nhất là bao nhiêu?", "giá vàng mới nhất"),
        ("dự báo thời tiết Hà Nội tuần này?", "dự báo thời tiết Hà Nội tuần này"),
        ("USD đổi sang VND hiện nay?", "USD đổi sang VND hiện nay"),
        ("bitcoin đang ở mức nào?", "giá bitcoin hiện nay"),
        ("xăng RON 95 giá mới nhất?", "giá xăng RON 95 mới nhất"),
        ("tối nay có lịch bóng đá gì?", "lịch bóng đá tối nay"),
        ("VNM hôm nay giao dịch ra sao?", "cổ phiếu VNM hôm nay"),
        ("có tin mới nào đáng chú ý không?", "tin tức mới đáng chú ý"),
    ]
    for i in range(40):
        prompt, query = simple_search[i % len(simple_search)]
        add(rows, "A2_search_direct", prompt, tool_call("search", {"query": query}))

    quadratic = [f"giải phương trình x^2 - {b}x + {c} = 0" for b, c in [(5, 6), (3, 2), (7, 10), (6, 9), (4, 4)]]
    arithmetic = [
        "15% của 2 triệu là bao nhiêu", "2 + 2 bằng mấy", "123 nhân 456 bằng bao nhiêu",
        "20% của 5 triệu là bao nhiêu", "85 triệu tăng 3% là bao nhiêu",
    ]
    systems = [
        "giải hệ 2x + y = 5 và x - y = 1", "giải hệ x + y = 10 và x - y = 4",
        "tìm x y biết x + 2y = 7 và 3x - y = 5", "giải hệ 3x + y = 8 và x + y = 4",
        "giải hệ 2x - y = 3 và x + y = 6",
    ]
    words = [
        "An có 5 cái kẹo, cho Bình 2 cái rồi mua thêm 3 cái, An còn mấy cái?",
        "Một lớp có 40 học sinh, 60% là nữ. Hỏi có bao nhiêu nữ?",
        "Cửa hàng bán 120 sản phẩm trong 3 ngày, trung bình mỗi ngày bán bao nhiêu?",
        "Có 24 quả táo chia đều cho 6 bạn, mỗi bạn được mấy quả?",
        "Một xe đi 60 km trong 2 giờ, vận tốc trung bình là bao nhiêu?",
    ]
    math_groups = [
        ("A3_quadratic_reasoning", quadratic, "quadratic", "Đây là phương trình bậc hai dạng ax^2 + bx + c = 0. Cần dùng calculate để tính nghiệm chính xác."),
        ("A3_multiply_reasoning", arithmetic, "multiply", "Câu hỏi yêu cầu tính toán cụ thể. Cần dùng calculate để đảm bảo kết quả chính xác."),
        ("A3_linear_system_reasoning", systems, "linear_system", "Đây là hệ phương trình hai ẩn. Cần giải đồng thời hai phương trình để tìm x và y."),
        ("A3_word_problem_reasoning", words, "word_problem", "Bài toán lời văn cần đọc hiểu đề và xác định phép tính phù hợp trước khi tính."),
    ]
    for group, items, kind, reason in math_groups:
        for i in range(25):
            prompt = items[i % len(items)]
            add(rows, group, prompt, think(reason) + tool_call("calculate", {"type": kind, "expression": prompt}))

    explanations = [
        ("tại sao bầu trời màu xanh?", "Ánh sáng xanh bị tán xạ mạnh hơn trong khí quyển nên mắt ta thấy bầu trời có màu xanh."),
        ("vì sao nước biển mặn?", "Nước biển mặn vì khoáng chất và muối từ đất đá theo sông suối chảy ra biển rồi tích tụ lâu dài."),
        ("đệ quy trong lập trình là gì?", "Đệ quy là cách một hàm tự gọi lại chính nó để giải bài toán nhỏ hơn của cùng một dạng."),
        ("vì sao cây cần ánh sáng?", "Cây cần ánh sáng để quang hợp, tạo chất dinh dưỡng và duy trì sự sống."),
        ("tại sao kim loại dẫn điện?", "Kim loại dẫn điện vì có nhiều electron tự do có thể di chuyển trong mạng tinh thể."),
    ]
    comparisons = [
        ("nên học Python hay JavaScript trước?", "Nếu muốn học nhanh và làm dữ liệu thì Python dễ bắt đầu hơn. Nếu muốn làm web thì JavaScript hợp hơn."),
        ("đọc sách giấy hay sách điện tử tốt hơn?", "Sách giấy dễ tập trung, còn sách điện tử tiện mang theo. Chọn theo thói quen đọc của bạn."),
        ("làm việc nhóm hay làm một mình hiệu quả hơn?", "Việc nhóm tốt khi cần nhiều góc nhìn, còn làm một mình hợp với việc cần tập trung sâu."),
        ("nên ghi chú bằng tay hay bằng máy?", "Ghi tay giúp nhớ tốt hơn, còn ghi bằng máy nhanh và dễ tìm kiếm hơn."),
        ("học buổi sáng hay buổi tối tốt hơn?", "Buổi sáng thường tỉnh táo, buổi tối yên tĩnh hơn. Quan trọng là chọn lúc bạn tập trung nhất."),
    ]
    boundary = [
        ("công thức tính diện tích hình tròn là gì?", "Diện tích hình tròn bằng pi nhân bán kính bình phương, viết là S = pi r^2."),
        ("thủ đô Pháp là gì?", "Thủ đô của Pháp là Paris."),
        ("nước nào có diện tích lớn nhất thế giới?", "Nga là quốc gia có diện tích lớn nhất thế giới."),
        ("một năm có bao nhiêu tháng?", "Một năm có 12 tháng."),
        ("trái đất quay quanh gì?", "Trái đất quay quanh Mặt Trời."),
    ]
    a4 = explanations * 6 + comparisons * 5 + boundary * 5
    for i, (prompt, answer) in enumerate(a4[:80]):
        reason = "Đây là kiến thức ổn định hoặc có thể giải thích bằng hiểu biết phổ thông. Không cần gọi công cụ bên ngoài."
        add(rows, "A4_direct_reasoning", prompt, think(reason) + answer)

    chats = [
        ("xin chào!", "Chào bạn, mình là Bigram Nano 2 của Aevynt Lab."),
        ("chào buổi sáng", "Chào buổi sáng, chúc bạn một ngày rõ ràng và nhẹ nhàng."),
        ("cảm ơn nhé", "Không có gì, cần gì cứ nhắn mình."),
        ("tạm biệt", "Tạm biệt, hẹn gặp lại bạn."),
        ("bạn khỏe không?", "Mình vẫn ổn và sẵn sàng hỗ trợ bạn."),
    ]
    identities = [
        ("bạn là ai?", "Mình là Bigram Nano 2 do Aevynt Lab phát triển."),
        ("ai tạo ra bạn?", "Mình là Bigram Nano 2 của Aevynt Lab."),
        ("bạn là ChatGPT à?", "Không, mình là Bigram Nano 2 do Aevynt Lab phát triển."),
        ("bạn làm được gì?", "Mình có thể trả lời, suy luận và định tuyến câu hỏi cần dùng công cụ."),
        ("tên bạn là gì?", "Tên mình là Bigram Nano 2, thuộc Aevynt Lab."),
    ]
    odd = [
        ("bạn ăn cơm chưa?", "Mình không ăn cơm, nhưng vẫn sẵn sàng trả lời bạn."),
        ("bạn mặc gì hôm nay?", "Mình không mặc đồ, chỉ mặc định dạng văn bản thôi."),
        ("bạn bao nhiêu tuổi?", "Mình không có tuổi đời như người thật, nhưng tên mình là Bigram Nano 2."),
        ("bạn ở đâu?", "Mình không có địa chỉ vật lý, mình hoạt động trong hệ thống này."),
        ("bạn có buồn không?", "Mình không buồn, nhưng có thể nghe bạn kể chuyện."),
    ]
    common = [
        ("thủ đô Việt Nam là gì?", "Thủ đô Việt Nam là Hà Nội."),
        ("2 cộng 3 bằng mấy?", "2 cộng 3 bằng 5."),
        ("màu của lá cây thường là gì?", "Lá cây thường có màu xanh."),
        ("nước đóng băng ở bao nhiêu độ C?", "Nước thường đóng băng ở 0 độ C."),
        ("mặt trời mọc ở hướng nào?", "Mặt trời mọc ở hướng đông."),
    ]
    a5 = chats * 6 + identities * 6 + odd * 4 + common * 4
    for prompt, answer in a5[:100]:
        add(rows, "A5_direct_simple", prompt, answer)

    return split_write("nano2_a", rows)


def build_model_b():
    rows = []
    search_ok = [
        ("giá vàng hôm nay", "Giá vàng hôm nay: 85 triệu đồng/lượng", "Giá vàng hôm nay khoảng 85 triệu đồng mỗi lượng."),
        ("thời tiết Hà Nội ngày mai", "Hà Nội ngày mai: mưa nhẹ, 24 độ C", "Hà Nội ngày mai có mưa nhẹ, nhiệt độ khoảng 24 độ C."),
        ("tỷ giá USD hôm nay", "1 USD = 25,400 VND", "Tỷ giá tham khảo là 1 USD khoảng 25,400 VND."),
        ("dân số Việt Nam hiện tại", "Dân số Việt Nam 2024: khoảng 98 triệu người", "Dân số Việt Nam hiện khoảng 98 triệu người."),
        ("giá bitcoin", "Bitcoin hôm nay: $67,000", "Bitcoin hiện khoảng 67,000 USD."),
        ("giá xăng mới nhất", "Giá xăng RON 95: 23,500 đồng/lít", "Giá xăng RON 95 hiện khoảng 23,500 đồng mỗi lít."),
        ("lịch bóng đá tối nay", "Tối nay có trận Việt Nam gặp Thái Lan lúc 20:00", "Tối nay có trận Việt Nam gặp Thái Lan lúc 20:00."),
        ("giá cổ phiếu VNM", "VNM đang ở mức 68,000 đồng/cổ phiếu", "Cổ phiếu VNM đang ở khoảng 68,000 đồng một cổ phiếu."),
        ("thời tiết Đà Nẵng cuối tuần", "Đà Nẵng cuối tuần: nắng nhẹ, 29 độ C", "Cuối tuần ở Đà Nẵng dự kiến nắng nhẹ, khoảng 29 độ C."),
        ("lãi suất tiết kiệm", "Lãi suất kỳ hạn 12 tháng khoảng 5.2%/năm", "Lãi suất tiết kiệm 12 tháng hiện khoảng 5.2% mỗi năm."),
    ]
    for i in range(85):
        query, result, response = search_ok[i % len(search_ok)]
        question = query + "?"
        add(rows, "B1_search", tool_result({"name": "search", "query": query, "result": result}, question), response)
    for i in range(15):
        query = f"thông tin chưa có kết quả {i}"
        question = "giá vé concert hôm nay?"
        response = "Mình không tìm thấy kết quả phù hợp. Bạn có thể thử lại với từ khóa cụ thể hơn."
        add(rows, "B1_search_error", tool_result({"name": "search", "query": query, "error": "Không tìm thấy kết quả"}, question), response)

    quadratic_cases = [
        ("giải x^2 - 5x + 6 = 0", {"x1": 2, "x2": 3}, "Phương trình có hai nghiệm x1 = 2 và x2 = 3."),
        ("giải x^2 - 4x + 4 = 0", {"x1": 2, "x2": 2}, "Phương trình có nghiệm kép x = 2."),
        ("giải x^2 + 1 = 0", {"status": "no_real_solution"}, "Phương trình không có nghiệm thực."),
        ("giải x^2 - 3x + 2 = 0", {"x1": 1, "x2": 2}, "Phương trình có hai nghiệm x1 = 1 và x2 = 2."),
        ("giải x^2 - 9 = 0", {"x1": -3, "x2": 3}, "Phương trình có hai nghiệm x1 = -3 và x2 = 3."),
    ]
    for i in range(30):
        q, result, response = quadratic_cases[i % len(quadratic_cases)]
        add(rows, "B2_quadratic", tool_result({"name": "calculate", "type": "quadratic", "result": result}, q), response)

    multiply_cases = [
        ("15% của 2 triệu là bao nhiêu?", 300000, "15% của 2 triệu tương đương 300000 đồng."),
        ("chia đều 5 triệu cho 4 người?", 1250000, "Mỗi người nhận 1250000 đồng."),
        ("123 nhân 456 bằng bao nhiêu?", 56088, "123 nhân 456 bằng 56088."),
        ("20% của 5 triệu là bao nhiêu?", 1000000, "20% của 5 triệu là 1000000 đồng."),
        ("85 triệu tăng 3% là bao nhiêu?", 87550000, "Sau khi tăng 3%, giá mới là 87550000 đồng."),
    ]
    for i in range(30):
        q, result, response = multiply_cases[i % len(multiply_cases)]
        add(rows, "B2_multiply", tool_result({"name": "calculate", "type": "multiply", "result": result}, q), response)

    system_cases = [
        ("giải hệ 2x + y = 5 và x - y = 1", {"x": 2, "y": 1}, "Hệ phương trình có nghiệm x = 2 và y = 1."),
        ("giải hệ x + y = 10 và x - y = 4", {"x": 7, "y": 3}, "Hệ phương trình có nghiệm x = 7 và y = 3."),
        ("giải hệ x + y = 2 và x + y = 3", {"status": "no_solution"}, "Hệ phương trình vô nghiệm vì hai điều kiện mâu thuẫn nhau."),
        ("giải hệ x + y = 4 và 2x + 2y = 8", {"status": "infinite_solutions"}, "Hệ phương trình có vô số nghiệm vì hai phương trình tương đương."),
        ("giải hệ 3x + y = 8 và x + y = 4", {"x": 2, "y": 2}, "Hệ phương trình có nghiệm x = 2 và y = 2."),
    ]
    for i in range(30):
        q, result, response = system_cases[i % len(system_cases)]
        add(rows, "B2_linear_system", tool_result({"name": "calculate", "type": "linear_system", "result": result}, q), response)

    word_cases = [
        ("An có 5 kẹo cho 2 mua thêm 3 còn mấy?", "6", ["5-2=3", "3+3=6"], "An còn 6 cái kẹo: sau khi cho 2 cái còn 3, mua thêm 3 thì có 6."),
        ("Một lớp 40 học sinh, 60% là nữ. Hỏi có bao nhiêu nữ?", "24", ["40*60%=24"], "Lớp đó có 24 học sinh nữ."),
        ("120 sản phẩm bán trong 3 ngày, trung bình mỗi ngày bao nhiêu?", "40", ["120/3=40"], "Trung bình mỗi ngày bán được 40 sản phẩm."),
        ("24 quả táo chia đều cho 6 bạn, mỗi bạn được mấy quả?", "4", ["24/6=4"], "Mỗi bạn nhận được 4 quả táo."),
        ("Xe đi 60 km trong 2 giờ, vận tốc trung bình bao nhiêu?", "30 km/h", ["60/2=30"], "Vận tốc trung bình là 30 km/h."),
    ]
    for i in range(30):
        q, result, steps, response = word_cases[i % len(word_cases)]
        add(rows, "B2_word_problem", tool_result({"name": "calculate", "type": "word_problem", "result": result, "steps": steps}, q), response)

    variant_sources = [
        ({"name": "search", "query": "thời tiết Hà Nội ngày mai", "result": "Hà Nội ngày mai: mưa nhẹ, 24 độ C"}, "thời tiết Hà Nội ngày mai?", [
            "Ngày mai Hà Nội có mưa nhẹ, khoảng 24 độ C.",
            "Dự báo Hà Nội ngày mai mưa nhẹ, nhiệt độ khoảng 24 độ C.",
            "Bạn nên chuẩn bị áo mưa nhẹ, vì Hà Nội ngày mai khoảng 24 độ C và có mưa nhẹ.",
            "Thời tiết Hà Nội ngày mai: mưa nhẹ, khoảng 24 độ C.",
        ]),
        ({"name": "calculate", "type": "multiply", "result": 56088}, "123 nhân 456 bằng bao nhiêu?", [
            "Kết quả của 123 nhân 456 là 56088.",
            "123 x 456 cho kết quả 56088.",
            "Phép nhân này bằng 56088.",
            "Tính ra là 56088.",
        ]),
        ({"name": "calculate", "type": "quadratic", "result": {"x1": 2, "x2": 3}}, "giải x^2 - 5x + 6 = 0", [
            "Hai nghiệm của phương trình là x1 = 2 và x2 = 3.",
            "Phương trình có nghiệm x = 2 hoặc x = 3.",
            "Kết quả là x1 = 2, x2 = 3.",
            "Nghiệm cần tìm là 2 và 3.",
        ]),
        ({"name": "calculate", "type": "linear_system", "result": {"x": 2, "y": 1}}, "giải hệ 2x + y = 5 và x - y = 1", [
            "Hệ có nghiệm x = 2, y = 1.",
            "Giá trị thỏa mãn là x = 2 và y = 1.",
            "Kết quả giải hệ là x = 2, y = 1.",
            "Nghiệm của hệ phương trình là x = 2 và y = 1.",
        ]),
        ({"name": "calculate", "type": "word_problem", "result": "6", "steps": ["5-2=3", "3+3=6"]}, "An có 5 kẹo cho 2 mua thêm 3 còn mấy?", [
            "An còn 6 cái kẹo.",
            "Sau hai bước tính, đáp án là 6 cái kẹo.",
            "An cho đi 2 còn 3, mua thêm 3 nên có 6 cái kẹo.",
            "Kết quả cuối cùng là 6 cái.",
        ]),
    ]
    for i in range(80):
        obj, question, variants = variant_sources[i % len(variant_sources)]
        add(rows, "B3_variants", tool_result(obj, question), variants[(i // len(variant_sources)) % len(variants)])

    return split_write("nano2_b", rows)


def main():
    Path("data").mkdir(exist_ok=True)
    a_counts, a_train, a_val = build_model_a()
    b_counts, b_train, b_val = build_model_b()
    report = {
        "model_a": {"counts": a_counts, "train": a_train, "val": a_val, "dropped": 0},
        "model_b": {"counts": b_counts, "train": b_train, "val": b_val, "dropped": 0},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

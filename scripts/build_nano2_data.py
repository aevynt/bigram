#!/usr/bin/env python
import json
import random
import re
from pathlib import Path


TOOLS = {
    "search": ("tìm kiếm thông tin mới", "query", "nội dung cần tìm kiếm"),
    "calculate": ("tính toán biểu thức", "expression", "biểu thức cần tính"),
    "get_weather": ("xem thời tiết theo địa điểm", "location", "địa điểm cần xem thời tiết"),
    "translate": ("dịch văn bản", "text", "văn bản cần dịch"),
    "summarize": ("tóm tắt văn bản", "text", "văn bản cần tóm tắt"),
    "call_api": ("gọi một API bên ngoài", "endpoint", "đường dẫn API cần gọi"),
}

FORBIDDEN = {"\u2014", "\u201c", "\u201d", "\u2018", "\u2019", "\u2026"}


def schema(tool):
    desc, param, param_desc = TOOLS[tool]
    return json.dumps({
        "name": tool,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {
                param: {"type": "string", "description": param_desc}
            },
        },
    }, ensure_ascii=False)


def tool_prompt(tool, user_text):
    return f"[TOOLS]\n{schema(tool)}\n[/TOOLS]\n\nUser: {user_text}"


def multi_prompt(tool_names, user_text):
    content = "\n".join(schema(tool) for tool in tool_names)
    return f"[TOOLS]\n{content}\n[/TOOLS]\n\nUser: {user_text}"


def simple_tools_prompt(tool_names, user_text):
    return f"[TOOLS] {', '.join(tool_names)} [/TOOLS] User: {user_text}"


def no_tools_prompt(user_text):
    return f"[NO TOOLS] User: {user_text}"


def tool_response(tool, value, think=False):
    param = TOOLS[tool][1]
    call = json.dumps({"name": tool, "arguments": {param: value}}, ensure_ascii=False)
    body = f"<tool_call>\n{call}\n</tool_call>"
    if think:
        return f"<think>câu hỏi này cần {tool} để lấy thông tin</think>\n{body}"
    return body


def group_a():
    specs = {
        "search": [
            ("giá vàng hôm nay là bao nhiêu?", "giá vàng hôm nay"),
            ("tin mới nhất về trí tuệ nhân tạo là gì?", "tin mới nhất về trí tuệ nhân tạo"),
            ("tỷ giá USD hôm nay là bao nhiêu?", "tỷ giá USD hôm nay"),
            ("lịch thi đấu bóng đá tối nay có gì?", "lịch thi đấu bóng đá tối nay"),
            ("ai đang là thủ tướng Nhật Bản hiện nay?", "thủ tướng Nhật Bản hiện nay"),
            ("giá xăng hôm nay thay đổi thế nào?", "giá xăng hôm nay"),
            ("kết quả xổ số hôm nay là gì?", "kết quả xổ số hôm nay"),
            ("cổ phiếu VNM hôm nay ra sao?", "cổ phiếu VNM hôm nay"),
            ("tin thời sự Việt Nam mới nhất là gì?", "tin thời sự Việt Nam mới nhất"),
            ("giá bitcoin hiện tại là bao nhiêu?", "giá bitcoin hiện tại"),
            ("ai vô địch giải quần vợt gần nhất?", "nhà vô địch quần vợt gần nhất"),
            ("lịch chiếu phim mới trong tuần này là gì?", "lịch chiếu phim mới tuần này"),
            ("thông tin mới về iPhone hiện tại là gì?", "thông tin mới về iPhone hiện tại"),
            ("tình hình giao thông Hà Nội lúc này thế nào?", "giao thông Hà Nội lúc này"),
            ("giá vé máy bay đi Đà Nẵng hôm nay?", "giá vé máy bay đi Đà Nẵng hôm nay"),
            ("tin mới về thị trường bất động sản là gì?", "tin thị trường bất động sản mới nhất"),
            ("đội tuyển Việt Nam sắp đá với ai?", "lịch đội tuyển Việt Nam sắp đá"),
            ("giá cà phê hôm nay là bao nhiêu?", "giá cà phê hôm nay"),
            ("tìm thông tin về hội nghị công nghệ mới nhất", "hội nghị công nghệ mới nhất"),
            ("tin mới về biến đổi khí hậu là gì?", "tin mới về biến đổi khí hậu"),
            ("ngày nghỉ lễ sắp tới ở Việt Nam là ngày nào?", "ngày nghỉ lễ sắp tới ở Việt Nam"),
            ("giá vàng thế giới hiện tại thế nào?", "giá vàng thế giới hiện tại"),
            ("bảng xếp hạng bóng đá Anh mới nhất?", "bảng xếp hạng bóng đá Anh mới nhất"),
            ("tin mới về giáo dục hôm nay là gì?", "tin giáo dục hôm nay"),
            ("xu hướng công nghệ năm nay là gì?", "xu hướng công nghệ năm nay"),
        ],
        "calculate": [
            ("15% của 2 triệu là bao nhiêu?", "2000000 * 15 / 100"),
            ("tính 125 cộng 378", "125 + 378"),
            ("9 nhân 87 bằng bao nhiêu?", "9 * 87"),
            ("chia 144 cho 12 giúp tôi", "144 / 12"),
            ("tính diện tích hình chữ nhật 8 nhân 15", "8 * 15"),
            ("2 lũy thừa 10 bằng bao nhiêu?", "2 ** 10"),
            ("giảm 20% của 500000 là bao nhiêu?", "500000 * 0.8"),
            ("tính trung bình của 6, 8, 10", "(6 + 8 + 10) / 3"),
            ("5 phần 8 đổi ra phần trăm", "5 / 8 * 100"),
            ("tính 3600 chia 45", "3600 / 45"),
            ("nếu mua 3 món mỗi món 120000 thì tổng bao nhiêu?", "3 * 120000"),
            ("tính tiền sau thuế 10% của 250000", "250000 * 1.1"),
            ("tính căn bậc hai của 81", "sqrt(81)"),
            ("7 bình phương bằng bao nhiêu?", "7 ** 2"),
            ("tính 1000000 trừ 275000", "1000000 - 275000"),
            ("tính chu vi hình vuông cạnh 9", "4 * 9"),
            ("tính 18% của 750000", "750000 * 18 / 100"),
            ("tính 48 chia 6 cộng 7", "48 / 6 + 7"),
            ("tính 12 nhân 12 nhân 3", "12 * 12 * 3"),
            ("tính 999 cộng 1", "999 + 1"),
        ],
        "get_weather": [
            ("thời tiết Hà Nội ngày mai thế nào?", "Hà Nội"),
            ("trời Đà Nẵng có mưa không?", "Đà Nẵng"),
            ("nhiệt độ Thành phố Hồ Chí Minh lúc này?", "Thành phố Hồ Chí Minh"),
            ("thời tiết Huế cuối tuần này?", "Huế"),
            ("Sa Pa hôm nay lạnh không?", "Sa Pa"),
            ("Cần Thơ có mưa chiều nay không?", "Cần Thơ"),
            ("thời tiết Hải Phòng ngày mai?", "Hải Phòng"),
            ("Nha Trang hôm nay có nắng không?", "Nha Trang"),
            ("Đà Lạt tối nay lạnh không?", "Đà Lạt"),
            ("Quy Nhơn cuối tuần có mưa không?", "Quy Nhơn"),
            ("thời tiết Vũng Tàu hôm nay?", "Vũng Tàu"),
            ("Hạ Long ngày mai trời thế nào?", "Hạ Long"),
            ("Phú Quốc có bão không?", "Phú Quốc"),
            ("thời tiết Buôn Ma Thuột hôm nay?", "Buôn Ma Thuột"),
            ("Thanh Hóa ngày mai có mưa không?", "Thanh Hóa"),
        ],
        "translate": [
            ("dịch hello world sang tiếng Việt", "hello world sang tiếng Việt"),
            ("dịch cảm ơn sang tiếng Anh", "cảm ơn sang tiếng Anh"),
            ("dịch tôi yêu lập trình sang tiếng Anh", "tôi yêu lập trình sang tiếng Anh"),
            ("dịch good morning sang tiếng Việt", "good morning sang tiếng Việt"),
            ("dịch học máy sang tiếng Anh", "học máy sang tiếng Anh"),
            ("dịch artificial intelligence sang tiếng Việt", "artificial intelligence sang tiếng Việt"),
            ("dịch xin lỗi sang tiếng Anh", "xin lỗi sang tiếng Anh"),
            ("dịch see you later sang tiếng Việt", "see you later sang tiếng Việt"),
            ("dịch dữ liệu sang tiếng Anh", "dữ liệu sang tiếng Anh"),
            ("dịch climate change sang tiếng Việt", "climate change sang tiếng Việt"),
            ("dịch tôi cần giúp đỡ sang tiếng Anh", "tôi cần giúp đỡ sang tiếng Anh"),
            ("dịch nice to meet you sang tiếng Việt", "nice to meet you sang tiếng Việt"),
            ("dịch phần mềm sang tiếng Anh", "phần mềm sang tiếng Anh"),
            ("dịch network security sang tiếng Việt", "network security sang tiếng Việt"),
            ("dịch chúc ngủ ngon sang tiếng Anh", "chúc ngủ ngon sang tiếng Anh"),
        ],
        "summarize": [
            ("tóm tắt đoạn văn về biến đổi khí hậu này", "đoạn văn về biến đổi khí hậu"),
            ("rút gọn nội dung cuộc họp này", "nội dung cuộc họp"),
            ("tóm tắt bài viết về học máy", "bài viết về học máy"),
            ("tóm tắt đoạn văn ngắn này", "đoạn văn ngắn này"),
            ("làm ngắn phần mô tả sản phẩm này", "phần mô tả sản phẩm"),
            ("tóm tắt email khách hàng vừa gửi", "email khách hàng"),
            ("tóm tắt báo cáo doanh thu", "báo cáo doanh thu"),
            ("rút ý chính của bài nghiên cứu", "bài nghiên cứu"),
            ("tóm tắt nội dung chính sách mới", "nội dung chính sách mới"),
            ("rút gọn thông báo nội bộ", "thông báo nội bộ"),
            ("tóm tắt ghi chú phỏng vấn", "ghi chú phỏng vấn"),
            ("tóm tắt câu chuyện ngắn", "câu chuyện ngắn"),
            ("lấy ý chính từ đoạn văn sau", "đoạn văn sau"),
            ("tóm tắt tài liệu hướng dẫn", "tài liệu hướng dẫn"),
            ("rút gọn nội dung bài thuyết trình", "bài thuyết trình"),
        ],
        "call_api": [
            ("lấy danh sách sản phẩm từ API", "/products"),
            ("gọi API lấy thông tin người dùng", "/users"),
            ("lấy danh sách đơn hàng", "/orders"),
            ("gọi API kiểm tra trạng thái thanh toán", "/payments/status"),
            ("lấy dữ liệu tồn kho", "/inventory"),
            ("gọi API tạo phiếu hỗ trợ", "/tickets"),
            ("lấy danh sách bình luận", "/comments"),
            ("gọi API cập nhật hồ sơ", "/profile/update"),
            ("lấy thống kê truy cập", "/analytics/visits"),
            ("gọi API lấy cấu hình hệ thống", "/settings"),
        ],
    }
    rows = []
    idx = 0
    for tool, pairs in specs.items():
        for question, value in pairs:
            rows.append({
                "prompt": tool_prompt(tool, question),
                "response": tool_response(tool, value, think=idx < 50),
                "group": "A",
            })
            idx += 1
    return rows


def group_b():
    results = []
    success = [
        ("search", "Giá vàng trong nước đang biến động theo thị trường và cần kiểm tra nguồn cập nhật."),
        ("search", "Tỷ giá USD thay đổi theo từng ngân hàng và thời điểm giao dịch."),
        ("search", "Tin mới nhất cho thấy thị trường công nghệ đang tập trung vào trí tuệ nhân tạo."),
        ("search", "Lịch thi đấu tối nay có nhiều trận, cần đối chiếu theo giải đấu cụ thể."),
        ("calculate", "Kết quả là 300000."),
        ("calculate", "Kết quả phép tính là 503."),
        ("calculate", "Kết quả là 783."),
        ("calculate", "Kết quả là 12."),
        ("get_weather", "Hà Nội ngày mai có khả năng nhiều mây, nhiệt độ khoảng 24 đến 29 độ C."),
        ("get_weather", "Đà Nẵng có thể có mưa rào nhẹ vào chiều tối."),
        ("translate", "hello world nghĩa là xin chào thế giới."),
        ("translate", "cảm ơn trong tiếng Anh là thank you."),
        ("summarize", "Nội dung chính nói về tác động của biến đổi khí hậu và nhu cầu giảm phát thải."),
        ("summarize", "Cuộc họp tập trung vào tiến độ dự án, rủi ro và việc phân công nhiệm vụ."),
        ("call_api", "API trả về 24 sản phẩm đang hoạt động."),
        ("call_api", "API trả về danh sách người dùng cùng trạng thái tài khoản."),
    ]
    errors = [
        ("search", "Lỗi mạng khi tìm kiếm."),
        ("calculate", "Biểu thức không hợp lệ."),
        ("get_weather", "Không tìm thấy địa điểm."),
        ("translate", "Ngôn ngữ đích không được hỗ trợ."),
        ("summarize", "Văn bản đầu vào quá ngắn để tóm tắt."),
        ("call_api", "API trả về mã lỗi 500."),
    ]
    for i in range(82):
        tool, result = success[i % len(success)]
        results.append((tool, result))
    for i in range(18):
        results.append(errors[i % len(errors)])
    rows = []
    for tool, result in results:
        prompt = f"<tool_result>\n{json.dumps({'name': tool, 'result': result}, ensure_ascii=False)}\n</tool_result>\n\nTổng hợp kết quả trên thành câu trả lời tự nhiên:"
        if "Lỗi" in result or "không" in result.lower() and tool in {"get_weather", "translate", "summarize"} or "500" in result:
            response = f"Mình chưa xử lý được vì công cụ báo: {result} Bạn có thể thử lại hoặc cung cấp thông tin rõ hơn."
        else:
            response = f"Kết quả từ {tool} cho biết: {result}"
        rows.append({"prompt": prompt, "response": response, "group": "B"})
    return rows


def group_c():
    base = [
        (["search", "calculate"], "nếu vàng tăng 2% thì giá mới là bao nhiêu?", "search", "giá vàng hiện tại"),
        (["search", "get_weather"], "trời Đà Nẵng có mưa không?", "get_weather", "Đà Nẵng"),
        (["search", "calculate"], "tìm giá vàng rồi tính thêm 2%", "search", "giá vàng hiện tại"),
        (["translate", "summarize"], "dịch đoạn này trước hay tóm tắt trước?", "summarize", "đoạn văn cần tóm tắt"),
        (["search", "summarize"], "tìm tin mới rồi tóm tắt cho tôi", "search", "tin mới cần tóm tắt"),
        (["call_api", "summarize"], "lấy danh sách sản phẩm rồi tóm tắt", "call_api", "/products"),
        (["search", "calculate", "get_weather"], "giá vàng hôm nay là bao nhiêu?", "search", "giá vàng hôm nay"),
        (["search", "translate"], "dịch tin mới về AI sang tiếng Việt", "search", "tin mới về AI"),
        (["calculate", "search"], "15% của 2 triệu là bao nhiêu?", "calculate", "2000000 * 15 / 100"),
        (["search", "call_api"], "lấy danh sách sản phẩm từ API", "call_api", "/products"),
    ]
    rows = []
    while len(rows) < 80:
        tools, question, chosen, value = base[len(rows) % len(base)]
        prompt = simple_tools_prompt(tools, question) if len(rows) < 24 else multi_prompt(tools, question)
        arg = {TOOLS[chosen][1]: value}
        call = json.dumps({"name": chosen, "arguments": arg}, ensure_ascii=False)
        response = (
            "<think>\n"
            f"Câu hỏi có thể liên quan nhiều công cụ, nhưng cần chọn {chosen} trước vì đây là thông tin phù hợp nhất.\n"
            "</think>\n"
            "<tool_call>\n"
            f"{call}\n"
            "</tool_call>"
        )
        rows.append({"prompt": prompt, "response": response, "group": "C"})
    return rows


def group_d():
    think_pairs = [
        ("2 + 2 bằng mấy?", "2 + 2 là phép cộng đơn giản.", "2 + 2 bằng 4."),
        ("thủ đô Việt Nam là gì?", "Đây là kiến thức phổ thông, không cần công cụ.", "Thủ đô Việt Nam là Hà Nội."),
        ("bạn là ai?", "Đây là câu hỏi danh tính, mình tự trả lời được.", "Mình là Bigram Nano 2 của Aevynt Lab."),
        ("xin chào!", "Đây là lời chào, không cần dùng công cụ.", "Chào bạn, mình là Bigram Nano 2 của Aevynt Lab."),
        ("tạm biệt", "Đây là kết thúc hội thoại.", "Tạm biệt, mình là Bigram Nano 2 của Aevynt Lab. Hẹn gặp lại."),
        ("một ngày có bao nhiêu giờ?", "Đây là kiến thức cơ bản.", "Một ngày có 24 giờ."),
    ]
    direct_pairs = [
        ("bạn là ai?", "Mình là Bigram Nano 2 của Aevynt Lab."),
        ("xin chào!", "Chào bạn, mình là Bigram Nano 2 của Aevynt Lab."),
        ("tạm biệt", "Tạm biệt, mình là Bigram Nano 2 của Aevynt Lab. Hẹn gặp lại."),
        ("2 + 2 bằng mấy?", "2 + 2 bằng 4."),
        ("thủ đô Việt Nam là gì?", "Thủ đô Việt Nam là Hà Nội."),
        ("số điện thoại của tôi là gì?", "Mình không biết số điện thoại riêng của bạn, nên không thể trả lời."),
        ("màu đỏ và màu xanh dương trộn lại thường ra màu gì?", "Thường sẽ tạo ra màu tím."),
        ("nước sôi ở bao nhiêu độ C?", "Ở áp suất tiêu chuẩn, nước sôi ở 100 độ C."),
    ]
    rows = []
    for i in range(30):
        user, thought, answer = think_pairs[i % len(think_pairs)]
        prompt = no_tools_prompt(user) if i < 18 else simple_tools_prompt(["search", "calculate"], user)
        rows.append({"prompt": prompt, "response": f"<think>{thought}</think>\n{answer}", "group": "D"})
    for i in range(40):
        user, answer = direct_pairs[i % len(direct_pairs)]
        prompt = no_tools_prompt(user) if i < 28 else simple_tools_prompt(["search", "calculate"], user)
        rows.append({"prompt": prompt, "response": answer, "group": "D"})
    return rows


def load_group_e():
    path = Path("data/plank1_sft.jsonl")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "prompt" in obj and "response" in obj:
                rows.append({"prompt": obj["prompt"], "response": obj["response"], "group": "E"})
    return rows


def validate_row(row):
    json.loads(json.dumps({"prompt": row["prompt"], "response": row["response"]}, ensure_ascii=False))
    text = row["prompt"] + row["response"]
    if any(ch in text for ch in FORBIDDEN):
        raise RuntimeError(f"Ký tự cấm trong mẫu: {row['prompt'][:80]}")
    if not row["prompt"].strip() or not row["response"].strip():
        raise RuntimeError("Mẫu rỗng")
    if "<tool_call>" in row["response"]:
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", row["response"], re.S)
        if not match:
            raise RuntimeError("Thiếu JSON tool_call")
        parsed = json.loads(match.group(1))
        if "name" not in parsed or "arguments" not in parsed:
            raise RuntimeError("tool_call thiếu trường")


def main():
    candidates = group_a() + group_b() + group_c() + group_d() + load_group_e()
    rows = []
    dropped = 0
    for row in candidates:
        try:
            validate_row(row)
        except RuntimeError:
            dropped += 1
            continue
        rows.append(row)
    random.Random(42).shuffle(rows)
    split = int(len(rows) * 0.9)
    train = rows[:split]
    val = rows[split:]
    for path, subset in [
        (Path("data/nano2_sft.jsonl"), rows),
        (Path("data/nano2_train.jsonl"), train),
        (Path("data/nano2_val.jsonl"), val),
    ]:
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in subset:
                fh.write(json.dumps({"prompt": row["prompt"], "response": row["response"]}, ensure_ascii=False) + "\n")
    counts = {}
    for row in rows:
        counts[row["group"]] = counts.get(row["group"], 0) + 1
    print(json.dumps({
        "counts": counts,
        "dropped": dropped,
        "total": len(rows),
        "train": len(train),
        "val": len(val),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

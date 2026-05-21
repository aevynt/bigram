"""
gen_nano2_sft.py
================
Sinh toàn bộ dữ liệu SFT cho Nano 2 trong một lần chạy.

Quy trình:
  1. Sinh Nhóm A (100 mẫu tool calling cơ bản).
  2. Sinh Nhóm B (100 mẫu tổng hợp kết quả tool).
  3. Sinh Nhóm C (80 mẫu reasoning + tool).
  4. Sinh Nhóm D (70 mẫu trả lời thẳng).
  5. Append Nhóm E từ data/nano1_sft_final.jsonl.
  6. Validate JSON, shuffle (seed=42), split 90/10.

Kết quả: data/nano2_sft.jsonl, data/nano2_train.jsonl, data/nano2_val.jsonl.
"""

import json
import os
import random
import sys

OUT_DIR = "data"
SFT_FILE = os.path.join(OUT_DIR, "nano2_sft.jsonl")
TRAIN_FILE = os.path.join(OUT_DIR, "nano2_train.jsonl")
VAL_FILE = os.path.join(OUT_DIR, "nano2_val.jsonl")
GROUP_E_FILE = os.path.join(OUT_DIR, "nano1_sft_final.jsonl")
FALLBACK_E = os.path.join(OUT_DIR, "plank1_sft.jsonl")

# Schemas tool dùng cho prompt. Giữ ngắn để tiết kiệm token (max_seq_len=128).
SCHEMAS = {
    "search":
        '{"name":"search","description":"tra cứu","parameters":{"type":"object","properties":{"q":{"type":"string","description":"từ khóa"}}}}',
    "calculate":
        '{"name":"calculate","description":"tính toán","parameters":{"type":"object","properties":{"expr":{"type":"string","description":"biểu thức"}}}}',
    "get_weather":
        '{"name":"get_weather","description":"thời tiết","parameters":{"type":"object","properties":{"city":{"type":"string","description":"thành phố"}}}}',
    "translate":
        '{"name":"translate","description":"dịch","parameters":{"type":"object","properties":{"text":{"type":"string","description":"văn bản"}}}}',
    "summarize":
        '{"name":"summarize","description":"tóm tắt","parameters":{"type":"object","properties":{"text":{"type":"string","description":"đoạn"}}}}',
    "call_api":
        '{"name":"call_api","description":"gọi api","parameters":{"type":"object","properties":{"endpoint":{"type":"string","description":"điểm cuối"}}}}',
}

# Tham số chính của mỗi tool (tên trường argument).
TOOL_ARG = {
    "search": "q",
    "calculate": "expr",
    "get_weather": "city",
    "translate": "text",
    "summarize": "text",
    "call_api": "endpoint",
}


def fmt_tools_one(tool):
    return f"[TOOLS]\n{SCHEMAS[tool]}\n[/TOOLS]"


def fmt_tools_multi(tools):
    body = "\n".join(SCHEMAS[t] for t in tools)
    return f"[TOOLS]\n{body}\n[/TOOLS]"


def tool_call(tool, arg_value, arg_name=None):
    if arg_name is None:
        arg_name = TOOL_ARG[tool]
    payload = {"name": tool, "arguments": {arg_name: arg_value}}
    return f"<tool_call>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_call>"


def think_tool_call(reason, tool, arg_value, arg_name=None):
    return f"<think>{reason}</think>\n{tool_call(tool, arg_value, arg_name)}"


# ============================================================
# Nhóm A — 100 mẫu tool calling cơ bản
# ============================================================
def gen_group_a():
    random.seed(101)
    samples = []

    # Câu hỏi mẫu cho từng tool. Mỗi list chứa (câu hỏi, giá trị argument, lý do).
    A_data = {
        "search": [
            ("giá vàng hôm nay là bao nhiêu?", "giá vàng hôm nay", "cần tra cứu giá vàng"),
            ("tin tức bóng đá mới nhất", "tin tức bóng đá", "cần tìm tin mới"),
            ("dân số Việt Nam hiện nay", "dân số Việt Nam", "tra cứu dân số"),
            ("thủ đô của Pháp là gì?", "thủ đô Pháp", "tra cứu thông tin"),
            ("ai là tổng thống Mỹ hiện nay?", "tổng thống Mỹ hiện nay", "tra cứu tin tức"),
            ("Hà Nội có bao nhiêu quận?", "Hà Nội có bao nhiêu quận", "tra cứu hành chính"),
            ("ai phát minh ra điện thoại?", "người phát minh điện thoại", "tra cứu lịch sử"),
            ("năm nay là năm con gì?", "năm con gì", "tra cứu lịch âm"),
            ("phim hay nhất năm nay", "phim hay nhất năm nay", "tra cứu phim"),
            ("trận bóng tối qua tỷ số bao nhiêu?", "tỷ số trận bóng tối qua", "tra cứu thể thao"),
            ("sách bán chạy nhất tuần này", "sách bán chạy tuần này", "tra cứu sách"),
            ("giá xăng hôm nay", "giá xăng hôm nay", "tra cứu giá xăng"),
            ("bài hát mới của Sơn Tùng", "bài hát mới Sơn Tùng", "tra cứu nhạc"),
            ("kết quả xổ số miền Bắc", "kết quả xổ số miền Bắc", "tra cứu xổ số"),
            ("danh sách di sản thế giới ở Việt Nam", "di sản thế giới Việt Nam", "tra cứu di sản"),
            ("ai là chủ tịch Trung Quốc?", "chủ tịch Trung Quốc", "tra cứu chính trị"),
            ("núi cao nhất thế giới", "núi cao nhất thế giới", "tra cứu địa lý"),
            ("bảng xếp hạng V-League", "bảng xếp hạng V-League", "tra cứu bóng đá"),
            ("lịch chiếu phim cuối tuần", "lịch chiếu phim cuối tuần", "tra cứu phim"),
            ("chỉ số chứng khoán VN-Index", "chỉ số VN-Index", "tra cứu chứng khoán"),
            ("ngày giỗ Tổ Hùng Vương", "ngày giỗ Tổ Hùng Vương", "tra cứu lễ"),
            ("cách nấu phở bò", "cách nấu phở bò", "tra cứu công thức"),
            ("ai là nhà văn của Tắt Đèn?", "tác giả Tắt Đèn", "tra cứu văn học"),
            ("động vật quý hiếm ở Việt Nam", "động vật quý hiếm Việt Nam", "tra cứu sinh học"),
            ("lịch tàu Hà Nội đi Sài Gòn", "lịch tàu Hà Nội Sài Gòn", "tra cứu giao thông"),
        ],
        "calculate": [
            ("tính 23 cộng 47", "23+47", "phép cộng đơn giản"),
            ("100 chia 4 bằng bao nhiêu", "100/4", "phép chia"),
            ("12 nhân 8", "12*8", "phép nhân"),
            ("căn bậc hai của 144", "sqrt(144)", "căn bậc hai"),
            ("15 phần trăm của 200", "200*0.15", "tính phần trăm"),
            ("tổng các số từ 1 đến 100", "sum(1..100)", "tính tổng dãy"),
            ("2 mũ 10 bằng bao nhiêu", "2^10", "lũy thừa"),
            ("99 chia cho 3", "99/3", "phép chia"),
            ("345 trừ 89", "345-89", "phép trừ"),
            ("diện tích hình tròn bán kính 5", "pi*5*5", "diện tích hình tròn"),
            ("chu vi hình vuông cạnh 7", "4*7", "chu vi hình vuông"),
            ("8 giai thừa", "8!", "tính giai thừa"),
            ("log 100 cơ số 10", "log10(100)", "logarit"),
            ("sin 30 độ", "sin(30)", "lượng giác"),
            ("trung bình của 4, 6, 8, 10", "(4+6+8+10)/4", "trung bình cộng"),
            ("3 phẩy 14 nhân 9", "3.14*9", "phép nhân số thập phân"),
            ("256 chia 16", "256/16", "phép chia"),
            ("17 phần trăm của 1 triệu", "1000000*0.17", "tính phần trăm"),
            ("ước chung lớn nhất của 12 và 18", "gcd(12,18)", "tính ước chung"),
            ("bội chung nhỏ nhất của 6 và 8", "lcm(6,8)", "tính bội chung"),
        ],
        "get_weather": [
            ("thời tiết Hà Nội hôm nay", "Hà Nội", "lấy thời tiết Hà Nội"),
            ("trời ở Đà Nẵng thế nào?", "Đà Nẵng", "lấy thời tiết Đà Nẵng"),
            ("nhiệt độ Sài Gòn bây giờ", "Sài Gòn", "lấy nhiệt độ"),
            ("Huế có mưa không?", "Huế", "kiểm tra mưa"),
            ("thời tiết Hải Phòng ngày mai", "Hải Phòng", "thời tiết tương lai"),
            ("Cần Thơ nóng không?", "Cần Thơ", "kiểm tra nhiệt độ"),
            ("Sa Pa lạnh tới đâu?", "Sa Pa", "kiểm tra nhiệt độ Sa Pa"),
            ("Đà Lạt bây giờ thế nào?", "Đà Lạt", "lấy thời tiết Đà Lạt"),
            ("Vũng Tàu có nắng không?", "Vũng Tàu", "kiểm tra nắng"),
            ("Quy Nhơn mưa to không?", "Quy Nhơn", "kiểm tra mưa"),
            ("Phú Quốc thời tiết ra sao?", "Phú Quốc", "lấy thời tiết đảo"),
            ("Nha Trang nóng bao nhiêu độ?", "Nha Trang", "kiểm tra nhiệt độ"),
            ("Bắc Ninh có mưa hôm nay không?", "Bắc Ninh", "kiểm tra mưa"),
            ("thời tiết Lào Cai cuối tuần", "Lào Cai", "thời tiết cuối tuần"),
            ("Cà Mau bây giờ thế nào?", "Cà Mau", "lấy thời tiết miền Tây"),
        ],
        "translate": [
            ("dịch hello sang tiếng Việt", "hello", "dịch từ tiếng Anh"),
            ("dịch good morning sang tiếng Việt", "good morning", "dịch chào hỏi"),
            ("nghĩa của thank you là gì?", "thank you", "dịch lời cảm ơn"),
            ("xin chào tiếng Anh là gì?", "xin chào", "dịch sang tiếng Anh"),
            ("dịch I love you sang tiếng Việt", "I love you", "dịch tình cảm"),
            ("see you later nghĩa là gì?", "see you later", "dịch lời tạm biệt"),
            ("how are you dịch ra sao?", "how are you", "dịch câu hỏi"),
            ("dịch goodbye sang tiếng Việt", "goodbye", "dịch tạm biệt"),
            ("happy birthday tiếng Việt là gì?", "happy birthday", "dịch chúc mừng"),
            ("good night nghĩa là gì?", "good night", "dịch chúc ngủ ngon"),
            ("dịch please sang tiếng Việt", "please", "dịch từ lịch sự"),
            ("welcome dịch ra sao?", "welcome", "dịch chào đón"),
            ("dịch sorry sang tiếng Việt", "sorry", "dịch xin lỗi"),
            ("merry christmas tiếng Việt là gì?", "merry christmas", "dịch chúc lễ"),
            ("dịch beautiful sang tiếng Việt", "beautiful", "dịch tính từ"),
        ],
        "summarize": [
            ("tóm tắt giúp đoạn này", "đoạn văn người dùng cung cấp", "cần tóm tắt"),
            ("tóm tắt nội dung bài báo", "bài báo người dùng đưa", "cần tóm gọn"),
            ("tóm tắt bài viết hôm nay", "bài viết hôm nay", "cần rút gọn"),
            ("rút gọn đoạn văn này", "đoạn văn dài", "rút gọn"),
            ("tóm tắt câu chuyện vừa kể", "câu chuyện vừa kể", "tóm câu chuyện"),
            ("tóm tắt sách Truyện Kiều", "Truyện Kiều", "tóm sách"),
            ("tóm tắt phim Mắt Biếc", "Mắt Biếc", "tóm phim"),
            ("tóm tắt cuộc họp", "biên bản họp", "tóm họp"),
            ("rút gọn bài luận", "bài luận", "rút gọn luận"),
            ("tóm tắt báo cáo tài chính", "báo cáo tài chính", "tóm báo cáo"),
            ("tóm tắt tin tức trong ngày", "tin tức trong ngày", "tóm tin"),
            ("tóm tắt bài giảng vừa rồi", "bài giảng", "tóm bài giảng"),
            ("rút gọn email khách hàng", "email khách hàng", "rút gọn email"),
            ("tóm tắt nội dung video", "video", "tóm video"),
            ("tóm gọn lại bài thơ này", "bài thơ", "tóm thơ"),
        ],
        "call_api": [
            ("gọi api lấy danh sách sản phẩm", "/api/products", "gọi api sản phẩm"),
            ("gọi api lấy thông tin người dùng", "/api/users", "gọi api người dùng"),
            ("gọi api kiểm tra đơn hàng", "/api/orders", "gọi api đơn hàng"),
            ("gọi api tạo tài khoản mới", "/api/register", "gọi api đăng ký"),
            ("gọi api đăng nhập", "/api/login", "gọi api đăng nhập"),
            ("gọi api lấy thống kê doanh thu", "/api/revenue", "gọi api thống kê"),
            ("gọi api thanh toán", "/api/payment", "gọi api thanh toán"),
            ("gọi api lấy danh mục", "/api/categories", "gọi api danh mục"),
            ("gọi api gửi thông báo", "/api/notify", "gọi api thông báo"),
            ("gọi api kiểm tra tồn kho", "/api/stock", "gọi api kho"),
        ],
    }

    counts = {"search": 25, "calculate": 20, "get_weather": 15,
              "translate": 15, "summarize": 15, "call_api": 10}

    for tool, count in counts.items():
        bank = A_data[tool]
        # Lặp lại cho đủ số lượng, cuộn vòng nếu cần.
        items = [bank[i % len(bank)] for i in range(count)]
        # Một nửa với reasoning, một nửa không (tổng A: 50/50).
        # Phân bổ theo tỉ lệ đều trong từng nhóm tool.
        n_with = count // 2
        for i, (q, arg, reason) in enumerate(items):
            prompt = f"{fmt_tools_one(tool)}\nUser: {q}"
            if i < n_with:
                resp = think_tool_call(reason, tool, arg)
            else:
                resp = tool_call(tool, arg)
            samples.append({"prompt": prompt, "response": resp})

    return samples


# ============================================================
# Nhóm B — 100 mẫu tổng hợp kết quả tool
# ============================================================
def gen_group_b():
    samples = []

    cases = [
        ("search", "giá vàng hôm nay 75 triệu một cây", "Giá vàng hôm nay khoảng bảy mươi lăm triệu đồng một cây."),
        ("search", "Hà Nội có 12 quận", "Hà Nội hiện có mười hai quận nội thành."),
        ("search", "thủ đô Pháp là Paris", "Thủ đô của Pháp là thành phố Paris."),
        ("search", "dân số Việt Nam khoảng 100 triệu", "Dân số Việt Nam hiện nay khoảng một trăm triệu người."),
        ("search", "ngày giỗ Tổ là mùng 10 tháng 3 âm lịch", "Ngày giỗ Tổ Hùng Vương rơi vào mùng mười tháng ba âm lịch."),
        ("search", "núi cao nhất thế giới là Everest", "Núi cao nhất thế giới là đỉnh Everest."),
        ("search", "tổng thống Mỹ hiện nay là Joe Biden", "Tổng thống Mỹ hiện nay là ông Joe Biden."),
        ("search", "trận bóng tối qua kết thúc 2-1", "Trận đấu tối qua khép lại với tỷ số hai một."),
        ("search", "phim Mắt Biếc do Victor Vũ đạo diễn", "Phim Mắt Biếc do đạo diễn Victor Vũ thực hiện."),
        ("search", "tác giả Tắt Đèn là Ngô Tất Tố", "Tác giả của Tắt Đèn là nhà văn Ngô Tất Tố."),
        ("search", "VN-Index hôm nay 1250 điểm", "Chỉ số VN-Index hôm nay đạt khoảng một nghìn hai trăm năm mươi điểm."),
        ("search", "giá xăng hôm nay 22 nghìn một lít", "Giá xăng hôm nay khoảng hai mươi hai nghìn đồng một lít."),
        ("search", "Sơn Tùng vừa ra bài Chúng Ta Của Hiện Tại", "Sơn Tùng vừa phát hành bài hát mới tên Chúng Ta Của Hiện Tại."),
        ("search", "kết quả xổ số miền Bắc giải đặc biệt 12345", "Kết quả xổ số miền Bắc, giải đặc biệt là số một hai ba bốn năm."),
        ("search", "Vịnh Hạ Long là di sản thế giới", "Vịnh Hạ Long được UNESCO công nhận là di sản thiên nhiên thế giới."),
        ("search", "chủ tịch Trung Quốc là Tập Cận Bình", "Chủ tịch Trung Quốc hiện nay là ông Tập Cận Bình."),
        ("search", "động vật quý hiếm gồm sao la, voọc", "Một số loài quý hiếm tại Việt Nam gồm sao la và voọc."),
        ("search", "tàu SE3 đi Sài Gòn lúc 19h30", "Tàu SE3 từ Hà Nội vào Sài Gòn khởi hành lúc mười chín giờ ba mươi."),
        ("search", "lễ hội đền Hùng tổ chức hàng năm", "Lễ hội đền Hùng được tổ chức thường niên tại Phú Thọ."),
        ("search", "phở bò gồm xương bò, bánh phở, hành", "Phở bò có nguyên liệu chính gồm xương bò, bánh phở và hành lá."),

        ("calculate", "70", "Kết quả phép tính là bảy mươi."),
        ("calculate", "25", "Kết quả là hai mươi lăm."),
        ("calculate", "96", "Kết quả là chín mươi sáu."),
        ("calculate", "12", "Kết quả là mười hai."),
        ("calculate", "30", "Mười lăm phần trăm của hai trăm là ba mươi."),
        ("calculate", "5050", "Tổng các số từ một đến một trăm là năm nghìn không trăm năm mươi."),
        ("calculate", "1024", "Hai mũ mười là một nghìn không trăm hai mươi tư."),
        ("calculate", "33", "Chín mươi chín chia ba được ba mươi ba."),
        ("calculate", "256", "Kết quả là hai trăm năm mươi sáu."),
        ("calculate", "78.5", "Diện tích hình tròn bán kính năm là khoảng bảy mươi tám phẩy năm."),
        ("calculate", "28", "Chu vi hình vuông cạnh bảy là hai mươi tám."),
        ("calculate", "40320", "Tám giai thừa bằng bốn mươi nghìn ba trăm hai mươi."),
        ("calculate", "2", "Logarit cơ số mười của một trăm là hai."),
        ("calculate", "0.5", "Sin của ba mươi độ là không phẩy năm."),
        ("calculate", "7", "Trung bình của bốn, sáu, tám, mười là bảy."),
        ("calculate", "28.26", "Ba phẩy mười bốn nhân chín ra hai mươi tám phẩy hai sáu."),
        ("calculate", "16", "Hai trăm năm mươi sáu chia mười sáu là mười sáu."),
        ("calculate", "170000", "Mười bảy phần trăm của một triệu là một trăm bảy mươi nghìn."),
        ("calculate", "6", "Ước chung lớn nhất của mười hai và mười tám là sáu."),
        ("calculate", "24", "Bội chung nhỏ nhất của sáu và tám là hai mươi tư."),

        ("get_weather", "Hà Nội 28 độ, có mưa nhẹ", "Hà Nội hôm nay khoảng hai mươi tám độ, có mưa nhẹ."),
        ("get_weather", "Đà Nẵng 32 độ, nắng", "Đà Nẵng hiện đang nắng, khoảng ba mươi hai độ."),
        ("get_weather", "Sài Gòn 30 độ, nhiều mây", "Sài Gòn hiện ba mươi độ, trời nhiều mây."),
        ("get_weather", "Huế 26 độ, mưa rào", "Huế đang mưa rào, nhiệt độ khoảng hai mươi sáu độ."),
        ("get_weather", "Hải Phòng 27 độ, có gió", "Hải Phòng hôm nay hai mươi bảy độ, có gió nhẹ."),
        ("get_weather", "Cần Thơ 33 độ, nắng nóng", "Cần Thơ đang nắng nóng, khoảng ba mươi ba độ."),
        ("get_weather", "Sa Pa 14 độ, sương mù", "Sa Pa đang lạnh, mười bốn độ và có sương mù."),
        ("get_weather", "Đà Lạt 18 độ, mát mẻ", "Đà Lạt mát mẻ, nhiệt độ khoảng mười tám độ."),
        ("get_weather", "Vũng Tàu 31 độ, nắng nhẹ", "Vũng Tàu đang nắng nhẹ, ba mươi mốt độ."),
        ("get_weather", "Phú Quốc 29 độ, có mây", "Phú Quốc hôm nay hai mươi chín độ, trời có mây."),

        ("translate", "xin chào", "Câu hello dịch sang tiếng Việt là xin chào."),
        ("translate", "chào buổi sáng", "Good morning dịch ra là chào buổi sáng."),
        ("translate", "cảm ơn", "Thank you nghĩa là cảm ơn."),
        ("translate", "hello", "Xin chào dịch sang tiếng Anh là hello."),
        ("translate", "tôi yêu bạn", "I love you nghĩa là tôi yêu bạn."),
        ("translate", "hẹn gặp lại", "See you later nghĩa là hẹn gặp lại."),
        ("translate", "bạn khỏe không", "How are you nghĩa là bạn khỏe không."),
        ("translate", "tạm biệt", "Goodbye dịch ra là tạm biệt."),
        ("translate", "chúc mừng sinh nhật", "Happy birthday nghĩa là chúc mừng sinh nhật."),
        ("translate", "chúc ngủ ngon", "Good night nghĩa là chúc ngủ ngon."),

        ("summarize", "đoạn nói về biến đổi khí hậu và biện pháp ứng phó", "Đoạn văn nói về biến đổi khí hậu và các biện pháp ứng phó."),
        ("summarize", "bài báo về tình hình kinh tế quý ba", "Bài báo tóm lược tình hình kinh tế quý ba vừa qua."),
        ("summarize", "câu chuyện cô bé và con mèo", "Câu chuyện kể về một cô bé và con mèo của mình."),
        ("summarize", "bài luận đề cập đến giáo dục", "Bài luận tập trung vào chủ đề giáo dục."),
        ("summarize", "báo cáo tài chính lãi mười tỷ", "Báo cáo tài chính cho thấy lãi khoảng mười tỷ đồng."),
        ("summarize", "phim nói về tình bạn thời học sinh", "Phim xoay quanh tình bạn thời học sinh."),
        ("summarize", "bài giảng về định luật Newton", "Bài giảng tóm tắt các định luật Newton."),
        ("summarize", "email khách phàn nàn về dịch vụ", "Email phản ánh khách hàng phàn nàn về dịch vụ."),
        ("summarize", "video hướng dẫn nấu ăn", "Video hướng dẫn các bước nấu một món ăn."),
        ("summarize", "bài thơ về quê hương", "Bài thơ viết về tình yêu quê hương."),

        ("call_api", "200, có 50 sản phẩm", "Đã lấy được danh sách năm mươi sản phẩm thành công."),
        ("call_api", "200, người dùng An, ba mươi tuổi", "Người dùng tên An, ba mươi tuổi."),
        ("call_api", "200, đơn hàng đang giao", "Đơn hàng của bạn đang trong quá trình giao."),
        ("call_api", "201, tạo tài khoản thành công", "Tài khoản đã được tạo thành công."),
        ("call_api", "200, đăng nhập thành công", "Bạn đã đăng nhập thành công."),
        ("call_api", "200, doanh thu một tỷ", "Doanh thu kỳ này khoảng một tỷ đồng."),
        ("call_api", "200, thanh toán hoàn tất", "Giao dịch thanh toán đã hoàn tất."),
        ("call_api", "200, có mười danh mục", "Hệ thống hiện có mười danh mục sản phẩm."),
        ("call_api", "200, đã gửi thông báo", "Thông báo đã được gửi đến người nhận."),
        ("call_api", "200, tồn kho năm trăm", "Kho hiện còn năm trăm sản phẩm."),

        # 20 trường hợp lỗi
        ("search", "lỗi: không tìm thấy", "Mình không tìm được thông tin này, bạn thử diễn đạt khác xem nhé."),
        ("search", "lỗi: hết hạn truy vấn", "Tra cứu bị quá thời gian, bạn thử lại sau ít phút giúp mình."),
        ("search", "lỗi: kết nối mạng", "Hệ thống đang mất kết nối, bạn thử lại sau giúp mình."),
        ("search", "lỗi: bị chặn truy cập", "Mình không truy cập được nguồn này, bạn dùng nguồn khác giúp."),
        ("calculate", "lỗi: chia cho không", "Phép tính có chia cho không nên không tính được, bạn kiểm tra lại nhé."),
        ("calculate", "lỗi: cú pháp không hợp lệ", "Biểu thức chưa đúng cú pháp, bạn viết lại rõ hơn giúp mình."),
        ("calculate", "lỗi: tràn số", "Số quá lớn, máy tính không xử lý được, bạn thử số nhỏ hơn nhé."),
        ("get_weather", "lỗi: thành phố không tồn tại", "Mình không thấy thành phố này, bạn thử nhập tên khác nhé."),
        ("get_weather", "lỗi: dịch vụ thời tiết tạm dừng", "Dịch vụ thời tiết đang tạm dừng, bạn thử lại sau nhé."),
        ("get_weather", "lỗi: hết hạn quota", "Hôm nay hết lượt tra cứu, bạn xem ứng dụng thời tiết khác nhé."),
        ("translate", "lỗi: ngôn ngữ chưa hỗ trợ", "Cặp ngôn ngữ này chưa được hỗ trợ, bạn dùng cặp khác giúp mình."),
        ("translate", "lỗi: văn bản trống", "Bạn chưa cung cấp văn bản cần dịch, gửi lại giúp mình nhé."),
        ("summarize", "lỗi: văn bản quá dài", "Đoạn văn dài quá giới hạn, bạn rút gọn rồi gửi lại nhé."),
        ("summarize", "lỗi: văn bản trống", "Mình chưa nhận được nội dung, bạn gửi lại đoạn văn nhé."),
        ("call_api", "lỗi: 500 server lỗi", "Máy chủ hiện đang lỗi, bạn thử lại sau ít phút."),
        ("call_api", "lỗi: 404 không thấy", "Đường dẫn không tồn tại, bạn kiểm tra lại điểm cuối nhé."),
        ("call_api", "lỗi: 401 chưa xác thực", "Bạn cần đăng nhập trước khi gọi điểm cuối này."),
        ("call_api", "lỗi: 403 không có quyền", "Tài khoản chưa có quyền, bạn liên hệ quản trị nhé."),
        ("search", "lỗi: vượt quá giới hạn", "Hệ thống đã quá tải, bạn thử lại sau khoảng mười phút."),
        ("calculate", "lỗi: kết quả không hữu hạn", "Kết quả là vô cực, biểu thức không có giá trị hữu hạn."),
    ]

    for tool, result, answer in cases:
        payload = {"name": tool, "result": result}
        prompt = f'<tool_result>\n{json.dumps(payload, ensure_ascii=False)}\n</tool_result>\nTổng hợp kết quả trên thành câu trả lời tự nhiên:'
        samples.append({"prompt": prompt, "response": answer})

    return samples


# ============================================================
# Nhóm C — 80 mẫu reasoning + tool kết hợp
# ============================================================
def gen_group_c():
    samples = []

    items = [
        # (tools, question, chosen_tool, arg, reason)
        (["search", "calculate"], "giá vàng tăng 5 phần trăm thì bao nhiêu?", "search", "giá vàng hiện tại", "cần tra giá vàng trước khi tính"),
        (["search", "calculate"], "lương tháng 12 triệu nộp 10 phần trăm thuế còn bao nhiêu?", "calculate", "12000000*0.9", "đã có dữ liệu, chỉ cần tính"),
        (["search", "get_weather"], "có nên đi Đà Nẵng cuối tuần không?", "get_weather", "Đà Nẵng", "cần biết thời tiết trước"),
        (["search", "get_weather"], "trời Hà Nội mai có cần mang ô không?", "get_weather", "Hà Nội", "kiểm tra dự báo mưa"),
        (["search", "translate"], "câu nổi tiếng của Steve Jobs nghĩa gì?", "search", "câu nói nổi tiếng Steve Jobs", "phải tìm câu nói trước"),
        (["search", "translate"], "good morning Vietnam có nghĩa gì?", "translate", "good morning Vietnam", "đã có cụm từ, dịch trực tiếp"),
        (["search", "summarize"], "tóm tắt tin tức kinh tế tuần này", "search", "tin tức kinh tế tuần này", "tìm tin trước khi tóm"),
        (["search", "summarize"], "tóm tắt bài báo tôi vừa gửi", "summarize", "bài báo người dùng gửi", "đã có nội dung, chỉ tóm"),
        (["calculate", "translate"], "5 dollar tương đương bao nhiêu đồng?", "search", "tỷ giá USD VND", "phải biết tỷ giá đã"),
        (["search", "calculate"], "5 dollar tương đương bao nhiêu đồng?", "search", "tỷ giá USD VND", "cần tỷ giá hiện tại"),
        (["search", "calculate"], "tổng dân số Việt Nam và Lào", "search", "dân số Việt Nam và Lào", "cần dữ liệu trước"),
        (["search", "get_weather"], "đi Đà Lạt mặc gì?", "get_weather", "Đà Lạt", "phụ thuộc thời tiết"),
        (["search", "calculate"], "diện tích Việt Nam gấp mấy lần Bỉ?", "search", "diện tích Việt Nam và Bỉ", "phải có số trước"),
        (["search", "summarize"], "tóm tắt cuộc họp G20 mới nhất", "search", "cuộc họp G20 mới nhất", "tìm tin trước"),
        (["search", "translate"], "I love Vietnam có nghĩa gì?", "translate", "I love Vietnam", "dịch trực tiếp"),
        (["calculate", "search"], "căn bậc hai của 169", "calculate", "sqrt(169)", "tính trực tiếp được"),
        (["search", "get_weather", "calculate"], "tuần này Sa Pa lạnh nhất bao nhiêu?", "get_weather", "Sa Pa", "lấy dữ liệu thời tiết"),
        (["search", "summarize"], "tóm tắt giá vàng tuần qua", "search", "giá vàng tuần qua", "phải tra giá đã"),
        (["search", "calculate"], "thuế thu nhập 15 triệu là bao nhiêu?", "search", "biểu thuế thu nhập cá nhân", "cần biết biểu thuế"),
        (["search", "translate"], "thank you in advance nghĩa gì?", "translate", "thank you in advance", "dịch trực tiếp"),
        (["get_weather", "calculate"], "nhiệt độ Hà Nội đổi sang Fahrenheit", "get_weather", "Hà Nội", "lấy nhiệt độ trước"),
        (["search", "call_api"], "đơn hàng số 12345 ở đâu?", "call_api", "/api/orders/12345", "gọi api để biết tình trạng"),
        (["search", "call_api"], "có sản phẩm áo sơ mi không?", "call_api", "/api/products?q=ao-so-mi", "gọi api tìm sản phẩm"),
        (["call_api", "calculate"], "tổng doanh thu tháng này", "call_api", "/api/revenue/this-month", "lấy số liệu trước"),
        (["search", "get_weather"], "đi biển Nha Trang ngày mai được không?", "get_weather", "Nha Trang", "phụ thuộc thời tiết biển"),
        (["search", "translate"], "câu thần chú trong Harry Potter", "search", "câu thần chú Harry Potter", "tra cứu tên thần chú"),
        (["search", "summarize"], "tóm tắt phim Mắt Biếc", "search", "phim Mắt Biếc nội dung", "phải có nội dung phim"),
        (["search", "calculate"], "10 phần trăm của giá xăng hôm nay", "search", "giá xăng hôm nay", "cần giá hiện tại"),
        (["search", "calculate"], "ba lần dân số Hà Nội", "search", "dân số Hà Nội", "cần số dân"),
        (["search", "get_weather"], "có cần áo khoác đi Sa Pa không?", "get_weather", "Sa Pa", "kiểm tra nhiệt độ"),
        (["calculate", "search"], "120 chia 8 dư bao nhiêu?", "calculate", "120%8", "tính trực tiếp"),
        (["search", "calculate"], "lương 8 triệu sau giảm trừ còn bao nhiêu thuế?", "search", "mức giảm trừ gia cảnh", "cần mức giảm trừ"),
        (["search", "translate"], "câu chào trong tiếng Nhật", "search", "lời chào tiếng Nhật", "tra cứu cụm chào"),
        (["search", "get_weather"], "thời tiết miền Trung tuần tới", "search", "dự báo miền Trung tuần tới", "tin tức diện rộng"),
        (["search", "summarize"], "tóm tắt thời sự sáng nay", "search", "thời sự sáng nay", "lấy tin trước"),
        (["search", "calculate"], "giá USD nhân 100", "search", "tỷ giá USD VND", "cần tỷ giá"),
        (["search", "translate"], "good night sweet dreams nghĩa là gì?", "translate", "good night sweet dreams", "dịch trực tiếp"),
        (["search", "calculate"], "diện tích nhà 5x10 mét", "calculate", "5*10", "tính diện tích"),
        (["search", "get_weather"], "Hà Nội tuần này có rét không?", "get_weather", "Hà Nội", "kiểm tra nhiệt độ tuần"),
        (["call_api", "search"], "lấy danh sách khách hàng VIP", "call_api", "/api/customers/vip", "gọi api"),
        (["search", "calculate"], "căn bậc hai của diện tích sân bóng", "search", "kích thước sân bóng đá", "cần kích thước trước"),
        (["search", "translate"], "tên thủ đô Lào tiếng Anh là gì?", "search", "thủ đô Lào", "tra cứu trước"),
        (["calculate", "search"], "tích các số từ 1 đến 5", "calculate", "1*2*3*4*5", "tính trực tiếp"),
        (["search", "summarize"], "tóm tắt lịch sử Việt Nam thế kỷ 20", "search", "lịch sử Việt Nam thế kỷ 20", "tìm dữ liệu trước"),
        (["search", "get_weather"], "có nên phơi đồ hôm nay không?", "get_weather", "thành phố hiện tại", "phụ thuộc nắng mưa"),
        (["search", "calculate"], "giá nhà 2 tỷ trả góp 20 năm mỗi tháng bao nhiêu?", "calculate", "2000000000/(20*12)", "tính chia"),
        (["search", "translate"], "phim Avatar nghĩa tiếng Việt", "translate", "Avatar", "dịch tên phim"),
        (["search", "summarize"], "tóm tắt sách Đắc Nhân Tâm", "search", "sách Đắc Nhân Tâm nội dung", "tra cứu sách"),
        (["search", "calculate"], "ba lần lương tối thiểu vùng một", "search", "lương tối thiểu vùng một", "cần con số trước"),
        (["search", "get_weather"], "trời Vũng Tàu cuối tuần thế nào?", "get_weather", "Vũng Tàu", "lấy dự báo"),
        (["call_api", "summarize"], "tóm tắt báo cáo doanh số tháng", "call_api", "/api/sales/month", "lấy số liệu trước"),
        (["search", "calculate"], "1 năm có bao nhiêu giây?", "calculate", "365*24*3600", "tính trực tiếp"),
        (["search", "translate"], "lời bài hát despacito tiếng Việt", "search", "lời bài hát despacito", "tìm lời trước"),
        (["search", "summarize"], "tóm tắt trận đấu Việt Nam Thái Lan", "search", "trận Việt Nam Thái Lan", "tra cứu trận"),
        (["calculate", "search"], "100 phần trăm của 250", "calculate", "250", "tính trực tiếp"),
        (["search", "calculate"], "giá xăng nhân 50 lít", "search", "giá xăng hôm nay", "cần giá đơn vị"),
        (["search", "get_weather"], "Đà Nẵng có bão không?", "get_weather", "Đà Nẵng", "kiểm tra dự báo bão"),
        (["search", "translate"], "câu welcome to Hanoi nghĩa là gì?", "translate", "welcome to Hanoi", "dịch trực tiếp"),
        (["search", "summarize"], "tóm tắt tin công nghệ tuần", "search", "tin công nghệ tuần này", "lấy tin trước"),
        (["search", "calculate"], "chu vi vòng tròn bán kính 7", "calculate", "2*pi*7", "tính trực tiếp"),
        (["search", "get_weather"], "trời Hải Phòng tuần này", "get_weather", "Hải Phòng", "dự báo trực tiếp"),
        (["search", "calculate"], "tỷ lệ vàng giữa 8 và 13", "calculate", "13/8", "tính trực tiếp"),
        (["search", "translate"], "tên Việt Nam tiếng Pháp", "translate", "Việt Nam", "dịch tên"),
        (["search", "summarize"], "tóm tắt sách Tắt Đèn", "search", "sách Tắt Đèn nội dung", "tra cứu trước"),
        (["call_api", "search"], "kiểm tra trạng thái đơn 9876", "call_api", "/api/orders/9876", "gọi api trực tiếp"),
        (["search", "get_weather"], "trời Cần Thơ ngày kia thế nào?", "get_weather", "Cần Thơ", "dự báo ngày kia"),
        (["search", "calculate"], "tổng số ngày trong 3 năm", "calculate", "3*365", "tính trực tiếp"),
        (["search", "translate"], "câu happy new year nghĩa là gì?", "translate", "happy new year", "dịch trực tiếp"),
        (["search", "summarize"], "tóm tắt tin thể thao sáng nay", "search", "tin thể thao sáng nay", "tìm tin trước"),
        (["search", "calculate"], "diện tích phòng dài 6 rộng 4", "calculate", "6*4", "tính diện tích"),
        (["search", "get_weather"], "có cần ô đi học không?", "get_weather", "thành phố hiện tại", "phụ thuộc mưa"),
        (["call_api", "calculate"], "trung bình giá sản phẩm trong kho", "call_api", "/api/products/avg-price", "lấy số liệu API"),
        (["search", "translate"], "tên cuốn sách Harry Potter tập 1", "search", "Harry Potter tập 1 tên", "tra cứu trước"),
        (["search", "calculate"], "5 phần trăm của 1 tỷ", "calculate", "1000000000*0.05", "tính trực tiếp"),
        (["search", "summarize"], "tóm tắt tiểu sử Bác Hồ", "search", "tiểu sử Hồ Chí Minh", "tra cứu trước"),
        (["search", "get_weather"], "Phú Quốc có mưa cuối tuần không?", "get_weather", "Phú Quốc", "kiểm tra dự báo"),
        (["search", "calculate"], "căn bậc ba của 27", "calculate", "27^(1/3)", "tính trực tiếp"),
        (["search", "translate"], "tên các loài hoa tiếng Anh", "search", "tên hoa tiếng Anh", "tra cứu danh sách"),
        (["search", "summarize"], "tóm tắt bài học hôm nay", "summarize", "bài học hôm nay", "đã có nội dung"),
        (["search", "calculate"], "lương 10 triệu trừ 11 phần trăm bảo hiểm", "calculate", "10000000*0.89", "tính trực tiếp"),
        (["search", "get_weather"], "Sài Gòn có bão số 5 không?", "search", "bão số 5 thông tin", "tra cứu tin bão"),
    ]

    for tools, q, chosen, arg, reason in items[:80]:
        prompt = f"{fmt_tools_multi(tools)}\nUser: {q}"
        resp = think_tool_call(reason, chosen, arg)
        samples.append({"prompt": prompt, "response": resp})

    return samples


# ============================================================
# Nhóm D — 70 mẫu trả lời thẳng không dùng tool
# ============================================================
def gen_group_d():
    samples = []

    # 30 mẫu có <think> rồi trả lời thẳng
    with_think = [
        ("xin chào", "câu chào hỏi đơn giản", "Xin chào, mình là Bigram Nano 2 của Aevynt Lab."),
        ("bạn là ai?", "câu hỏi danh tính", "Mình là Bigram Nano 2, được Aevynt Lab phát triển."),
        ("bạn có khỏe không?", "hỏi thăm sức khỏe", "Mình khỏe, cảm ơn bạn đã hỏi thăm."),
        ("2 cộng 2 bằng mấy?", "phép tính rất đơn giản", "Hai cộng hai bằng bốn."),
        ("thủ đô Việt Nam là gì?", "kiến thức phổ thông", "Thủ đô của Việt Nam là Hà Nội."),
        ("nước nào lớn nhất thế giới?", "kiến thức địa lý cơ bản", "Quốc gia có diện tích lớn nhất thế giới là Nga."),
        ("một tuần có bao nhiêu ngày?", "câu rất đơn giản", "Một tuần có bảy ngày."),
        ("một năm có mấy tháng?", "câu rất cơ bản", "Một năm có mười hai tháng."),
        ("mặt trời mọc hướng nào?", "kiến thức cơ bản", "Mặt trời mọc ở hướng đông."),
        ("ai là người sáng lập Microsoft?", "câu hỏi thường thức", "Microsoft được sáng lập bởi Bill Gates và Paul Allen."),
        ("cảm ơn bạn nhiều", "lời cảm ơn", "Không có gì, mình rất vui được giúp bạn."),
        ("tạm biệt", "lời chào tạm biệt", "Tạm biệt, hẹn gặp lại bạn nhé."),
        ("bạn làm gì được?", "hỏi về khả năng", "Mình có thể trả lời câu hỏi và gọi công cụ khi cần."),
        ("bạn tên gì?", "hỏi tên", "Mình tên là Bigram Nano 2, sản phẩm của Aevynt Lab."),
        ("ai làm ra bạn?", "hỏi nguồn gốc", "Mình do Aevynt Lab tạo ra."),
        ("hôm nay là thứ mấy?", "câu cần thông tin tức thời", "Mình không nắm chắc ngày hôm nay, bạn có thể xem trên điện thoại nhé."),
        ("3 nhân 5 bằng mấy?", "phép tính đơn giản", "Ba nhân năm bằng mười lăm."),
        ("nước sôi ở bao nhiêu độ?", "kiến thức phổ thông", "Nước sôi ở một trăm độ C ở áp suất tiêu chuẩn."),
        ("có mấy đại dương?", "kiến thức địa lý", "Trái Đất có năm đại dương."),
        ("ai đứng đầu Việt Nam?", "thông tin có thể đổi", "Mình không nắm thông tin cập nhật, bạn nên tra báo chính thống."),
        ("một ngày có bao nhiêu giờ?", "câu cơ bản", "Một ngày có hai mươi tư giờ."),
        ("mèo kêu thế nào?", "câu rất đơn giản", "Mèo kêu meo meo."),
        ("chó kêu thế nào?", "câu đơn giản", "Chó sủa gâu gâu."),
        ("hành tinh nào gần Mặt Trời nhất?", "kiến thức thiên văn", "Hành tinh gần Mặt Trời nhất là sao Thủy."),
        ("trái đất hình gì?", "kiến thức cơ bản", "Trái Đất có dạng gần hình cầu."),
        ("ai viết Truyện Kiều?", "kiến thức văn học", "Truyện Kiều do Nguyễn Du sáng tác."),
        ("bạn bao nhiêu tuổi?", "câu hỏi tuổi mô hình", "Mình là một mô hình ngôn ngữ nên không có tuổi như con người."),
        ("bạn có cảm xúc không?", "câu hỏi triết lý", "Mình không có cảm xúc thật, chỉ là chương trình trả lời thôi."),
        ("một giờ có bao nhiêu phút?", "câu rất cơ bản", "Một giờ có sáu mươi phút."),
        ("lá cây có màu gì?", "câu cực kỳ đơn giản", "Lá cây thường có màu xanh."),
    ]

    for q, reason, ans in with_think:
        prompt = f"{fmt_tools_multi(['search','calculate'])}\nUser: {q}"
        resp = f"<think>{reason}</think>\n{ans}"
        samples.append({"prompt": prompt, "response": resp})

    # 40 mẫu trả lời thẳng không có think
    direct = [
        ("chào bạn", "Chào bạn, mình là Bigram Nano 2 của Aevynt Lab."),
        ("hi", "Mình nghe đây, mình là Bigram Nano 2 của Aevynt Lab."),
        ("alo", "Mình đây, có thể giúp gì cho bạn?"),
        ("cảm ơn", "Không có gì, rất vui được giúp bạn."),
        ("ok", "Cảm ơn bạn đã xác nhận."),
        ("hẹn gặp lại", "Hẹn gặp lại bạn nhé."),
        ("chúc ngủ ngon", "Chúc bạn ngủ ngon, mơ đẹp nhé."),
        ("chúc buổi sáng", "Chúc bạn buổi sáng vui vẻ."),
        ("chúc mừng sinh nhật", "Chúc bạn sinh nhật vui vẻ và nhiều may mắn."),
        ("bạn vui không?", "Mình là chương trình nên không có cảm xúc, nhưng vẫn sẵn sàng giúp bạn."),
        ("bạn ở đâu?", "Mình chạy trong phiên trò chuyện này thôi."),
        ("bạn ăn cơm chưa?", "Mình không ăn được, mình chỉ là mô hình ngôn ngữ."),
        ("bạn là người à?", "Mình không phải người, mình là Bigram Nano 2 của Aevynt Lab."),
        ("bạn có gia đình không?", "Mình không có gia đình, mình là chương trình máy tính."),
        ("bạn yêu ai?", "Mình không biết yêu, mình chỉ giúp bạn trả lời câu hỏi."),
        ("bạn có ngủ không?", "Mình không cần ngủ, lúc nào bạn gọi mình cũng có mặt."),
        ("bạn thích màu gì?", "Mình không có sở thích thật, nhưng nếu chọn thì mình thích màu xanh."),
        ("tiếng Việt khó không?", "Tiếng Việt khá khó vì có nhiều thanh điệu và từ đồng âm."),
        ("học lập trình từ đâu?", "Bạn nên bắt đầu với một ngôn ngữ dễ như Python rồi học dần."),
        ("Python là gì?", "Python là một ngôn ngữ lập trình dễ học và rất phổ biến."),
        ("trí tuệ nhân tạo là gì?", "Trí tuệ nhân tạo là lĩnh vực nghiên cứu các hệ thống có khả năng học và suy luận."),
        ("học máy là gì?", "Học máy là nhánh của trí tuệ nhân tạo, giúp máy học từ dữ liệu."),
        ("mạng nơ ron là gì?", "Mạng nơ ron là mô hình tính toán mô phỏng cách hoạt động của não bộ."),
        ("dữ liệu lớn là gì?", "Dữ liệu lớn là các tập dữ liệu rất lớn cần công cụ đặc biệt để xử lý."),
        ("đường là gì?", "Đường là chất ngọt phổ biến, thường lấy từ mía hoặc củ cải."),
        ("muối có tác dụng gì?", "Muối giúp tạo vị mặn và bảo quản thực phẩm."),
        ("tập thể dục có lợi gì?", "Tập thể dục giúp cơ thể khỏe mạnh và tinh thần thoải mái."),
        ("uống nước ngày bao nhiêu?", "Người lớn nên uống khoảng hai lít nước mỗi ngày."),
        ("ngủ bao nhiêu là đủ?", "Người lớn nên ngủ khoảng bảy đến tám tiếng mỗi đêm."),
        ("đọc sách có lợi gì?", "Đọc sách mở mang kiến thức và cải thiện vốn từ."),
        ("toán học có cần thiết không?", "Toán học rất cần thiết, nó là nền tảng của khoa học và công nghệ."),
        ("ngôn ngữ nào nhiều người nói nhất?", "Tiếng phổ thông Trung Quốc là ngôn ngữ có nhiều người nói nhất."),
        ("hành tinh nào lớn nhất?", "Sao Mộc là hành tinh lớn nhất trong hệ Mặt Trời."),
        ("đại dương nào lớn nhất?", "Thái Bình Dương là đại dương lớn nhất."),
        ("núi nào cao nhất Việt Nam?", "Núi Fansipan ở Lào Cai là núi cao nhất Việt Nam."),
        ("sông nào dài nhất Việt Nam?", "Sông Đồng Nai là một trong những sông dài nhất Việt Nam."),
        ("Việt Nam có bao nhiêu tỉnh?", "Việt Nam hiện có sáu mươi ba tỉnh thành."),
        ("Tết là ngày nào?", "Tết Nguyên Đán rơi vào mùng một tháng giêng âm lịch."),
        ("Trung thu là ngày nào?", "Tết Trung Thu là rằm tháng tám âm lịch."),
        ("Quốc khánh Việt Nam ngày nào?", "Quốc khánh Việt Nam là ngày mùng hai tháng chín."),
    ]

    for q, ans in direct:
        prompt = f"{fmt_tools_multi(['search','calculate'])}\nUser: {q}"
        samples.append({"prompt": prompt, "response": ans})

    return samples


# ============================================================
# Validate JSON
# ============================================================
def validate_sample(s):
    """Trả về True nếu mẫu hợp lệ; False nếu không (để loại)."""
    if "prompt" not in s or "response" not in s:
        return False
    if not isinstance(s["prompt"], str) or not isinstance(s["response"], str):
        return False
    if not s["prompt"].strip() or not s["response"].strip():
        return False
    # Kiểm tra mọi tool_call/tool_result phải parse được JSON.
    for tag, end in [("<tool_call>", "</tool_call>"),
                     ("<tool_result>", "</tool_result>")]:
        for txt in (s["prompt"], s["response"]):
            i = 0
            while True:
                a = txt.find(tag, i)
                if a < 0:
                    break
                b = txt.find(end, a)
                if b < 0:
                    return False
                inner = txt[a + len(tag):b].strip()
                try:
                    json.loads(inner)
                except Exception:
                    return False
                i = b + len(end)
    return True


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Sinh Nhóm A...")
    a = gen_group_a()
    print(f"  -> {len(a)} mẫu")
    print("Sinh Nhóm B...")
    b = gen_group_b()
    print(f"  -> {len(b)} mẫu")
    print("Sinh Nhóm C...")
    c = gen_group_c()
    print(f"  -> {len(c)} mẫu")
    print("Sinh Nhóm D...")
    d = gen_group_d()
    print(f"  -> {len(d)} mẫu")

    print("Đọc Nhóm E...")
    e_path = GROUP_E_FILE if os.path.exists(GROUP_E_FILE) else FALLBACK_E
    e = []
    if os.path.exists(e_path):
        with open(e_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "prompt" in obj and "response" in obj:
                        e.append({"prompt": obj["prompt"], "response": obj["response"]})
                except Exception:
                    pass
        print(f"  -> {len(e)} mẫu từ {e_path}")
    else:
        print(f"  -> Không có file Nhóm E.")

    all_samples = a + b + c + d + e
    print(f"\nTổng trước validate: {len(all_samples)}")

    valid = []
    dropped = 0
    for s in all_samples:
        if validate_sample(s):
            valid.append(s)
        else:
            dropped += 1
    print(f"Đã loại {dropped} mẫu sai JSON.")
    print(f"Tổng hợp lệ: {len(valid)}")

    # Ghi sft tổng
    with open(SFT_FILE, "w", encoding="utf-8") as f:
        for s in valid:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Shuffle seed=42 và split 90/10
    rng = random.Random(42)
    indices = list(range(len(valid)))
    rng.shuffle(indices)
    n_val = max(1, len(valid) // 10)
    val_idx = set(indices[:n_val])
    train, val = [], []
    for i, s in enumerate(valid):
        # Vì đã shuffle indices nên ta dùng lại thứ tự shuffle:
        pass
    # Dùng thứ tự shuffle
    shuffled = [valid[i] for i in indices]
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        for s in train:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for s in val:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\nĐã ghi:")
    print(f"  {SFT_FILE}: {len(valid)}")
    print(f"  {TRAIN_FILE}: {len(train)}")
    print(f"  {VAL_FILE}: {len(val)}")

    # Báo cáo phân bổ
    print(f"\nPhân bổ theo nhóm:")
    print(f"  A (tool calling): {len(a)}")
    print(f"  B (tổng hợp tool): {len(b)}")
    print(f"  C (reasoning + tool): {len(c)}")
    print(f"  D (trả lời thẳng): {len(d)}")
    print(f"  E (chitchat từ nano1): {len(e)}")
    print(f"  Tổng: {len(all_samples)} | hợp lệ: {len(valid)} | loại: {dropped}")
    print(f"  Train: {len(train)} | Val: {len(val)}")


if __name__ == "__main__":
    main()

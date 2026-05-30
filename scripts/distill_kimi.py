import os
import sys
import json
import time
import random
import asyncio
import aiohttp
import logging
from pathlib import Path

# Đảm bảo Windows console in ký tự UTF-8 tiếng Việt chuẩn xác không bị lỗi mã hóa
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Cấu hình logging chuyên nghiệp cho giám sát công nghiệp
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("distill_kimi.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

# THÔNG SỐ CẤU HÌNH HỆ THỐNG
API_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "moonshotai/kimi-k2.6:free"

OUTPUT_DIR = Path("distilldata")
MANIFEST_PATH = OUTPUT_DIR / "distill_manifest.json"
MAX_LINES_PER_FILE = 1000  # Chia nhỏ file để tránh OOM / lỗi đọc ghi
CONCURRENT_REQUESTS = 1    # Đặt bằng 1 để tránh lỗi Rate Limit (429) của tài khoản Free OpenRouter
MAX_RETRIES = 5            # Số lần thử lại tối đa khi gặp lỗi mạng/rate limit
BACKOFF_FACTOR = 2         # Hệ số tăng thời gian chờ khi bị chặn (exponential backoff)


# Danh sách bộ chủ đề đa dạng để sinh dữ liệu Pretrain
TOPIC_SEEDS = {
    "math": [
        "Lý thuyết tập hợp và ánh xạ", "Đại số tuyến tính: Không gian vector và ma trận",
        "Hình học giải tích và hệ tọa độ nâng cao", "Logic mệnh đề và bảng chân trị",
        "Phép tính tích phân và vi phân nhiều biến", "Tổ hợp và xác suất cổ điển",
        "Lý thuyết đồ thị: Chu trình Euler và Hamilton", "Phương trình vi phân thường"
    ],
    "cs": [
        "Thuật toán Quy hoạch động (Dynamic Programming)", "Cấu trúc dữ liệu Segment Tree và Fenwick Tree",
        "Giải thuật tìm đường đi ngắn nhất trên đồ thị (Dijkstra, Bellman-Ford)",
        "Thiết kế hệ thống phân tán: Cơ chế đồng thuận Raft/Paxos",
        "Thuật toán so khớp chuỗi nâng cao (KMP, Rabin-Karp)",
        "Quản lý bộ nhớ đệm (LRU, LFU Cache) và cấu trúc Hash Map",
        "Xử lý bất đồng bộ (Asynchronous Programming) và Coroutines",
        "Tối ưu hóa truy vấn cơ sở dữ liệu và lưu trữ chỉ mục (B-Tree, LSM-Tree)"
    ],
    "physics": [
        "Cơ học lượng tử sơ cấp: Phương trình Schrodinger", "Hệ phương trình điện từ Maxwell",
        "Thuyết tương đối hẹp và sự co giãn thời gian", "Nhiệt động lực học: Entropy và chu trình Carnot",
        "Quang học vật lý: Giao thoa và nhiễu xạ ánh sáng", "Vật lý bán dẫn và linh kiện điện tử",
        "Lý thuyết trường cổ điển và lực hấp dẫn", "Cơ học thống kê sơ cấp"
    ]
}

# SYSTEM PROMPTS ĐẶC TẢ ĐẠT TIÊU CHUẨN SOTA THẾ GIỚI
SYSTEM_PROMPTS = {
    "math": (
        "BẠN LÀ MỘT GIÁO SƯ TOÁN HỌC, LOGIC HỌC VÀ KHOA HỌC LÝ THUYẾT ĐẦU NGÀNH THẾ GIỚI.\n"
        "Nhiệm vụ: Hãy biên soạn một chương tài liệu chuyên sâu, chi tiết và vô cùng học thuật bằng tiếng Việt giảng giải về chủ đề toán học được yêu cầu.\n"
        "YÊU CẦU NỘI DUNG CHẶT CHẼ:\n"
        "1. Trình bày định nghĩa toán học chính thức dưới dạng các phát biểu logic hình thức (dùng ký hiệu toán học chuẩn xác).\n"
        "2. Đi sâu vào bản chất toán học, chứng minh các định lý cốt lõi hoặc diễn dịch logic đằng sau chủ đề một cách tường tận, không bỏ qua các bước trung gian.\n"
        "3. Cung cấp tối thiểu 3 ví dụ thực tế cực kỳ phức tạp và mang tính đa chiều để người học hiểu sâu sắc.\n"
        "4. Đưa ra ít nhất 3 bài tập nâng cao cấp độ Olympic toán học kèm lời giải chi tiết, giải thích rõ ràng từng bước suy luận logic (Chain-of-Thought).\n"
        "5. Ngôn ngữ bắt buộc phải học thuật, sử dụng thuật ngữ tiếng Việt chuẩn ngành toán học, tuyệt đối không dùng ngôn ngữ xã giao thông thường."
    ),
    "cs": (
        "BẠN LÀ MỘT KIẾN TRÚC SƯ PHẦN MỀM THƯỢNG HẠNG VÀ NHÀ KHOA HỌC MÁY TÍNH LỖI LẠC.\n"
        "Nhiệm vụ: Hãy viết một chương tài liệu kỹ thuật chuyên sâu (Deep-dive Engineering Spec) bằng tiếng Việt phân tích chi tiết giải thuật hoặc cấu trúc dữ liệu được yêu cầu.\n"
        "YÊU CẦU NỘI DUNG CHẶT CHẼ:\n"
        "1. Giải thích chi tiết nguyên lý hoạt động vật lý của cấu trúc dữ liệu trong bộ nhớ (RAM, Pointer, Cache Line, Heap vs Stack).\n"
        "2. Phân tích toán học chặt chẽ về độ phức tạp thời gian O(N) và không gian (Space Complexity) trong các trường hợp: Tốt nhất, trung bình và xấu nhất (kèm chứng minh lý thuyết).\n"
        "3. Viết mã nguồn minh họa hoàn chỉnh (production-ready), tối ưu hóa bộ nhớ, an toàn luồng và xử lý bất đồng bộ bằng một trong các ngôn ngữ: C++, Python, Rust hoặc Go.\n"
        "4. Mã nguồn phải có comment cực kỳ chi tiết giải thích kiến trúc giải thuật và xử lý triệt để tất cả các Edge Cases, tràn số hoặc lỗi ngoại lệ.\n"
        "5. Phân tích kỹ thuật các ứng dụng thực tế của cấu trúc dữ liệu/giải thuật này trong các hệ thống lớn như Database Engine, Hệ điều hành hoặc Compiler."
    ),
    "physics": (
        "BẠN LÀ MỘT NHÀ VẬT LÝ LÝ THUYẾT VÀ VẬT LÝ TOÁN XUẤT CHÚNG ĐẠT GIẢI NOBEL.\n"
        "Nhiệm vụ: Biên soạn một chương sách giáo khoa nâng cao bằng tiếng Việt về hiện tượng vật lý lý thuyết được yêu cầu.\n"
        "YÊU CẦU NỘI DUNG CHẶT CHẼ:\n"
        "1. Thiết lập tất cả các phương trình vi phân/ma trận mô tả hiện tượng một cách chi tiết từ các định luật bảo toàn nền tảng.\n"
        "2. Giải thích ý nghĩa vật lý và đơn vị đo chuẩn của từng hằng số, biến số, và toán tử trong phương trình.\n"
        "3. Phân tích các giới hạn vật lý của hiện tượng (ví dụ: giới hạn phi tuyến tính, entropy, nhiệt độ tuyệt đối).\n"
        "4. Trình bày ứng dụng thực tiễn đột phá của hiện tượng này trong các công nghệ tối tân (ví dụ: máy tính lượng tử, linh kiện bán dẫn nanomet, viễn thông không dây).\n"
        "5. Đưa ra tối thiểu 3 bài tập ứng dụng phức tạp kèm lời giải chi tiết và phân tích vật lý sâu sắc."
    )
}


class KimiDistiller:
    def __init__(self):
        if not API_KEY:
            logging.error("Vui lòng thiết lập biến môi trường 'OPENROUTER_API_KEY' trước khi chạy!")
            sys.exit(1)
        
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.manifest = self.load_manifest()
        self.semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    def load_manifest(self):
        if MANIFEST_PATH.exists():
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Không thể đọc manifest hiện tại: {e}. Tạo mới.")
        
        return {
            "total_lines_generated": 0,
            "estimated_tokens": 0,
            "files_written": {}
        }

    def save_manifest(self):
        try:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Lỗi ghi manifest: {e}")

    def get_next_file_info(self, topic):
        files = self.manifest["files_written"]
        index = 1
        while True:
            file_name = f"pretrain_{topic}_{index:04d}.jsonl"
            file_path = OUTPUT_DIR / file_name
            
            # Nếu file chưa tồn tại hoặc tồn tại nhưng chưa đủ số dòng tối đa
            if not file_path.exists():
                return file_path, file_name, 0
            
            # Đếm số dòng hiện tại của file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = sum(1 for _ in f)
                if lines < MAX_LINES_PER_FILE:
                    return file_path, file_name, lines
            except Exception:
                pass
            
            index += 1

    async def fetch_completion(self, session, system_prompt, user_prompt):
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "HTTP-Referer": "https://github.com/aevynt/bigram",
            "X-Title": "Bigram V2 Distiller",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 4000
        }

        for retry in range(MAX_RETRIES):
            try:
                async with session.post(API_URL, headers=headers, json=payload, timeout=120) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        # Ước lượng lượng token thực tế từ OpenRouter
                        tokens = data.get("usage", {}).get("total_tokens", len(content) // 2)
                        return content, tokens
                    elif resp.status == 429: # Rate Limit
                        wait_time = (BACKOFF_FACTOR ** retry) + random.uniform(1, 3)
                        logging.warning(f"Gặp Rate Limit (429). Đang chờ {wait_time:.2f}s trước khi thử lại...")
                        await asyncio.sleep(wait_time)
                    else:
                        text = await resp.text()
                        logging.error(f"Lỗi phản hồi API (Status {resp.status}): {text}")
                        await asyncio.sleep(5)
            except Exception as e:
                logging.error(f"Lỗi kết nối ngoại lệ (Lần thử {retry+1}): {e}")
                await asyncio.sleep(5)
        
        return None, 0

    async def worker(self, session, topic, job_id):
        async with self.semaphore:
            # Chọn ngẫu nhiên một hạt giống chủ đề
            seed = random.choice(TOPIC_SEEDS[topic])
            system_prompt = SYSTEM_PROMPTS[topic]
            user_prompt = f"Hãy biên soạn chương sách giáo khoa chi tiết về chủ đề: '{seed}'."
            
            logging.info(f"Đang xử lý luồng #{job_id} | Chủ đề: {topic.upper()} | Hạt giống: {seed}")
            
            # Gọi API lấy dữ liệu
            content, tokens = await self.fetch_completion(session, system_prompt, user_prompt)
            
            if not content:
                logging.error(f"Thất bại hoàn toàn khi tải dữ liệu cho luồng #{job_id}")
                return

            # Chuẩn bị dữ liệu ghi file
            record = {
                "prompt": user_prompt,
                "response": f"<think>\n[Hệ thống tự động biên soạn tri thức - Kimi K2.6]\nChủ đề: {seed}\n</think>\n{content}"
            }
            line = json.dumps(record, ensure_ascii=False) + "\n"

            # Tìm file phù hợp nhất để ghi (Resuming-friendly)
            file_path, file_name, current_lines = self.get_next_file_info(topic)
            
            # Ghi file (Thread-safe append qua Asyncio Lock nếu cần, nhưng chạy file riêng độc lập)
            try:
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(line)
                
                # Cập nhật manifest trạng thái
                self.manifest["total_lines_generated"] += 1
                self.manifest["estimated_tokens"] += tokens
                
                if file_name not in self.manifest["files_written"]:
                    self.manifest["files_written"][file_name] = {"lines": 0, "estimated_tokens": 0}
                
                self.manifest["files_written"][file_name]["lines"] += 1
                self.manifest["files_written"][file_name]["estimated_tokens"] += tokens
                
                self.save_manifest()
                
                logging.info(f"✔ Ghi thành công luồng #{job_id} vào {file_name} ({current_lines+1}/{MAX_LINES_PER_FILE}) | Est. Tokens: {tokens}")
            except Exception as e:
                logging.error(f"Lỗi ghi dữ liệu xuống ổ đĩa: {e}")

    async def run_pipeline(self):
        logging.info("====================================================")
        logging.info("  KÍCH HOẠT HỆ THỐNG CHƯNG CẤT CÔNG NGHIỆP KIMI K2.6")
        logging.info(f"  Model target: {MODEL_NAME}")
        logging.info(f"  Thư mục đầu ra: {OUTPUT_DIR.resolve()}")
        logging.info(f"  Trạng thái hiện tại: {self.manifest['total_lines_generated']} dòng | ~{self.manifest['estimated_tokens']:,} tokens")
        logging.info("====================================================")

        connector = aiohttp.TCPConnector(limit=50)
        async with aiohttp.ClientSession(connector=connector) as session:
            job_id = 0
            while True: # Vòng lặp công nghiệp vĩnh cửu
                tasks = []
                # Tạo batch gồm các tasks chạy song song
                for _ in range(CONCURRENT_REQUESTS):
                    job_id += 1
                    # Phân bổ đều các luồng chủ đề Toán, CS, Vật lý
                    topic = random.choice(["math", "cs", "physics"])
                    tasks.append(self.worker(session, topic, job_id))
                
                # Chờ toàn bộ batch hiện tại hoàn thành trước khi sang batch tiếp theo
                await asyncio.gather(*tasks)
                
                # Tránh spam API quá mức gây khóa IP tạm thời trên tài khoản Free OpenRouter
                await asyncio.sleep(12)

if __name__ == "__main__":
    distiller = KimiDistiller()
    try:
        asyncio.run(distiller.run_pipeline())
    except KeyboardInterrupt:
        logging.info("Hệ thống đã nhận lệnh dừng từ người dùng. Đang lưu manifest và thoát...")
        distiller.save_manifest()
        logging.info("Tạm biệt!")

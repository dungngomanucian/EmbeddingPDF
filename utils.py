import os
import re
import unicodedata
import datetime
import uuid
from dateutil import parser, tz
from config import Config, supabase

def clean_text(text: str) -> str:
    """Chuẩn hóa và làm sạch văn bản."""
    if not text: return ""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_file_identifier(filename: str) -> str:
    """
    Tạo ID định danh duy nhất cho nội dung file.
    Loại bỏ các hậu tố version/copy phổ biến để nhận diện trùng lặp thông minh.
    Ví dụ: "Quy che (1).pdf" -> "quy-che" (trùng với "Quy che.pdf")
    """
    name, _ = os.path.splitext(filename)
    
    # 1. Loại bỏ (số) ở cuối: "File (1)", "File(2)"
    name = re.sub(r'\s*\(\d+\)$', '', name)
    
    # 2. Loại bỏ - Copy, _copy, - Sao chép
    name = re.sub(r'\s*[-_]?\s*(copy|sao chép)$', '', name, flags=re.IGNORECASE)
    
    # 3. Loại bỏ v1, v2 (phiên bản) ở cuối
    name = re.sub(r'\s*[-_]?\s*v\d+$', '', name, flags=re.IGNORECASE)

    # 4. Chuẩn hóa slug
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    name = re.sub(r"[-\s]+", "-", name)
    return name

def slugify_filename(filename: str) -> str:
    """Tạo tên file an toàn lưu trữ (kèm timestamp)."""
    name, ext = os.path.splitext(filename)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    name = re.sub(r"[-\s]+", "-", name)
    ts = datetime.datetime.now(tz.gettz("Asia/Ho_Chi_Minh")).strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{ts}-{name}-{uid}{ext}"

def pretty_name(filename: str) -> str:
    """Chuyển tên file safe thành tên hiển thị đẹp."""
    try:
        name, ext = os.path.splitext(filename)
        parts = name.split("-")
        core = "-".join(parts[1:-1]) if len(parts) >= 3 else name
        return core.replace('-', ' ').replace('_', ' ').title() + ext
    except Exception:
        return filename

def format_time(iso_str: str) -> str:
    """Format thời gian ISO sang định dạng Việt Nam."""
    try:
        dt = parser.isoparse(iso_str).astimezone(tz.gettz("Asia/Ho_Chi_Minh"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return iso_str

def get_file_list():
    """Lấy danh sách file PDF từ Storage."""
    try:
        files = supabase.storage.from_(Config.BUCKET_NAME).list()
        file_list = [
            {"pretty_name": pretty_name(f["name"]), "safe_name": f["name"],
             "url": supabase.storage.from_(Config.BUCKET_NAME).get_public_url(f["name"]),
             "created_at": format_time(f.get("created_at", ""))}
            for f in files if f["name"].lower().endswith(".pdf")
        ]
        file_list.sort(key=lambda x: x['safe_name'], reverse=True)
        return file_list
    except Exception as e:
        print(f"Lỗi khi liệt kê file: {e}")
        return []

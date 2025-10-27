import os
import fitz # PyMuPDF
import re
import datetime
import uuid
import unicodedata
import numpy as np
import faiss
from dateutil import parser, tz
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from supabase import create_client, Client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import ast

# --- 1. KHỞI TẠO VÀ CẤU HÌNH ---
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET")
SECRET_KEY = os.getenv("SECRET_KEY", "a_very_secret_key_that_should_be_changed")

# Khởi tạo các client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__, template_folder="templates")
app.secret_key = SECRET_KEY

# Tải mô hình embedding (chỉ 1 lần)
print("Đang tải mô hình Sentence Transformer...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
print("Mô hình đã tải xong.")

# Khởi tạo biến toàn cục cho chỉ mục tìm kiếm
faiss_index = None
index_to_id = {}

# --- 2. CÁC HÀM XỬ LÝ DỮ LIỆU CỐT LÕI ---

def clean_text(text: str) -> str:
    """Chuẩn hóa và làm sạch văn bản thô từ PDF."""
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_faiss_index():
    """Tải tất cả vector từ Supabase và xây dựng chỉ mục FAISS trong bộ nhớ."""
    global faiss_index, index_to_id
    print("--- Bắt đầu xây dựng chỉ mục FAISS ---")
    try:
        response = supabase.table('documents').select('id, embedding').execute()
        docs = response.data
        if not docs:
            print("Cảnh báo: Không có tài liệu nào trong CSDL để xây dựng chỉ mục.")
            faiss_index = None
            return

        ids = [doc['id'] for doc in docs]
        embeddings_list = [ast.literal_eval(doc['embedding']) for doc in docs]
        embeddings = np.array(embeddings_list).astype('float32')
        index_to_id = {i: doc_id for i, doc_id in enumerate(ids)}

        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        faiss_index.add(embeddings)
        print(f"--- Đã xây dựng xong chỉ mục FAISS với {faiss_index.ntotal} vector ---")
    except Exception as e:
        print(f"!!! Lỗi nghiêm trọng khi xây dựng chỉ mục FAISS: {e}")
        faiss_index = None


def parse_and_chunk_pdf(pdf_bytes, safe_filename):
    """
    Cải tiến: Vẫn chia nhỏ PDF nhưng giờ đây mỗi chunk sẽ ghi nhớ được
    tiêu đề mục (heading) gần nhất chứa nó.
    """
    print(f"--- Bắt đầu chunking (hybrid approach) cho file: {safe_filename} ---")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    current_heading = "Lời nói đầu"
    current_text_buffer = ""

    # Mẫu regex để nhận diện các đầu mục/tiêu đề
    heading_pattern = re.compile(
        r"^(PHẦN\s+[\d\.]+|[IVXLCDM]+\s*\.|Chương\s+[IVXLCDM]+|Điều\s+\d+|Mục\s+\d+|[A-Z]\.|\d+\.\d*\.)",
        re.IGNORECASE
    )

    start_page = 6
    for page_num in range(start_page, len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")  # Lấy theo block để giữ cấu trúc tốt hơn
        for block in blocks:
            block_text = clean_text(block[4])
            if not block_text:
                continue

            # Kiểm tra xem block này có phải là một tiêu đề không
            if heading_pattern.match(block_text) and len(block_text) < 150:
                # Nếu có buffer cũ, lưu nó lại thành 1 chunk
                if current_text_buffer.strip():
                    chunks.append({
                        "content": f"Tiêu đề: {current_heading}\nNội dung: {current_text_buffer.strip()}",
                        "metadata": {"source_file": safe_filename, "section_title": current_heading}
                    })
                # Cập nhật tiêu đề mới và reset buffer
                current_heading = block_text
                current_text_buffer = ""
            else:
                current_text_buffer += block_text + "\n"

    # Lưu lại phần buffer cuối cùng
    if current_text_buffer.strip():
        chunks.append({
            "content": f"Tiêu đề: {current_heading}\nNội dung: {current_text_buffer.strip()}",
            "metadata": {"source_file": safe_filename, "section_title": current_heading}
        })

    print(f"--- Hoàn tất chunking. Tổng số chunks được tạo: {len(chunks)} ---")
    return chunks

# --- 3. CÁC HÀM HỖ TRỢ ---
def slugify_filename(filename: str) -> str:
    name, ext = os.path.splitext(filename)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = re.sub(r"[^\w\s-]", "", name).strip().lower()
    name = re.sub(r"[-\s]+", "-", name)
    ts = datetime.datetime.now(tz.gettz("Asia/Ho_Chi_Minh")).strftime("%Y%m%dT%H%M%S")
    uid = uuid.uuid4().hex[:6]
    return f"{ts}-{name}-{uid}{ext}"

def pretty_name(filename: str) -> str:
    try:
        name, ext = os.path.splitext(filename)
        parts = name.split("-")
        core = "-".join(parts[1:-1]) if len(parts) >= 3 else name
        return core.replace('-', ' ').replace('_', ' ').title() + ext
    except Exception:
        return filename

def format_time(iso_str: str) -> str:
    try:
        dt = parser.isoparse(iso_str).astimezone(tz.gettz("Asia/Ho_Chi_Minh"))
        return dt.strftime("%d/%m/%Y %H:%M")
    except Exception:
        return iso_str

def list_files():
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
        file_list = [
            {"pretty_name": pretty_name(f["name"]), "safe_name": f["name"],
             "url": supabase.storage.from_(BUCKET_NAME).get_public_url(f["name"]),
             "created_at": format_time(f.get("created_at", ""))}
            for f in files if f["name"].lower().endswith(".pdf")
        ]
        file_list.sort(key=lambda x: x['safe_name'], reverse=True)
        return file_list
    except Exception as e:
        print(f"Lỗi khi liệt kê file: {e}")
        return []

# --- 4. FLASK ROUTES ---
@app.route("/")
def home():
    return render_template("search_site.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        try:
            auth_response = supabase.auth.sign_in_with_password({"email": request.form.get("email"), "password": request.form.get("password")})
            session['user'] = {'id': auth_response.user.id, 'email': auth_response.user.email}
            return redirect(url_for('upload_page'))
        except Exception:
            flash("❌ Đăng nhập thất bại.", "error")
    return render_template("login_site.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email, password = request.form.get("email"), request.form.get("password")
        if password != request.form.get("confirm"):
            flash("Mật khẩu xác nhận không khớp.", "error")
            return redirect(url_for('signup'))
        try:
            supabase.auth.sign_up({"email": email, "password": password})
            flash("✅ Đăng ký thành công! Vui lòng kiểm tra email.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"❌ Lỗi: {'Email đã tồn tại' if 'already registered' in str(e) else e}", 'error')
    return render_template("signup_site.html")

@app.route("/logout")
def logout():
    session.pop('user', None)
    flash("Bạn đã đăng xuất.", "info")
    return redirect(url_for('home'))

@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    if 'user' not in session: return redirect(url_for('login'))
    if request.method == "POST":
        try:
            file = request.files.get("file")
            if not file or not file.filename:
                flash("❌ Vui lòng chọn một file để upload.", "error")
                return redirect(request.url)

            pdf_bytes = file.read()
            safe_pdf_name = slugify_filename(file.filename)

            # 1. Tải file lên Storage
            supabase.storage.from_(BUCKET_NAME).upload(safe_pdf_name, pdf_bytes, {"content-type": "application/pdf"})

            # 2. Phân đoạn PDF thành các chunks
            chunks = parse_and_chunk_pdf(pdf_bytes, safe_pdf_name)
            if not chunks:
                flash("⚠️ Không thể trích xuất nội dung từ file.", "warning")
                return redirect(request.url)

            # 3. Xóa dữ liệu cũ của file này trong DB
            supabase.table("documents").delete().eq('metadata->>source_file', safe_pdf_name).execute()

            # 4. Tạo embedding cho nội dung của các chunks
            contents_to_embed = [item['content'] for item in chunks]
            embeddings = embed_model.encode(contents_to_embed, batch_size=32, show_progress_bar=True)

            # 5. Chuẩn bị và lưu dữ liệu mới vào DB
            data_to_insert = [{
                "content": item['content'],
                "metadata": item['metadata'],
                "embedding": embeddings[i].tolist(),
            } for i, item in enumerate(chunks)]

            if data_to_insert:
                supabase.table("documents").insert(data_to_insert).execute()
                build_faiss_index() # Xây dựng lại chỉ mục sau khi có dữ liệu mới
                flash(f"✅ Upload và xử lý thành công file: {file.filename}", "success")

        except Exception as e:
            flash(f"❌ Lỗi xử lý khi upload: {e}", "error")
            print(f"!!! Lỗi upload: {e}")
        return redirect(url_for('upload_page'))

    return render_template("upload_site.html", files=list_files())

@app.route("/api/search")
def search():
    """API tìm kiếm đã được đơn giản hóa và hiệu quả hơn."""
    global faiss_index, index_to_id
    q = request.args.get("q", "")
    if not q: return jsonify({"error": "Thiếu từ khóa tìm kiếm"}), 400
    if faiss_index is None: return jsonify({"error": "Chỉ mục tìm kiếm chưa sẵn sàng."}), 503

    try:
        # 1. Tạo embedding cho câu truy vấn
        query_embedding = embed_model.encode(q).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)

        # 2. Tìm kiếm trực tiếp K kết quả gần nhất trong FAISS
        k = 10
        distances, indices = faiss_index.search(query_embedding, k)

        # 3. Lấy ID của các documents tương ứng
        top_ids = [index_to_id[i] for i in indices[0] if i > -1]
        if not top_ids: return jsonify({"query": q, "results": []})

        # 4. Truy vấn CSDL để lấy nội dung đầy đủ của các kết quả
        scores = {index_to_id[i]: float(dist) for i, dist in zip(indices[0], distances[0]) if i > -1}
        response = supabase.table('documents').select('id, content, metadata').in_('id', top_ids).execute()

        # 5. Sắp xếp và định dạng lại kết quả
        sorted_docs = sorted(response.data, key=lambda x: top_ids.index(x['id']))
        results = []
        for doc in sorted_docs:
            metadata = doc.get('metadata', {})
            safe_name = metadata.get('source_file', '')
            results.append({
                "id": doc['id'],
                "title": metadata.get('section_title', 'N/A'),
                "content_body": doc.get('content', '').replace('\n', '<br>'),
                "similarity": scores[doc['id']],
                "source_file_pretty_name": pretty_name(safe_name),
                "source_file_url": supabase.storage.from_(BUCKET_NAME).get_public_url(safe_name) if safe_name else "#"
            })

        return jsonify({"query": q, "results": results})
    except Exception as e:
        print(f"!!! Lỗi API tìm kiếm FAISS: {e}")
        return jsonify({"error": f"Lỗi server: {e}"}), 500

@app.route('/api/delete_file', methods=['POST'])
def delete_file():
    """API xóa file đã được cập nhật để nhất quán với metadata mới."""
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    safe_pdf_name = request.json.get('safe_name')
    if not safe_pdf_name: return jsonify({'error': 'Tên file không hợp lệ'}), 400
    try:
        # Xóa các chunks trong DB dựa trên metadata.source_file
        supabase.table('documents').delete().eq('metadata->>source_file', safe_pdf_name).execute()
        # Xóa file trong Storage
        supabase.storage.from_(BUCKET_NAME).remove([safe_pdf_name])
        # Xây dựng lại chỉ mục
        build_faiss_index()
        flash(f"✅ Đã xóa thành công file. Chỉ mục tìm kiếm đã được cập nhật.", "success")
        return jsonify({'message': 'Xóa file thành công'})
    except Exception as e:
        flash(f"❌ Lỗi khi xóa file: {e}", "error")
        return jsonify({'error': str(e)}), 500

# --- 5. KHỞI CHẠY ỨNG DỤNG ---
if __name__ == "__main__":
    build_faiss_index()  # Xây dựng chỉ mục lần đầu tiên khi app chạy
    app.run(debug=True)
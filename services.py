import os
import fitz  # PyMuPDF
import re
import subprocess
import tempfile
import hashlib
from config import Config, supabase
from models import embed_model, faiss_manager
from utils import clean_text, slugify_filename, create_file_identifier

# Hàm tạo ảnh cho từng trang PDF và upload lên Storage
def generate_and_upload_page_images(pdf_bytes, safe_filename):
    print(f"--- Bắt đầu tạo ảnh cho: {safe_filename} ---")
    page_url_map = {}
    base_name = os.path.splitext(safe_filename)[0]
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes(output="jpg", jpg_quality=80)
            
            current_page = page_num + 1
            image_path = f"page_images/{base_name}/page_{current_page}.jpg"
            
            supabase.storage.from_(Config.BUCKET_NAME).remove([image_path])
            supabase.storage.from_(Config.BUCKET_NAME).upload(
                image_path, img_bytes, {"content-type": "image/jpeg", "cache-control": "3600"}
            )
            page_url_map[current_page] = supabase.storage.from_(Config.BUCKET_NAME).get_public_url(image_path)
        
        return page_url_map
    except Exception as e:
        print(f"Error generating images: {e}")
        return {}

# Hàm chạy Tesseract OCR lên file PDF để lấy lớp text và xóa chữ ký số.
def perform_ocr(pdf_bytes_in, safe_filename):
    temp_in_path = "" # Tạo file tạm để lưu file PDF gốc.
    temp_unsigned_path = "" # Tạo file tạm để lưu file PDF sau khi gỡ chữ ký số.
    temp_out_path = "" # Tạo file tạm để lưu file PDF sau khi OCR.
    ocr_pdf_bytes = pdf_bytes_in # Lưu file PDF gốc vào file tạm.

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t_in: 
            t_in.write(pdf_bytes_in)
            temp_in_path = t_in.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t_unsign: 
            temp_unsigned_path = t_unsign.name
            
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as t_out: 
            temp_out_path = t_out.name

        subprocess.run(['gswin64c', '-o', temp_unsigned_path, '-sDEVICE=pdfwrite', '-dNOSAFER', temp_in_path], capture_output=True, check=False) # Chạy Ghostscript để gỡ chữ ký số.
        
        subprocess.run(['ocrmypdf', '-l', 'vie+eng', '-O0', '--force-ocr', '--continue-on-soft-render-error', temp_unsigned_path, temp_out_path], 
            capture_output=True, check=False) # Chạy Tesseract OCR để OCR lên file PDF sau khi gỡ chữ ký số và lưu vào file tạm.
        
        if os.path.exists(temp_out_path) and os.path.getsize(temp_out_path) > 0: # Kiểm tra file tạm đã tạo thành công chưa.
            with open(temp_out_path, 'rb') as f:
                ocr_pdf_bytes = f.read()
                print("OCR thành công!")
    except Exception as e:
        print(f"OCR Error: {e}")
    finally:
        for p in [temp_in_path, temp_unsigned_path, temp_out_path]: # Xóa file tạm sau khi sử dụng xong.
            if p and os.path.exists(p): 
                try: os.remove(p) 
                except: pass
            
    return ocr_pdf_bytes

# Hàm tính hash của nội dung chunk.
def compute_chunk_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

# Hàm chia nhỏ PDF thành các chunks, kết hợp Heading và Max Length (= 800 ký tự).
def chunk_pdf(pdf_bytes, safe_filename, process_mode='image', page_url_map=None, max_chunk_size=800):
    print(f"--- Chunking ({process_mode}) ---")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf") # Mở file PDF để lấy các trang.
    chunks = []
    current_heading = "Nội dung chung" # Tiêu đề của chunk hiện tại.
    current_buffer = "" # Nội dung của chunk hiện tại.
    
    heading_pattern = re.compile(r"^(PHẦN\s+[\d\.]+|[IVXLCDM]+\s*\.|Chương\s+[IVXLCDM]+|Điều\s+\d+|Mục\s+\d+|[A-Z]\.|\d+\.\d*\.)", re.IGNORECASE) # Phát hiện tiêu đề theo Điều, Mục, Chương, Phần
    
    start_page = 0 # Trang bắt đầu.
    current_chunk_page = 1 # Trang hiện tại.

    for page_num in range(start_page, len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            text = clean_text(block[4]) # fitz trả về dạng list, block[4] là text của block, các vị trí còn lại là tọa độ và các thông số khác.
            if not text: continue

            is_heading = heading_pattern.match(text) and len(text) < 150 # Phát hiện tiêu đề nếu text ngắn hơn 150 ký tự và khớp với pattern.
            
            if is_heading or len(current_buffer) > max_chunk_size: # Nếu là tiêu đề hoặc chunk hiện tại đã lớn hơn max_chunk_size thì tạo chunk mới.
                if current_buffer.strip():
                    metadata = {
                        "source_file": safe_filename,
                        "section_title": current_heading,
                        "page": current_chunk_page
                    }
                    if process_mode == 'image' and page_url_map:
                        metadata["image_url"] = page_url_map.get(current_chunk_page)

                    content_str = f"{current_heading}\n{current_buffer.strip()}" # Tạo nội dung của chunk.
                    chunks.append({
                        "content": content_str,
                        "metadata": metadata,
                        "hash": compute_chunk_hash(content_str)
                    })
                    
                    current_buffer = ""
                    current_chunk_page = page_num + 1

                if is_heading:
                    current_heading = text
                else:
                    current_buffer += text + "\n"
            else:
                current_buffer += text + "\n"

    if current_buffer.strip(): # Nếu còn nội dung trong buffer thì tạo chunk cuối cùng.
        metadata = {
            "source_file": safe_filename, 
            "section_title": current_heading,
            "page": current_chunk_page
        }
        if process_mode == 'image' and page_url_map:
            metadata["image_url"] = page_url_map.get(current_chunk_page)
            
        content_str = f"{current_heading}\n{current_buffer.strip()}"
        chunks.append({
            "content": content_str,
            "metadata": metadata,
            "hash": compute_chunk_hash(content_str)
        })
    
    print(f"--- Created {len(chunks)} chunks ---")
    return chunks

# Hàm xử lý luồng upload.
def process_upload(file):
    pdf_bytes = file.read()
    
    original_filename = file.filename
    # Tạo ID cố định dựa trên tên gốc: "Quy che.pdf" -> "quy-che" -> Dùng để tìm đúng phiên bản trước của tài liệu này.
    file_id = create_file_identifier(original_filename) 
    safe_name = slugify_filename(original_filename) 
    
    # 1. Upload file gốc
    supabase.storage.from_(Config.BUCKET_NAME).upload(safe_name, pdf_bytes, {"content-type": "application/pdf"})
    
    # 2. Xử lý OCR cho file và tạo ảnh cho các trang trong file.
    ocr_bytes = perform_ocr(pdf_bytes, safe_name)
    page_map = generate_and_upload_page_images(ocr_bytes, safe_name)
    
    # 3. Chunking
    new_chunks = chunk_pdf(ocr_bytes, safe_name, process_mode='image', page_url_map=page_map)

    # Gán ID vào metadata chunk
    for chunk in new_chunks:
        chunk['metadata']['file_id'] = file_id
        chunk['metadata']['original_filename'] = original_filename

    table_name = "documents"
    
    # 4. Lấy giá trị hash cũ dựa trên FILE_ID của phiên bản trước của tài liệu để so sánh với hash mới.
    old_docs_response = supabase.table(table_name)\
        .select('id, metadata')\
        .eq('metadata->>file_id', file_id)\
        .execute()
    
    old_hashes_map = {} 
    
    if old_docs_response.data: # Lưu lại map giữa hash cũ và id của chunk cũ.
        for doc in old_docs_response.data:
            meta = doc.get('metadata', {})
            h = meta.get('content_hash')
            if h:
                old_hashes_map[h] = doc['id']
            
    # 5. Phân loại thành 2 loại chunk.
    chunks_to_insert = [] # Chunks mới.
    ids_to_keep = [] # Chunks cũ.
    
    for chunk in new_chunks:
        h = chunk['hash'] # Lấy hash của chunk mới.
        if h in old_hashes_map:
            ids_to_keep.append(old_hashes_map[h])
            # Cập nhật lại metadata (page_url, source_file) cho chunk cũ
            # Supabase vector update hơi có vấn đề nên đôi khi link ảnh của chunk cũ sẽ trỏ về file PDF cũ (vẫn xem được bình thường).
        else:
            chunk['metadata']['content_hash'] = h # Cập nhật hash mới cho chunk cũ.
            chunks_to_insert.append(chunk)
    
    new_hashes_set = set(c['hash'] for c in new_chunks) # Lấy tất cả hash của chunks mới.
    ids_to_delete = [oid for h, oid in old_hashes_map.items() if h not in new_hashes_set] # Lấy tất cả id của chunks cũ không có trong chunks mới.

    print(f"--- Update: Insert {len(chunks_to_insert)}, Keep {len(ids_to_keep)}, Delete {len(ids_to_delete)} ---")

    # 6. Thực hiện các thao tác trên database.
    if ids_to_delete: # Xóa chunks cũ.
        supabase.table(table_name).delete().in_('id', ids_to_delete).execute()
        
    if chunks_to_insert:
        contents = [c['content'] for c in chunks_to_insert]
        print("--- Embedding new chunks ---")
        embeddings = embed_model.encode(contents, batch_size=32, show_progress_bar=True)
        
        data = [{
            "content": c['content'],
            "metadata": c['metadata'],
            "embedding": e.tolist()
        } for c, e in zip(chunks_to_insert, embeddings)]
        
        supabase.table(table_name).insert(data).execute()
        
    if ids_to_delete or chunks_to_insert:
        faiss_manager.refresh_index()
            
    return True, f"Cập nhật {original_filename}: +{len(chunks_to_insert)} mới, -{len(ids_to_delete)} cũ."
    
    return False, "Không trích xuất được nội dung."

# Hàm xóa tài liệu và các dữ liệu liên quan.
def delete_document(safe_name):
    try:
        # Xóa DB dựa trên source_file (safe_name)
        supabase.table('documents').delete().eq('metadata->>source_file', safe_name).execute()
        
        # Xóa file PDF gốc
        supabase.storage.from_(Config.BUCKET_NAME).remove([safe_name])
        
        # Xóa folder ảnh
        base = os.path.splitext(safe_name)[0]
        path = f"page_images/{base}"
        files = supabase.storage.from_(Config.BUCKET_NAME).list(path=path)
        if files:
            targets = [f"{path}/{f['name']}" for f in files]
            supabase.storage.from_(Config.BUCKET_NAME).remove(targets)
            
        # Refresh index
        faiss_manager.refresh_index()
        return True, "Xóa thành công"
    except Exception as e:
        return False, str(e)
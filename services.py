import os
import fitz  # PyMuPDF
import re
import subprocess
import tempfile
import hashlib
from config import Config, supabase
from models import embed_model, faiss_manager
from utils import clean_text, slugify_filename, create_file_identifier

# --- CÁC HÀM XỬ LÝ PDF & OCR ---

def generate_and_upload_page_images(pdf_bytes, safe_filename):
    """Tạo ảnh cho từng trang PDF và upload lên Storage."""
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

def perform_ocr(pdf_bytes_in, safe_filename):
    """Thực hiện OCR: Gỡ chữ ký -> OCR -> Trả về bytes PDF mới."""
    temp_in_path = ""
    temp_unsigned_path = ""
    temp_out_path = ""
    ocr_pdf_bytes = pdf_bytes_in 

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t_in:
            t_in.write(pdf_bytes_in)
            temp_in_path = t_in.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t_unsign:
            temp_unsigned_path = t_unsign.name
            
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as t_out:
            temp_out_path = t_out.name

        print("--- Running Ghostscript ---")
        subprocess.run(['gswin64c', '-o', temp_unsigned_path, '-sDEVICE=pdfwrite', '-dNOSAFER', temp_in_path],
                       capture_output=True, check=False)
        
        print("--- Running OCRMyPDF ---")
        subprocess.run(['ocrmypdf', '-l', 'vie+eng', '-O0', '--force-ocr', 
                       '--continue-on-soft-render-error', temp_unsigned_path, temp_out_path],
                       capture_output=True, check=False)
        
        if os.path.exists(temp_out_path) and os.path.getsize(temp_out_path) > 0:
            with open(temp_out_path, 'rb') as f:
                ocr_pdf_bytes = f.read()
                print("--- OCR Completed ---")
    except Exception as e:
        print(f"OCR Error: {e}")
    finally:
        for p in [temp_in_path, temp_unsigned_path, temp_out_path]:
            if p and os.path.exists(p): 
                try: os.remove(p) 
                except: pass
            
    return ocr_pdf_bytes

def compute_chunk_hash(content):
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def chunk_pdf(pdf_bytes, safe_filename, process_mode='image', page_url_map=None, max_chunk_size=800):
    """Chia nhỏ PDF thành các chunks, kết hợp Heading và Max Length."""
    print(f"--- Chunking ({process_mode}) ---")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []
    current_heading = "Nội dung chung" 
    current_buffer = ""
    
    heading_pattern = re.compile(r"^(PHẦN\s+[\d\.]+|[IVXLCDM]+\s*\.|Chương\s+[IVXLCDM]+|Điều\s+\d+|Mục\s+\d+|[A-Z]\.|\d+\.\d*\.)", re.IGNORECASE)
    
    start_page = 0
    current_chunk_page = 1

    for page_num in range(start_page, len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        for block in blocks:
            text = clean_text(block[4])
            if not text: continue

            is_heading = heading_pattern.match(text) and len(text) < 150
            
            if is_heading or len(current_buffer) > max_chunk_size:
                if current_buffer.strip():
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
                    
                    current_buffer = ""
                    current_chunk_page = page_num + 1

                if is_heading:
                    current_heading = text
                else:
                    current_buffer += text + "\n"
            else:
                current_buffer += text + "\n"

    if current_buffer.strip():
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

def process_upload(file):
    """Xử lý luồng upload chính (Smart Update with File ID)."""
    pdf_bytes = file.read()
    
    original_filename = file.filename
    # Tạo ID cố định dựa trên tên gốc: "Quy che.pdf" -> "quy-che"
    file_id = create_file_identifier(original_filename) 
    # Tạo tên safe mới cho lần upload này: "2024...-quy-che.pdf"
    safe_name = slugify_filename(original_filename) 
    
    # 1. Upload file gốc
    supabase.storage.from_(Config.BUCKET_NAME).upload(safe_name, pdf_bytes, {"content-type": "application/pdf"})
    
    # 2. OCR & Image
    ocr_bytes = perform_ocr(pdf_bytes, safe_name)
    page_map = generate_and_upload_page_images(ocr_bytes, safe_name)
    
    # 3. Chunking
    new_chunks = chunk_pdf(ocr_bytes, safe_name, process_mode='image', page_url_map=page_map)

    # Gán ID vào metadata chunk
    for chunk in new_chunks:
        chunk['metadata']['file_id'] = file_id
        chunk['metadata']['original_filename'] = original_filename

    table_name = "documents"
    
    # 4. Lấy hash cũ dựa trên FILE_ID (để tìm đúng phiên bản trước của tài liệu này)
    # Thay vì tìm theo safe_name (vì safe_name luôn mới), ta tìm theo file_id
    old_docs_response = supabase.table(table_name)\
        .select('id, metadata')\
        .eq('metadata->>file_id', file_id)\
        .execute()
    
    old_hashes_map = {}
    
    if old_docs_response.data:
        for doc in old_docs_response.data:
            meta = doc.get('metadata', {})
            h = meta.get('content_hash')
            if h:
                old_hashes_map[h] = doc['id']
            
    # 5. Phân loại (Diff)
    chunks_to_insert = []
    ids_to_keep = []
    
    for chunk in new_chunks:
        h = chunk['hash']
        if h in old_hashes_map:
            ids_to_keep.append(old_hashes_map[h])
            # Cập nhật lại metadata (ví dụ page url mới, source_file mới) cho chunk cũ
            # Tuy nhiên Supabase vector update hơi phức tạp, nên đơn giản nhất là:
            # Nếu muốn update url ảnh mới -> Phải coi là chunk mới -> Xóa cũ insert mới.
            # Nhưng ở đây ta ưu tiên GIỮ NGUYÊN chunk cũ để tiết kiệm embedding.
            # Hạn chế: Link ảnh của chunk cũ sẽ trỏ về file PDF cũ (vẫn xem được bình thường).
        else:
            chunk['metadata']['content_hash'] = h 
            chunks_to_insert.append(chunk)
    
    new_hashes_set = set(c['hash'] for c in new_chunks)
    ids_to_delete = [oid for h, oid in old_hashes_map.items() if h not in new_hashes_set]

    print(f"--- Smart Update: Insert {len(chunks_to_insert)}, Keep {len(ids_to_keep)}, Delete {len(ids_to_delete)} ---")

    # 6. Thực thi
    if ids_to_delete:
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

def delete_document(safe_name):
    """Xóa tài liệu khỏi hệ thống."""
    try:
        # Xóa DB dựa trên source_file (safe_name)
        # Lưu ý: Nếu dùng file_id thì sẽ xóa hết các phiên bản, 
        # nhưng ở đây UI đang gửi safe_name của từng file cụ thể.
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

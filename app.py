from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from config import Config, supabase
from models import faiss_manager
from utils import get_file_list, pretty_name
from services import process_upload, delete_document
from ai_services import ask_gemini

app = Flask(__name__, template_folder="templates")
app.secret_key = Config.SECRET_KEY

@app.route("/") # Trang chủ
def home():
    return render_template("search_site.html")

@app.route("/login", methods=["GET", "POST"]) # Trang đăng nhập
def login():
    if request.method == "POST":
        try:
            email = request.form.get("email")
            password = request.form.get("password")
            res = supabase.auth.sign_in_with_password({"email": email, "password": password})
            session['user'] = {'id': res.user.id, 'email': res.user.email}
            return redirect(url_for('upload_page'))
        except Exception:
            flash("❌ Đăng nhập thất bại.", "error")
    return render_template("login_site.html")

@app.route("/signup", methods=["GET", "POST"]) # Trang đăng ký
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        if password != request.form.get("confirm"):
            flash("Mật khẩu xác nhận không khớp.", "error")
            return redirect(url_for('signup'))
        try:
            supabase.auth.sign_up({"email": email, "password": password})
            flash("✅ Đăng ký thành công!", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"❌ Lỗi: {e}", 'error')
    return render_template("signup_site.html")

@app.route("/logout") # Đăng xuất
def logout():
    session.pop('user', None)
    flash("Đã đăng xuất.", "info")
    return redirect(url_for('home'))

@app.route("/upload", methods=["GET", "POST"]) # Trang upload
def upload_page():
    if 'user' not in session: return redirect(url_for('login'))
    
    if request.method == "POST":
        file = request.files.get("file") # Lấy file từ form
        
        if not file or not file.filename:
            flash("❌ Vui lòng chọn file.", "error")
        else:
            try:
                success, msg = process_upload(file) # Xử lý upload
                if success: flash(f"✅ {msg}", "success")
                else: flash(f"⚠️ {msg}", "warning")
            except Exception as e:
                flash(f"❌ Lỗi upload: {e}", "error")
                print(f"Upload error: {e}")
        return redirect(request.url)

    return render_template("upload_site.html", files=get_file_list())

@app.route("/api/search") # API tìm kiếm
def search():
    search_type = request.args.get("type", "ai") 
    q = request.args.get("q", "")
    
    if not q: return jsonify({"error": "Thiếu từ khóa"}), 400 
    
    try:
        dists, idxs, id_map = faiss_manager.search(q, top_k=Config.TOP_K_SEARCH) # Tìm kiếm
        
        if not id_map or idxs.size == 0 or idxs[0].size == 0: # Nếu không tìm thấy kết quả
             return jsonify({"query": q, "results": [], "type": search_type})

        valid_matches = []
        for i, dist in zip(idxs, dists):
            if i > -1 and dist >= Config.SIMILARITY_THRESHOLD: # Nếu kết quả tương đồng lớn hơn ngưỡng
                if i in id_map: 
                    valid_matches.append((id_map[i], float(dist))) # Thêm kết quả vào list
                
        if not valid_matches:
            return jsonify({"query": q, "results": [], "type": search_type}) 
            
        top_ids = [mid for mid, _ in valid_matches] # Lấy ID của kết quả
        scores = {mid: s for mid, s in valid_matches} # Lấy điểm tương đồng của kết quả
        
        res = supabase.table('documents').select('id, content, metadata').in_('id', top_ids).execute() # Lấy dữ liệu từ DB
        
        sorted_docs = sorted(res.data, key=lambda x: top_ids.index(x['id'])) # Sắp xếp kết quả theo ID
        results = []
        context_for_ai = [] # List context cho AI

        for doc in sorted_docs: # Duyệt qua từng kết quả
            meta = doc.get('metadata', {})
            image_url = meta.get('image_url') # Lấy URL ảnh
            raw_content = doc.get('content', '') # Lấy nội dung
            
            # Loại bỏ prefix thừa
            clean_content = raw_content.replace('Tiêu đề: ', '', 1).replace('\nNội dung: ', '\n', 1) # Loại bỏ prefix thừa

            if search_type == 'image' and not image_url:
                continue 
                
            safe_name = meta.get('source_file', '')
            item = {
                "id": doc['id'],
                "title": meta.get('section_title', 'N/A'),
                "similarity": scores[doc['id']],
                "source_file_pretty_name": pretty_name(safe_name),
                "source_file_url": supabase.storage.from_(Config.BUCKET_NAME).get_public_url(safe_name) if safe_name else "#"
            }
            
            if search_type == 'image':
                item["page_number"] = meta.get('page', 'N/A')
                item["image_url"] = image_url
                item["content_body"] = "" 
            elif search_type == 'ai':
                # Mode AI: Lấy context, item result để hiển thị reference gọn
                context_for_ai.append(clean_content) # Thêm context vào list
                item["content_body"] = "" # Không cần body dài dòng ở reference
            else:
                # Mode Text
                item["content_body"] = clean_content.replace('\n', '<br>')
                item["image_url"] = image_url 
                
            results.append(item)
            
        # --- AI PROCESSING ---
        ai_answer = None
        # Chỉ gọi AI ở mode AI
        if search_type == 'ai' and results and Config.GEMINI_API_KEY:
            ai_answer = ask_gemini(q, context_for_ai)

        return jsonify({
            "query": q, 
            "results": results, 
            "type": search_type,
            "ai_answer": ai_answer
        })

    except Exception as e:
        print(f"Search error: {e}")
        if "10054" in str(e):
            return jsonify({"query": q, "results": [], "type": search_type})
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_file', methods=['POST']) # API xóa file
def delete_file_api():
    if 'user' not in session: return jsonify({'error': 'Unauthorized'}), 401
    safe_name = request.json.get('safe_name')
    
    success, msg = delete_document(safe_name)
    if success:
        flash(f"✅ {msg}", "success")
        return jsonify({'message': msg})
    else:
        flash(f"❌ {msg}", "error")
        return jsonify({'error': msg}), 500

if __name__ == "__main__":
    faiss_manager.refresh_index()
    app.run(debug=True)
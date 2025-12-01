import faiss
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from config import supabase

# Khởi tạo mô hình Embedding (Singleton)
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")

class FaissManager:
    def __init__(self):
        # Chỉ giữ 1 index chung cho toàn bộ hệ thống
        self.index = None
        self.id_map = {}

    def build_index(self):
        try:
            # Lấy toàn bộ data từ bảng documents
            response = supabase.table('documents').select('id, embedding').execute()
            docs = response.data
            if not docs:
                print("Không có dữ liệu trong database!")
                self.index = None
                self.id_map = {}
                return

            ids = [doc['id'] for doc in docs]
            embeddings_list = [ast.literal_eval(doc['embedding']) for doc in docs] # Dữ liệu từ DB trả về dạng chuỗi (String), cần chuyển thành List số thực.
            embeddings = np.array(embeddings_list).astype('float32') # Chuyển List thành mảng Numpy định dạng float32 (FAISS yêu cầu bắt buộc định dạng này).
            
            self.id_map = {i: doc_id for i, doc_id in enumerate(ids)} # Lưu lại vị trí ánh xạ giữa index và id của tài liệu.
            
            # Khởi tạo index 
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension) # Khởi tạo index với định dạng Inner Product (IP) cho phép tính cosine similarity.
            faiss.normalize_L2(embeddings) # Chuẩn hóa vector về độ dài đơn vị để tính cosine similarity chính xác.
            self.index.add(embeddings) # Đưa toàn bộ vector vào bộ nhớ RAM để sẵn sàng tìm kiếm.
            
            print(f"--- Đã xong chỉ mục chung với {self.index.ntotal} vectors ---")
        except Exception as e:
            print(f"Error building index: {e}")
            self.index = None
            self.id_map = {}

    # Refresh lại chỉ mục
    def refresh_index(self):
        self.build_index()

    # Hàm này được gọi khi người dùng bấm nút "Tìm kiếm"
    def search(self, query: str, top_k: int = 10):
        if self.index is None:
            print("Chỉ mục chưa sẵn sàng, đang thử build lại...")
            self.refresh_index()
            if self.index is None:
                # Chắc chắn không có index, trả về kết quả rỗng
                return np.array([[]], dtype='float32'), np.array([[]], dtype='int64'), {}

        query_embedding = embed_model.encode(query).astype('float32').reshape(1, -1) # Chuyển đổi query thành vector embedding.
        faiss.normalize_L2(query_embedding) # Chuẩn hóa vector về độ dài đơn vị để tính cosine similarity chính xác.
        
        distances, indices = self.index.search(query_embedding, top_k) # FAISS quét trong RAM trả về 2 giá trị: distances (điểm số tương đồng xấp xỉ 1.0) và indices (vị trí) của các vector tìm thấy.
        return distances[0], indices[0], self.id_map # Trả về id_map để app.py dịch ngược lại ra ID thật trong Database.

# Singleton Instance
faiss_manager = FaissManager()
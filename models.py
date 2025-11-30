import faiss
import numpy as np
import ast
from sentence_transformers import SentenceTransformer
from config import supabase

# Khởi tạo mô hình Embedding (Singleton)
print(">>> Đang tải mô hình Sentence Transformer...")
embed_model = SentenceTransformer("intfloat/multilingual-e5-large")
print(">>> Mô hình đã tải xong.")

class FaissManager:
    def __init__(self):
        # Chỉ giữ 1 index chung cho toàn bộ hệ thống
        self.index = None
        self.id_map = {}

    def build_index(self):
        """Xây dựng chỉ mục FAISS từ bảng 'documents' duy nhất."""
        print("--- Đang xây dựng chỉ mục chung (documents) ---")
        try:
            # Lấy toàn bộ data từ bảng documents
            response = supabase.table('documents').select('id, embedding').execute()
            docs = response.data
            if not docs:
                print("Warning: No data in documents table!")
                self.index = None
                self.id_map = {}
                return

            ids = [doc['id'] for doc in docs]
            embeddings_list = [ast.literal_eval(doc['embedding']) for doc in docs]
            embeddings = np.array(embeddings_list).astype('float32')
            
            self.id_map = {i: doc_id for i, doc_id in enumerate(ids)}
            
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            print(f"--- Đã xong chỉ mục chung với {self.index.ntotal} vectors ---")
        except Exception as e:
            print(f"Error building index: {e}")
            self.index = None
            self.id_map = {}

    def refresh_index(self):
        """Làm mới chỉ mục."""
        self.build_index()

    def search(self, query: str, top_k: int = 10):
        """Tìm kiếm vector trên chỉ mục chung."""
        if self.index is None:
            print("Chỉ mục chưa sẵn sàng, đang thử build lại...")
            self.refresh_index()
            if self.index is None:
                # Thay vì báo lỗi, trả về kết quả rỗng nếu không có index
                return np.array([[]], dtype='float32'), np.array([[]], dtype='int64'), {}

        query_embedding = embed_model.encode(query).astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        return distances[0], indices[0], self.id_map

# Singleton Instance
faiss_manager = FaissManager()

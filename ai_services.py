import google.generativeai as genai
from config import Config

# Cấu hình API
if Config.GEMINI_API_KEY:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    print(">>> Đã cấu hình Gemini API.")
else:
    print("!!! Cảnh báo: Chưa có API_KEY trong .env")

def ask_gemini(question, context_docs):
    """
    Gửi câu hỏi và ngữ cảnh cho Gemini để tổng hợp câu trả lời.
    """
    if not Config.GEMINI_API_KEY:
        return None

    try:
        # Tạo model
        model = genai.GenerativeModel('gemini-2.5-flash') 

        # Chuẩn bị ngữ cảnh
        context_text = ""
        # Giới hạn context để tránh quá dài (ví dụ: lấy tối đa 5 đoạn tốt nhất)
        limit_docs = context_docs[:5] 
        for i, doc in enumerate(limit_docs):
            context_text += f"--- Thông tin {i+1} ---\n{doc}\n\n"

        # Tạo Prompt
        prompt = f"""
        Bạn là trợ lý AI của trường Đại học Giao thông Vận tải (UTC).
        Nhiệm vụ: Trả lời câu hỏi của sinh viên dựa CHÍNH XÁC vào thông tin được cung cấp.
        
        Yêu cầu:
        - Trả lời bằng tiếng Việt tự nhiên, lịch sự.
        - Tóm tắt các ý chính liên quan.
        - Nếu thông tin không có trong ngữ cảnh, hãy nói "Xin lỗi, tôi chưa tìm thấy thông tin chính xác trong tài liệu hiện có."
        - Trình bày đẹp bằng Markdown.

        Câu hỏi: {question}

        Thông tin tham khảo:
        {context_text}
        """

        # Gọi API
        response = model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"Lỗi gọi Gemini: {e}")
        return "Xin lỗi, AI đang gặp sự cố kết nối."

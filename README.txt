
# Facebook Comment Sentiment Analyzer

## Cách sử dụng

1. Cài thư viện:
   pip install flask pandas joblib scikit-learn requests

2. Huấn luyện mô hình:
   python model/train_model.py

3. Chạy web:
   python app.py

4. Truy cập: http://127.0.0.1:5000

## Lưu ý:
- Tạo Facebook App trên developers.facebook.com
- Lấy Access Token (phải có quyền đọc bình luận Page)
- Lấy Post ID từ link bài viết

# Dịch máy Nơ-ron Anh-Pháp (English-French Neural Machine Translation)

Dự án này triển khai một hệ thống Dịch máy Nơ-ron (NMT) để dịch văn bản tiếng Anh sang tiếng Pháp sử dụng PyTorch. Dự án khám phá và so sánh hai kiến trúc Sequence-to-Sequence (Seq2Seq): mô hình LSTM hai chiều tiêu chuẩn và mô hình nâng cao tích hợp Cơ chế Chú ý (Attention Mechanism).

## Tổng quan dự án

Mục tiêu của dự án là xây dựng một mô hình học sâu có khả năng dịch các câu tiếng Anh sang tiếng Pháp. Dự án bao gồm toàn bộ quy trình từ tiền xử lý dữ liệu, xây dựng từ điển đến huấn luyện và đánh giá mô hình.

### Các tính năng chính
*   **Tiền xử lý dữ liệu:**
    *   Tách từ (Tokenization) sử dụng `spacy` (tiếng Anh & tiếng Pháp).
    *   Làm sạch văn bản (loại bỏ HTML, chuẩn hóa).
    *   Tải dữ liệu hiệu quả với `torch.utils.data.Dataset` và `DataLoader`.
    *   Xử lý các chuỗi có độ dài thay đổi bằng padding và packing.
*   **Kiến trúc mô hình:**
    1.  **Seq2Seq (Cơ bản):**
        *   **Encoder:** LSTM hai chiều (Bidirectional LSTM) để nắm bắt ngữ cảnh từ cả hai hướng.
        *   **Decoder:** LSTM một chiều (Unidirectional LSTM).
    2.  **Seq2Seq với Attention:**
        *   Tích hợp Cơ chế Chú ý để cho phép decoder tập trung vào các phần cụ thể của câu nguồn tại mỗi bước, cải thiện chất lượng dịch cho các câu dài.
*   **Huấn luyện:**
    *   Chiến lược Teacher Forcing.
    *   Dừng sớm (Early Stopping) để ngăn chặn overfitting.
    *   Tối ưu hóa hàm mất mát CrossEntropyLoss.
*   **Đánh giá:**
    *   **Điểm BLEU:** Thước đo tiêu chuẩn cho chất lượng dịch máy.
    *   **Độ chính xác (Accuracy):** Độ chính xác ở cấp độ token.
    *   **Perplexity (PPL):** Thước đo mức độ mô hình dự đoán mẫu tốt như thế nào.

## Cấu trúc dự án

```
NLP_Project/
├── dataset/
│   └── raw/                # Chứa các file dữ liệu nén (.gz)
│       ├── train.en.gz
│       ├── train.fr.gz
│       ├── val.en.gz
│       ├── val.fr.gz
│       ├── test.en.gz
│       └── test.fr.gz
├── scripts/
│   ├── en_fr.ipynb         # Notebook chính để huấn luyện và suy luận
│   └── models/             # Thư mục lưu các mô hình đã huấn luyện
│       ├── best_model.pth       # Mô hình cơ bản tốt nhất
│       ├── best_model_attn.pth  # Mô hình attention tốt nhất
│       └── ...
├── LICENSE
└── README.md
```

## Yêu cầu hệ thống

Để chạy dự án này, bạn cần các thư viện sau:

*   Python 3.x
*   PyTorch
*   Spacy
*   NLTK
*   Tqdm

Có thể cài đặt các thư viện phụ thuộc bằng pip:

```bash
pip install torch spacy nltk tqdm
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## Hướng dẫn

Toàn bộ quy trình làm việc nằm trong notebook `scripts/en_fr.ipynb`.

1.  **Chuẩn bị dữ liệu:**
    Script tự động kiểm tra dữ liệu trong `dataset/raw/`. Đảm bảo các file `.gz` của bạn được đặt đúng vị trí.

2.  **Huấn luyện:**
    Chạy các cell trong notebook để bắt đầu huấn luyện. Notebook cho phép bạn huấn luyện cả mô hình Seq2Seq cơ bản và mô hình Attention.
    *   Các siêu tham số như `BATCH_SIZE`, `LEARNING_RATE`, `N_EPOCHS`, `HID_DIM` có thể cấu hình ở đầu các phần huấn luyện.

3.  **Suy luận / Dịch:**
    Sử dụng hàm `translate` để dịch các câu tùy chỉnh:
    ```python
    sentence = "Hello, how are you?"
    translation = translate(sentence)
    print(f"English: {sentence}")
    print(f"French: {translation}")
    ```

4.  **Đánh giá:**
    Notebook bao gồm các phần để đánh giá mô hình trên tập Test, tính toán Loss, Accuracy và điểm BLEU.

## Chi tiết mô hình

### Seq2Seq Cơ bản
*   **Encoder:** LSTM hai chiều 2 lớp.
*   **Decoder:** LSTM một chiều 2 lớp.
*   **Kích thước Embedding:** 256.
*   **Kích thước Ẩn (Hidden Dimension):** 512.
*   **Dropout:** 0.5.

### Mô hình Attention
*   **Lớp Attention:** Tính toán điểm năng lượng giữa trạng thái ẩn của decoder và đầu ra của encoder để tạo vector ngữ cảnh.
*   **Decoder:** Nối vector ngữ cảnh với đầu vào đã được nhúng (embedded input) để dự đoán token tiếp theo.

## License

[MIT License](LICENSE)

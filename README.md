# Dịch máy Nơ-ron Anh-Pháp (English-French Neural Machine Translation)

Dự án này xây dựng và so sánh hai hệ thống Dịch máy Nơ-ron (Neural Machine Translation - NMT) để dịch văn bản từ tiếng Anh sang tiếng Pháp. Dự án được triển khai bằng **PyTorch** và thực hiện so sánh giữa mô hình **Seq2Seq LSTM** cơ bản và mô hình **Seq2Seq tích hợp cơ chế Attention**.

## Cấu trúc dự án

```text
NLP_Project/
├── dataset/
│   └── raw/                # Thư mục chứa dữ liệu thô (nén .gz)
│       ├── train.en.gz     # Tập huấn luyện (Anh)
│       ├── train.fr.gz     # Tập huấn luyện (Pháp)
│       ├── val.en.gz       # Tập kiểm thử (Anh)
│       ├── val.fr.gz       # Tập kiểm thử (Pháp)
│       ├── test.en.gz      # Tập đánh giá (Anh)
│       └── test.fr.gz      # Tập đánh giá (Pháp)
├── scripts/
│   ├── en_fr.ipynb         # Notebook chính (Tiền xử lý, Train, Eval)
│   └── models/             # Thư mục lưu trữ model và lịch sử huấn luyện
│       ├── best_model.pth       # Checkpoint model cơ bản tốt nhất
│       ├── best_model_attn.pth  # Checkpoint model attention tốt nhất
│       ├── training_history.json
│       ├── history_attn.json
│       └── comparison_chart.png # Biểu đồ so sánh kết quả
├── LICENSE
└── README.md
```

## Yêu cầu hệ thống

### Thư viện Python
Dự án yêu cầu Python 3.x và các thư viện sau:
*   `torch`: Framework Deep Learning.
*   `spacy`: Xử lý ngôn ngữ tự nhiên (Tokenization).
*   `nltk`: Tính điểm BLEU.
*   `pandas`, `matplotlib`: Xử lý dữ liệu và vẽ biểu đồ.
*   `tqdm`: Hiển thị thanh tiến trình.

### Cài đặt
Chạy lệnh sau để cài đặt các gói cần thiết:

```bash
pip install torch spacy nltk pandas matplotlib tqdm
```

Tải các mô hình ngôn ngữ cho Spacy:

```bash
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
```

## Phương pháp & Kiến trúc Mô hình

### 1. Tiền xử lý dữ liệu
*   **Làm sạch:** Loại bỏ thẻ HTML, chuẩn hóa khoảng trắng, chuyển về chữ thường.
*   **Tokenization:** Sử dụng Spacy (`en_core_web_sm` cho tiếng Anh, `fr_core_news_sm` cho tiếng Pháp).
*   **Từ điển (Vocabulary):** Giới hạn kích thước (ví dụ: 10,000 từ phổ biến nhất), thêm các token đặc biệt: `<unk>`, `<pad>`, `<sos>`, `<eos>`.

### 2. Mô hình Seq2Seq (Baseline)
*   **Encoder:** LSTM 2 lớp, 2 chiều (Bidirectional).
*   **Decoder:** LSTM 2 lớp, 1 chiều (Unidirectional).
*   **Cơ chế:** Vector ngữ cảnh (context vector) được tạo từ trạng thái ẩn cuối cùng của Encoder và truyền vào Decoder.

### 3. Mô hình Seq2Seq với Attention
*   **Encoder:** Tương tự Baseline.
*   **Attention Layer:** Tính toán trọng số chú ý (attention weights) dựa trên trạng thái ẩn của Decoder và toàn bộ đầu ra của Encoder.
*   **Decoder:** Sử dụng vector ngữ cảnh có trọng số (weighted context vector) để tập trung vào các phần quan trọng của câu nguồn tại mỗi bước dịch.

## Huấn luyện & Đánh giá

Quy trình huấn luyện được thực hiện trong file `scripts/en_fr.ipynb`:

1.  **Cấu hình:**
    *   Optimizer: Adam (Learning rate: 0.001).
    *   Loss Function: CrossEntropyLoss (bỏ qua padding).
    *   Batch size: 128.
    *   Epochs: 15 (có Early Stopping với patience = 3).

2.  **Đánh giá:**
    *   **Loss & Accuracy:** Theo dõi trên tập Validation qua từng epoch.
    *   **BLEU Score:** Đánh giá chất lượng dịch trên tập Test.
    *   **Biểu đồ:** So sánh trực quan quá trình hội tụ của hai mô hình.

## Hướng dẫn sử dụng

1.  Đảm bảo dữ liệu đã được tải vào thư mục `dataset/raw/` đúng theo cấu trúc.
2.  Mở file `scripts/en_fr.ipynb` bằng Jupyter Notebook hoặc VS Code.
3.  Chạy tuần tự các cell để:
    *   Xử lý dữ liệu.
    *   Huấn luyện mô hình Baseline.
    *   Huấn luyện mô hình Attention.
    *   So sánh kết quả và dịch thử câu văn bất kỳ.

## Kết quả mong đợi

Sau khi chạy notebook, bạn sẽ nhận được:
*   Hai file model (`.pth`) lưu trạng thái tốt nhất.
*   File JSON lưu lịch sử huấn luyện.
*   Biểu đồ so sánh Loss/Accuracy giữa hai mô hình.
*   Điểm BLEU trên tập Test.

---
**Lưu ý:** Thời gian huấn luyện có thể lâu tùy thuộc vào phần cứng (khuyến nghị sử dụng GPU).

##  Thông tin chung
* **Giảng viên   hướng dẫn:** PGS.TS.Nguyễn Tuấn Đăng
* **Lớp:** DCT122C3

###  Nhóm sinh viên thực hiện:
1. **Họ và tên:** Huỳnh Minh Quân - **MSSV:** 3122411167
2. **Họ và tên:** Hồ Thái Vũ - **MSSV:** 3122411251

## License

[MIT License](LICENSE)

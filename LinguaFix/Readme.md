# LinguaFix AI – Nền tảng kiểm tra Ngữ pháp & Chính tả Tiếng Việt

**LinguaFix AI** là hệ thống **AI hỗ trợ viết tiếng Việt** giúp phát hiện và sửa:

* lỗi **chính tả**
* lỗi **ngữ pháp**
* lỗi **dấu câu**
* lỗi **phong cách viết**

Hệ thống được thiết kế theo mô hình **API-first SaaS**, có thể tích hợp vào:

* ứng dụng web
* hệ thống CMS
* phần mềm soạn thảo
* chatbot
* call center
* hệ thống xử lý tài liệu

Mục tiêu của LinguaFix AI là xây dựng **"Grammarly cho tiếng Việt"**.

---

# Mục lục

1. Tổng quan
2. Tính năng chính
3. Đối tượng sử dụng
4. Các loại lỗi hệ thống phát hiện
5. Các loại input hỗ trợ
6. Output của hệ thống
7. Kiến trúc hệ thống
8. Pipeline Machine Learning
9. Kiến trúc repository
10. Cài đặt
11. Chạy hệ thống
12. API cơ bản
13. Dữ liệu và training
14. Đánh giá mô hình
15. Vận hành production
16. Bảo mật
17. Roadmap phát triển
18. Đóng góp

---

# 1. Tổng quan

Việc viết tiếng Việt chính xác gặp nhiều khó khăn do:

* dấu thanh phức tạp
* gõ telex sai
* thiếu dấu
* nhầm từ đồng âm
* lỗi ngữ pháp
* lỗi do OCR / ASR

LinguaFix AI giải quyết các vấn đề này bằng **pipeline AI kết hợp nhiều mô hình NLP** để:

1. phát hiện vị trí lỗi
2. phân loại lỗi
3. gợi ý sửa
4. sửa toàn bộ câu

---

# 2. Tính năng chính

## Kiểm tra chính tả

Phát hiện:

* sai dấu
* thiếu dấu
* gõ telex sai
* thiếu chữ
* nhầm từ

Ví dụ:

```
ko biet lam sao
→ không biết làm sao
```

---

## Kiểm tra ngữ pháp

Phát hiện:

* sai trật tự từ
* sai chủ vị
* cấu trúc câu không hợp lý

---

## Kiểm tra dấu câu

* thiếu dấu chấm
* thiếu dấu phẩy
* viết hoa sai

---

## Gợi ý cải thiện văn phong

* lặp từ
* câu khó đọc
* từ ngữ chưa phù hợp

---

## Hỗ trợ nhiều loại input

Hệ thống có thể xử lý:

* text
* tài liệu
* ảnh
* audio

---

# 3. Đối tượng sử dụng

LinguaFix AI hướng tới:

### Sinh viên / học sinh

* kiểm tra bài viết
* viết luận

### Báo chí / xuất bản

* kiểm tra lỗi trước khi xuất bản

### Doanh nghiệp

* viết email
* viết tài liệu

### Call center

* sửa transcript

### Developer

* tích hợp API vào hệ thống

---

# 4. Các loại lỗi hệ thống phát hiện

### Chính tả

* sai dấu
* thiếu dấu
* telex sai
* nhầm từ đồng âm

### Ngữ pháp

* trật tự từ
* sai cấu trúc câu
* sai từ loại

### Dấu câu

* thiếu dấu câu
* dấu câu sai
* viết hoa sai

### Văn phong

* lặp từ
* từ không phù hợp
* câu khó hiểu

---

# 5. Các loại Input hỗ trợ

| Input      | Ví dụ                     |
| ---------- | ------------------------- |
| Text       | người dùng nhập trực tiếp |
| PDF / DOCX | tài liệu                  |
| Image      | scan tài liệu             |
| Audio      | mp3 / wav                 |

---

# 6. Output của hệ thống

Ví dụ kết quả:

```json
{
  "summary": {
    "total": 7,
    "spelling": 3,
    "grammar": 2,
    "punctuation": 1,
    "style": 1
  },
  "errors": [
    {
      "id": "e1",
      "span": [15, 20],
      "original": "ko biet",
      "suggestions": ["không biết"],
      "type": "spelling",
      "confidence": 0.96
    }
  ],
  "corrected_text": "...",
  "accept_all_text": "...",
  "quality_score": 0.92
}
```

---

# 7. Kiến trúc hệ thống

Kiến trúc tổng thể:

```
Client
   |
API Gateway
   |
Grammar API
   |
Queue
   |
Worker Pipeline
   |
ML Inference
   |
Model Server
```

Các service chính:

* Grammar API
* OCR Service
* ASR Service
* Worker
* Model Serving

---

# 8. Pipeline Machine Learning

Pipeline xử lý:

```
Input text
   ↓
Tokenization
   ↓
Phát hiện lỗi
   ↓
Phân loại lỗi
   ↓
Sinh gợi ý sửa
   ↓
Post-processing
```

---

# 9. Model Stack

| Nhiệm vụ           | Model        |
| ------------------ | ------------ |
| Phát hiện lỗi      | PhoBERT      |
| Phân loại lỗi      | XLM-R        |
| Sửa câu            | mT5          |
| Gợi ý văn phong    | LLaMA / Qwen |
| OCR                | PaddleOCR    |
| Speech recognition | Whisper      |

---

# 10. Kiến trúc repository

```
project/

checkpoints/
configs/
data/

src/
  api/
  ml/
      data/
      features/
      inference/
      models/
      preprocessing/
      training/

  services/
  ui/
  utils/

scripts/

tests/

main.py
pyproject.toml
requirements.txt
README.md
```

---

# 11. Cài đặt

Clone repo:

```
git clone https://github.com/your-org/linguafix-ai
cd linguafix-ai
```

Tạo môi trường:

```
python -m venv .venv
source .venv/bin/activate
```

Cài dependencies:

```
pip install -r requirements.txt
```

---

# 12. Chạy hệ thống

Chạy API:

```
python main.py
```

Train model:

```
python scripts/train_model.py
```

Test model:

```
python scripts/test_model.py
```

---

# 13. API cơ bản

### Kiểm tra lỗi

```
POST /api/v1/check
```

Request:

```
{
  "text": "toi ko biet lam sao"
}
```

Response:

```
{
  "errors": [...],
  "corrected_text": "Tôi không biết làm sao."
}
```

---

# 14. Dữ liệu & Training

Nguồn dữ liệu:

### Public dataset

* VLSP
* UIT-ViCTSD
* UIT-ViSpell
* Vietnamese Wikipedia
* CC100 Vietnamese

### Synthetic data

Sinh lỗi nhân tạo:

* bỏ dấu
* gõ telex sai
* noise OCR
* noise ASR

### Human feedback

Hệ thống ghi nhận phản hồi người dùng để cải thiện model.

---

# 15. Đánh giá mô hình

Các metric sử dụng:

* Precision / Recall
* GEC score
* BLEU
* Fluency score
* Human accept rate

---

# 16. Vận hành Production

Production system bao gồm:

* logging
* tracing
* monitoring GPU
* drift detection
* canary deployment
* rollback model

---

# 17. Bảo mật

Các cơ chế bảo mật:

* JWT authentication
* RBAC
* mã hóa dữ liệu
* audit log
* scan file upload

---

# 18. Roadmap

### v1

* spelling
* grammar
* punctuation
* capitalization

### v2

* style
* tone
* consistency

### v3

* semantic correction
* readability scoring
* multi-language
* real-time correction

---

# 19. Đóng góp

Chúng tôi hoan nghênh đóng góp từ cộng đồng.

Hãy tạo issue hoặc pull request để tham gia phát triển dự án.

---

# License

MIT License

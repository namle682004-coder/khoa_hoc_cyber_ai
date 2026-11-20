
# MVP & Product Roadmap – LinguaFix AI

## 1. MVP – Grammar & Spelling Correction (Core Engine)

### Mục tiêu

Xây dựng **AI sửa lỗi chính tả và ngữ pháp tiếng Việt cơ bản**.

### Input

```
Text
```

Ví dụ

```
toi ko biet lam sao de viet dung chinh ta
```

### Output

```
Tôi không biết làm sao để viết đúng chính tả.
```

### Tính năng

* sửa lỗi **chính tả**
* sửa lỗi **ngữ pháp cơ bản**
* chuẩn hóa **dấu câu**
* chuẩn hóa **viết hoa**

### Tech đơn giản nhất

```
FastAPI
Transformers
Vietnamese LLM
```

### UI

1 textbox

```
Input text
↓
Corrected text
```

### Mục tiêu

* demo được AI
* có thể deploy
* có thể test người dùng

---

# Version 1 – Smart Correction

Sau khi MVP chạy ổn định.

### Tính năng

```
highlight lỗi
multiple suggestions
confidence score
```

Ví dụ

Input

```
toi khong biet lam sao
```

Output

```
Tôi không biết làm sao
```

UI highlight

```
toi → Tôi
khong → không
```

### Thêm tính năng

* highlight lỗi
* click để xem suggestion

### Tech thêm

```
token-level prediction
alignment model
```

---

# Version 2 – Writing Assistant

Bắt đầu giống **Grammarly cho tiếng Việt**.

### Tính năng

```
rewrite sentence
improve style
formal / informal tone
shorten sentence
```

Ví dụ

Input

```
tôi không biết phải làm gì tiếp theo
```

Output

```
Tôi chưa biết nên làm gì tiếp theo.
```

---

# Version 3 – Document Support

Cho phép sửa lỗi **file lớn**.

### Input

```
txt
docx
pdf
```

### Tính năng

```
upload document
AI scan toàn bộ văn bản
suggest corrections
```

Ứng dụng:

```
essay
report
email
document
```

---

# Version 4 – Multimodal AI

Mở rộng input.

### Input

```
text
image
audio
```

### Tính năng

**OCR**

```
image → text → grammar correction
```

**Speech**

```
audio → text → grammar correction
```

---

# Version 5 – SaaS Platform

Biến thành **AI product thật sự**.

### Tính năng

```
user account
usage tracking
subscription
API access
```

### API

```
POST /grammar-check
```

Cho phép dev tích hợp vào:

```
editor
CMS
website
apps
```

---

# Roadmap tổng thể

| Version | Feature                 | Độ khó |
| ------- | ----------------------- | ------ |
| MVP     | text grammar correction | ⭐      |
| V1      | highlight + suggestion  | ⭐⭐     |
| V2      | rewrite / style         | ⭐⭐⭐    |
| V3      | document upload         | ⭐⭐⭐    |
| V4      | image + audio           | ⭐⭐⭐⭐   |
| V5      | SaaS + API              | ⭐⭐⭐⭐   |

---

# Timeline realistic

Nếu làm 1 mình:

```
MVP     : 2–3 tuần
V1      : +2 tuần
V2      : +1 tháng
V3      : +1 tháng
V4      : +1–2 tháng
```

---

💡 Lời khuyên thật cho dự án này:

**90% dự án AI chết vì làm quá lớn ngay từ đầu.**

Mày chỉ nên tập trung:

```
MVP = text grammar correction
```



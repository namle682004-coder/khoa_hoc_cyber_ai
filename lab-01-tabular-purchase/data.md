Trong dataset **Online Shoppers Purchasing Intention**, việc **mua hay không mua** thường bị ảnh hưởng mạnh bởi một số feature. Dựa trên kinh nghiệm train model và các paper về dataset này, các giá trị **ảnh hưởng lớn đến khả năng mua** thường có pattern như sau:

---

# 📊 Các Feature ảnh hưởng mạnh đến việc mua

## 1️⃣ PageValues (Quan trọng nhất)

**Ý nghĩa:** giá trị trang (thường là trang sản phẩm / checkout).

| PageValues | Khả năng mua      |
| ---------- | ----------------- |
| 0          | gần như không mua |
| 0 – 5      | thấp              |
| 5 – 20     | trung bình        |
| **> 20**   | **rất dễ mua**    |

👉 Nếu:

```
PageValues > 30
```

thì model thường dự đoán **probability > 0.8**

---

# 2️⃣ ProductRelated

**Số trang sản phẩm khách xem**

| ProductRelated | Khả năng mua |
| -------------- | ------------ |
| < 5            | thấp         |
| 5 – 20         | trung bình   |
| **> 20**       | **cao**      |

Ví dụ:

```
ProductRelated = 40
```

→ người dùng đang xem nhiều sản phẩm → **ý định mua cao**

---

# 3️⃣ ProductRelated_Duration

**Thời gian xem sản phẩm**

| Duration   | Ý nghĩa          |
| ---------- | ---------------- |
| < 100s     | lướt qua         |
| 100 – 500s | quan tâm         |
| **> 500s** | **rất quan tâm** |

Ví dụ:

```
ProductRelated_Duration = 1200
```

→ khả năng mua tăng mạnh

---

# 4️⃣ BounceRates

**Tỷ lệ rời trang ngay**

| BounceRates | Ý nghĩa    |
| ----------- | ---------- |
| > 0.1       | xấu        |
| 0.05 – 0.1  | trung bình |
| **< 0.02**  | **tốt**    |

Ví dụ tốt:

```
BounceRates = 0.005
```

---

# 5️⃣ ExitRates

**Tỷ lệ thoát trang**

| ExitRates  | Ý nghĩa    |
| ---------- | ---------- |
| > 0.1      | thấp       |
| 0.05       | trung bình |
| **< 0.02** | **cao**    |

---

# 6️⃣ VisitorType

| Type                  | Khả năng mua |
| --------------------- | ------------ |
| New_Visitor           | thấp         |
| Other                 | trung bình   |
| **Returning_Visitor** | **cao**      |

---

# 7️⃣ Month

Một số tháng mua nhiều hơn:

| Month   | Ý nghĩa                    |
| ------- | -------------------------- |
| Feb     | thấp                       |
| Mar     | thấp                       |
| May     | trung bình                 |
| **Nov** | **rất cao (Black Friday)** |
| **Dec** | **cao (Christmas)**        |

---

# 8️⃣ SpecialDay

Gần ngày lễ mua nhiều hơn.

| SpecialDay | Ý nghĩa        |
| ---------- | -------------- |
| 0          | bình thường    |
| 0.5        | gần lễ         |
| **1**      | **rất gần lễ** |

---

# 📈 Ví dụ dữ liệu dễ dự đoán mua

```json
{
"Administrative": 3,
"Administrative_Duration": 100,
"Informational": 2,
"Informational_Duration": 60,

"ProductRelated": 45,
"ProductRelated_Duration": 1400,

"BounceRates": 0.005,
"ExitRates": 0.01,

"PageValues": 35,
"SpecialDay": 0.8,

"Month": "Nov",
"OperatingSystems": 2,
"Browser": 2,
"Region": 3,
"TrafficType": 2,

"VisitorType": "Returning_Visitor",
"Weekend": true
}
```

Model thường trả:

```
Probability ≈ 0.85 – 0.95
```

---

# 📉 Ví dụ dữ liệu không mua

```json
{
"ProductRelated": 3,
"ProductRelated_Duration": 40,
"BounceRates": 0.2,
"ExitRates": 0.18,
"PageValues": 0,
"VisitorType": "New_Visitor",
"Month": "Feb"
}
```

Probability thường:

```
0.05 – 0.2
```

---

# 🧠 Feature Importance (thường thấy khi train)

Thường top feature là:

1️⃣ **PageValues**
2️⃣ **ProductRelated_Duration**
3️⃣ **ProductRelated**
4️⃣ **BounceRates**
5️⃣ **ExitRates**
6️⃣ **VisitorType**

---

# 🚀 Nếu muốn project AI của mày xịn hơn

Trong **Gradio UI** có thể thêm:

```
Tips to increase purchase probability:
- Increase product engagement
- Lower bounce rate
- Improve page value
```

Hoặc:

```
Top factors influencing prediction
```

→ dùng **SHAP**

---

Nếu mày muốn tao chỉ thêm **1 đoạn code 20 dòng** để:

```
tự động hiển thị top feature ảnh hưởng đến prediction
```

trong **Gradio UI**

→ nhìn **giống AI product thật luôn**.

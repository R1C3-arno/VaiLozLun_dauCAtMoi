# CHECKLIST HỎI GIẢNG VIÊN – KIỂM TRA THUẬT TOÁN & CODE

## 📋 MỤC ĐÍCH
Document này liệt kê các câu hỏi cần xác nhận với giảng viên để đảm bảo implementation đúng 100% theo paper "Impact of safety factors and setup time reduction in a two-echelon supply chain management" (Sarkar et al., 2018).

---

## I. ✅ CÔNG THỨC TOÁN (CỰC KỲ QUAN TRỌNG)

### 1. Công thức tính tập t_{s,j}

**Câu hỏi 1.1:**
```
Công thức tính ts,j trong paper có đúng là:
    ts,j = Σ(i=j+1 to m) ai - Σ(i=1 to j) bi

```

**Câu hỏi 1.2:**
```
Miền hợp lệ của ts có đúng là:
    Σbi ≤ ts ≤ Σai
Nếu ts,j nằm ngoài miền này thì xử lý như thế nào:
    a) Bỏ qua ts,j đó?
    b) Clamp về biên gần nhất?
    c) Vẫn giữ để so sánh?
```

**Câu hỏi 1.3:**
```
Có bắt buộc phải sort (a, b, C) theo C tăng dần 
TRƯỚC KHI tính ts,j không ?
Hay chỉ sort khi tính CR?
```

---

### 2. Công thức Crashing Cost CR(ts)

**Câu hỏi 2.1:**
```
Công thức CR(ts,j) trong paper page 3:
    CR(ts) = Cj(ts,j-1 - ts,j) + Σ(i=1 to j+1) Ci(ai - bi)

Câu hỏi:
- Tổng có đúng chạy đến j+1 không?
- Khi j = m (component cuối) thì xử lý thế nào?
- ts,j-1 khi j=1 thì lấy ts,max hay 0?
```

**Câu hỏi 2.2:**
```
implement như sau:
    term1 = C[j-1] * max(ts_prev - ts_curr, 0)
    term2 = Σ(i=1 to min(j+1, m)) Ci(ai - bi)

Có đúng không 
```

---

### 3. Công thức TCv (Vendor Cost)

**Câu hỏi 3.1:**
```
Equation (3) trong paper:
    TCv = (AvD)/(nQ) + (hvQ/2)[n(1-D/P) - 1 + 2D/P] 
          + D·CP(P) + B1·ln(θ/θ0) + B2·ln(Av0/Av) 
          + ρDθnQ/2

Xác nhận:
- T4 = B1·ln(θ/θ0) có dấu ÂM không? (vì θ < θ0)
- T5 = B2·ln(Av0/Av) có dấu ÂM không? (vì Av < Av0)
- T6 = ρDθnQ/2 có thiếu hệ số nào không?
```

**Câu hỏi 3.2:**
```
CP(P) = ξ1·P + ξ2/P có đúng không ?
```

---

### 4. Công thức TCb (Buyer Cost)


**Câu hỏi 4.1:**
```
Crashing cost trong TCb:
    (nD·CR(ts))/(nQ) = D·CR(ts)/Q

Có đơn giản thành D·CR/Q được không?
```

---

### 5. Công thức Q* - Equation (10)

**Câu hỏi 5.1:**
```
Đạo hàm ∂TC/∂Q = 0 cho:
    -β1/Q² + β2/Q + β3 = 0

Điều này tương đương phương trình bậc 2:
    β3·Q² + β2·Q - β1 = 0
    ⇒ Q = (-β2 + √(β2² + 4β1β3))/(2β3)

HAY chỉ cần dùng fixed-point iteration:
    Q = β1/(β2 + β3·Q)

Cách nào đúng thầy/cô?
```




---

### 6. Công thức P* - Equation (11)

**Câu hỏi 6.1:**
```
∂TC/∂P = 0 cho:
    P* = √(β4/(D·ξ1))

Câu hỏi:
- Nếu β4 < 0 thì xử lý thế nào?
- Nếu P* < Pmin thì set P = Pmin đúng không?
- Có phải P > Pmin (strict inequality) không?
```




---

### 7. Công thức k1* - Equation (12)


**Câu hỏi 7.1:**
```
k1 có cần vòng lặp hội tụ không?
Hay có công thức trực tiếp?
```

---


---

## II. LOGIC GIẢI THUẬT (FLOWCHART)

### 9. Thứ tự update biến

**Câu hỏi 9.1:**
```
Step 2b-2d trong Algorithm:
    P → Q → k1 → Av → θ

Có đúng thứ tự này không thầy/cô?
Hay phải:
    Q → P → k1 → Av → θ?
```

---

### 10. Điều kiện hội tụ Step 2e

**Câu hỏi 10.1:**
```
Convergence criteria:
    |Q^(k+1) - Q^k| < ε
    |P^(k+1) - P^k| < ε
    |k1^(k+1) - k1^k| < ε
    |Av^(k+1) - Av^k| < ε
    |θ^(k+1) - θ^k| < ε

Có cần check TẤT CẢ 5 biến không?
Hay chỉ cần Q, P là đủ?
```

**Câu hỏi 10.2:**
```
ε = 1e-6 có hợp lý không thầy/cô?
```

---

### 11. Step 3 – Xử lý ràng buộc (QUAN TRỌNG NHẤT)

**Câu hỏi 11.1:**
```
Step 3b: If Av* > Av0 and θ* < θ0
         → Set Av = Av0 and GO BACK TO STEP 2

XÁC NHẬN:
Khi "GO BACK TO STEP 2", có nghĩa là:
    a) RESET TẤT CẢ biến (Q, P, k1, Av, θ) về giá trị ban đầu?
    HAY
    b) CHỈ FIX Av = Av0, GIỮ NGUYÊN Q, P, k1, θ đã tính?

QUAN TRỌNG
```

**Câu hỏi 11.2:**
```
Nếu constraint bị vi phạm:
    - Số lần lặp lại tối đa Step 2 là bao nhiêu?
    - Nếu sau 50 lần vẫn không thỏa mãn thì:
        a) Bỏ j này, chuyển sang j tiếp theo?
        b) Báo lỗi và dừng?
```

---

### 12. Step 5 – Điều kiện dừng theo n

**Câu hỏi 12.1:**
```
Dừng khi:
    TC(n) > TC(n-1)

HAY phải kiểm tra thêm:
    TC(n) > TC(n-1) AND TC(n) > TC(n-2)?

Hay thử thêm n+1, n+2 để chắc chắn?
```

---

## III. ĐẦU VÀO 

### 13. Đơn vị tham số

**Câu hỏi 13.1:**
```
Các tham số trong paper Example 1:
- D = 600 units: Đơn vị theo NĂM hay NGÀY?
- σ = 5: Độ lệch chuẩn của demand/NĂM hay/NGÀY?
- ts = 0.17: Đơn vị là NĂM, THÁNG, hay NGÀY?
- tT = 1.9: Đơn vị là NĂM, THÁNG, hay NGÀY?
```

**Câu hỏi 13.2:**
```
tT là:
    a) Lead time vận chuyển thuần (chỉ transportation)?
    b) Bao gồm cả setup time của buyer?
```

**Câu hỏi 13.3:**
```
Assumption 5: tT = α·ts
    α = tT/ts = 1.9/0.17 ≈ 11.18

Có đúng không thầy/cô?
```

---

### 14. Giá trị tham số

**Câu hỏi 14.1:**
```
Dùng bộ tham số:
    D = 600, Pmin = 700, Pmax = 1200
    Av0 = 400, A0 = 150, CT = 100
    σ = 5, ρ = 1
    hb = 19.234, hv = 1.7, π = 80
    ξ1 = ξ2 = 1/300
    B1 = 10000, B2 = 20
    θ0 = 0.0002, tT = 1.9

    a = [0.1, 0.15, 0.1]
    b = [0.05, 0.08, 0.04]
    C = [10, 30, 70]

Có khớp 100% với Table 1 trong paper không ?
```

---


## NOTE

### Các lỗi thường gặp:
1. **β2 tính sai** → Q* sai hoàn toàn
2. **β4 tính sai** → P* luôn = Pmin
3. **k1 tử số sai** → k1* quá nhỏ hoặc âm
4. **Step 3 reset sai** → vòng lặp vô hạn
5. **Đơn vị tham số sai** → kết quả lệch 1000 lần



---






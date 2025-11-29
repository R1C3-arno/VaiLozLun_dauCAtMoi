# Supply Chain Optimization - README

## Bài toán là gì?

Tối ưu hóa chuỗi cung ứng 2 cấp (vendor-buyer) để tìm:
- **Q**: lot size (số lượng sản phẩm/lô)
- **P**: production rate (tốc độ sản xuất)
- **n**: số lần vận chuyển
- **k1**: safety factor (hệ số an toàn)
- **Av**: setup cost (chi phí thiết lập)
- **theta**: xác suất sản xuất lỗi

Mục tiêu: **minimize TC** (total cost)

---

## Code này làm gì?

### 1. Tính `ts` và `CR(ts)` - Setup time & Crashing cost

```python
ts_list = compute_ts(a, b)      # setup time cho từng component j
CR_list = compute_CR(ts_list, a, b, C)  # crashing cost tương ứng
```

**Công thức**:
- `ts,j = Σ(i=j+1 to m) ai - Σ(i=1 to j) bi`
- `CR(ts,j) = Cj(ts,j-1 - ts,j) + Σ(i=1 to j) Ci(ai - bi)`

**Lưu ý**: 
- Khi j=1, `ts,0 = Σai` (setup time ban đầu chưa crash)
- ts có thể âm (ví dụ ts,2 = -0.03) → OK, vì `ts_safe = ts + Q/P` sẽ dương

### 2. Tính Total Cost

```python
TC = TCb + TCv
```

- **TCb** (buyer cost): ordering + holding + backorder + crashing
- **TCv** (vendor cost): setup + holding + production + investment + defective

### 3. Thuật toán tối ưu (Iterative Algorithm)

**Vòng lặp ngoài**: Tăng n từ 1 → dừng khi TC(n) >= TC(n-1)

**Vòng lặp trong**: Với mỗi n, lặp đến khi hội tụ:

```python
# Update theo thứ tự:
Q_new = beta1 / (beta2 + beta3*Q)       # equation (10)
P_new = sqrt(beta4 / (D*xi1))           # equation (11)
k1_new = n(Dπ - 2Qhb) / [Dπ(...)]      # equation (12)
Av_new = nQB2 / D                       # equation (13)
theta_new = 2B1 / (ρnDQ)                # equation (14)
```

---

## CÁC LỖI CHƯA FIX ĐƯỢC

### **Lỗi 1: b4 âm → không sqrt được**

**Vấn đề**: 
- Khi n lớn, `term4 = hv*Q*D*(n-2)/2` rất lớn → b4 < 0
- Không thể `sqrt(b4)` → crash

**Fix**:
- Công thức b4 thiếu step handle khi b4 âm , tao chỉ đọc thấy P<0 thì Pmin chứ tao chưa thấy B4 âm thì handle như nào
- Hiện tại toàn bộ hệ thống vẫn còn bất ổn nghiêm trọng do giá trị b4:

- B4 Rất lớn → P bị khóa tại Pmax
- B4 Rất nhỏ → P bị khóa tại Pmin = 700
- B4 Bị âm → gây lỗi sqrt khi tính P
## Làm cho:

- P luôn bị kéo về 700

- n tăng lên > 2000

- TC giảm chậm bất thường

Một số iteration bị vỡ domain số học
```python


def b4(ts, Q, P, n, hv, tT, k1,params):
    D = params["D"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]
    xi2 = params["xi2"]

    ts_safe = ts + Q / P
    ts_safe = max(ts_safe, 1e-8)

    tT_safe = tT + k1 ** 2 * ts_safe
    tT_safe = max(tT_safe, 1e-8)

    term1_1 = -1 / math.sqrt(ts_safe)
    term1=k1*(n-1)*(term1_1+k1/math.sqrt(tT_safe))


    term2 = ((math.sqrt(1+k1**2))-k1)/math.sqrt(ts_safe)

    term3= hb*k1*sigma*Q/(2*math.sqrt(ts_safe))

    term4= hv*Q*D*(n-2)/2

    print(
        f"[b4 DEBUG] n={n}, "
        f"term1={term1:.3e}, "
        f"term2={term2:.3e}, "
        f"term3={term3:.3e}, "
        f"term4={term4:.3e}, "
        f"D*xi2={(D * xi2):.3e}"
    )

    return term3+ (D*pi*sigma/(4*n))*(term2+term1) -term4 +D*xi2
```

## CÁI CHỖ TERM 4 NÓ QUÁ LỚN SO VỚI CÁC TERM CÒN LẠI NÊN GÂY ÂM CỰC LỚN


## Cách chạy

```bash
python test2.py
```

**Output**:
```
ts_list = [0.2, -0.03, -0.17]
CR_list = [0.6, 13.7, 16.6]

[LOG n] n = 1, TC_min_n = 4218.57, TC_min_prev = inf
[LOG n] n = 2, TC_min_n = 4468.99, TC_min_prev = 4218.57

Result
TC = 4218.57
n = 1
j = 1
Q = 122.69
P = 700.0
k1 = 2.085
Av = 4.09
theta = 0.0002
```

---

## Notes quan trọng

### 1. Về ts âm
- ts có thể âm vì `ts,j = Σa[j:] - Σb[:j]`
- Ví dụ: ts,2 = 0.1 - (0.05+0.08) = -0.03
- **OK vì**: `ts_safe = ts + Q/P` → khi Q đủ lớn, ts_safe > 0

### 2. Về n
- Bắt đầu từ n=1
- Tăng dần đến khi TC(n) >= TC(n-1) → STOP
- Best solution thường ở n=1 hoặc n=2

### 3. Về hội tụ
- Vòng lặp trong (Q, P, k1, Av, theta) thường hội tụ sau 10-100 iterations
- Nếu không hội tụ sau 1000 iterations → check lại công thức

### 4. Về b4 âm
- Xảy ra khi n lớn (n≥3)
- Do `term4 = hv*Q*D*(n-2)/2` tăng nhanh
- Handle bằng cách set P = Pmin

---



## References

Paper: "Impact of safety factors and setup time reduction in a two-echelon supply chain management"
- Equation (10): Q*
- Equation (11): P*
- Equation (12): k1*
- Equation (13): Av*
- Equation (14): theta*
- Solution Algorithm: Step 1-6

---

## Developer ở đây là LÊ HOÀNG QUỐC ANH
![29l1](https://cdn-images.dzcdn.net/images/cover/b556c589b533deb0ea6cd636e35362db/1900x1900-000000-80-0-0.jpg)

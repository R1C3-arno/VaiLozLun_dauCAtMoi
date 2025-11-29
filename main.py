import math
import numpy as np

## input ban đầu, tự thêm tùy tụi mày nhé
params = {
    "D": 600.0,
    "Pmin": 700.0,
    "Pmax": 1200.0,
    "Av0": 400.0,
    "A0": 150.0,
    "CT": 100.0,
    "sigma": 5.0,           ##nó là cái σ
    "rho": 1.0,             ## nó là cái ρ
    "hb": 19.234,
    "hv": 1.7,
    "pi": 80.0,             ## nó là π
    "xi1": 1.0/300.0,       ##ξ1 nó là cái này nè ξ1
    "xi2": 1.0/300.0,       ## nó là cái này nè ξ2
    "B1": 10000.0,
    "B2": 20.0,
    "theta0": 0.0002,       ## nó là cái này nè θ0
    "tT": 1.9,
}

a = [0.1, 0.15, 0.1]        ## cái set a1 a2 ae3
b = [0.05, 0.08, 0.04]      ## cái set b1 b2 b3
C = [10.0, 30.0, 70.0]      ## set C1 C2 C3
m = len(a)

def compute_ts(a, b):
    m = len(a)
    ts_list = []

    for j in range(1, m + 1):   # j = 1..m (đúng như paper)
        sum_a = sum(a[j:])     # i = j+1 .. m  (đúng mapping)
        sum_b = sum(b[:j])     # i = 1 .. j
        ts_j = sum_a - sum_b
        ts_list.append(ts_j)

    return ts_list


def compute_CR(ts_list, a, b, C):
    m = len(a)
    ts_max = sum(a)
    CR = []

    for j in range(1, m + 1):   # j = 1..m
        ts_prev = ts_list[j - 2] if j > 1 else 0.0   # t_{s,j-1}
        ts_curr = ts_list[j - 1]                     # t_{s,j}

        term1 = C[j - 1] * max(ts_prev - ts_curr,0)

        # sum i = 1 .. min(j+1, m)  (chú ý: Python index = i-1)
        last_i = min(j + 1, m)      ## đây là j
        term2 = 0.0
        for i in range(1, last_i + 1):  # i from 1 to j+1
            idx = i - 1  # python index
            term2 += C[idx] * (a[idx] - b[idx])

        CR.append(term1 + term2)

    return CR



def compute_TCv(Q, P, n, Av, theta, params):
    D      = params["D"]
    hv     = params["hv"]
    B1     = params["B1"]
    B2     = params["B2"]
    theta0 = params["theta0"]
    rho    = params["rho"]
    Av0    = params["Av0"]
    xi1 = params["xi1"]
    xi2 = params["xi2"]

    if Q <= 0:
        raise ValueError(f"Q <= 0 : Q = {Q}")
    if P <= 0:
        raise ValueError(f"P <= 0 : P = {P}")
    if Av <= 0:
        raise ValueError(f"Av <= 0 : Av = {Av}")
    if theta <= 0:
        raise ValueError(f"theta <= 0 : theta = {theta}")

    CP = xi1 * P + xi2 / P

    T1 = Av * D / (n * Q)

    T2 = hv * Q / 2 * (
        n * (1 - D / P) - 1 + 2 * D / P
    )

    T3 = D * CP

    T4 = B1 * math.log(theta / theta0)

    T5 = B2 * math.log(Av0 / Av)

    T6 = rho * D * theta * n * Q / 2

    return T1 + T2 + T3 + T4 + T5 + T6

def compute_TCb(Q, P, n, k1, ts, tT, CR_ts, params):
    D     = params["D"]
    A0    = params["A0"]
    CT    = params["CT"]
    hb    = params["hb"]
    sigma = params["sigma"]
    pi    = params["pi"]

    ts_safe = max(ts + Q / P, 1e-10)

    # --------- Term 1 ----------
    T1 = (A0 + n * CT) * D / (n * Q)

    # --------- Term 2 ----------
    T2 = hb * (Q / 2 + k1 * sigma * math.sqrt(ts_safe))

    # --------- E(x1 - R1)+ ----------
    Ex1 = (sigma / 2) * math.sqrt(ts_safe) * \
          (math.sqrt(1 + k1**2) - k1)

    # --------- E(x2 - R2)+ ----------
    Ex2 = (sigma / 2) * math.sqrt(tT) * (
        math.sqrt(1 + k1**2 * ts_safe / tT)
        - k1 * math.sqrt(ts_safe / tT)
    )

    T3 = D * pi / (n * Q) * Ex1
    T4 = D * pi * (n - 1) / (n * Q) * Ex2

    # --------- Crashing cost ----------
    T5 = D * n * CR_ts / (n * Q)

    return T1 + T2 + T3 + T4 + T5

def compute_TC_total(Q, P, n, k1, Av, theta, params, ts, tT, CR_ts):
    TCb = compute_TCb(Q, P, n, k1, ts, tT, CR_ts, params)
    TCv = compute_TCv(Q, P, n, Av, theta, params)
    return TCb + TCv

def daoham_Tcb_theoQ(Q, P, n, k1, ts, tT, CR_ts, params):
    # địt con mẹ dài vãi lồn nên tách ra term 1 2 3 4 ,5,6 nhé
    D = params["D"]
    A0 = params["A0"]
    CT = params["CT"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]

    ts_safe = max(ts + Q / P, 1e-10)
    #đạo hàm cái củ cặc D(A0+nCt)/nQ
    term1 = -((A0+n*CT) * D) / (n*Q**2)
    # đạo hàm cái củ cặc hb/2ksigma/2Psqrt(ts+Q/p
    term2 = (hb /2)+(hb*k1*sigma)/(2*P*math.sqrt(ts_safe))
    # đạo hàm cái củ cặc hb/2ksigma/2Psqrt(ts+Q/p
    term3_1 = (D*pi*sigma*(math.sqrt(1+k1**2)-k1))/(4*n*P*Q*math.sqrt(ts_safe))
    term3_2 =  ((D*pi)/(n*Q**2)) * (sigma/2)*(math.sqrt(ts_safe)*(math.sqrt(1+k1**2)-k1))
    term3= term3_1 - term3_2

    # Ex2 gốc
    Ex2 = (sigma / 2) * math.sqrt(tT) * (
            math.sqrt(1 + k1 ** 2 * ts_safe / tT)
            - k1 * math.sqrt(ts_safe / tT)
    )

    # đạo hàm Ex2 theo Q
    u = ts_safe
    dEx2_dQ = (sigma / (2 * P)) * (
            (k1 ** 2) / (2 * math.sqrt(tT) * math.sqrt(1 + k1 ** 2 * u / tT))
            - k1 / (2 * math.sqrt(u))
    )

    term4 = (
            - (D * pi * (n - 1)) / (n * Q ** 2) * Ex2
            + (D * pi * (n - 1)) / (n * Q) * dEx2_dQ
    )

    term5 = -D*CR_ts/(Q**2)

    daoham_tcb = term1 + term2 + term3 + term4 + term5
    return daoham_tcb

def daoham_Tcv_theoQ(Q, P, n, Av, theta, params):
    D = params["D"]
    hv = params["hv"]
    rho = params["rho"]

    dT1 = - Av * D / (n * Q ** 2)

    const = n * (1 - D / P) - 1 + 2 * D / P
    dT2 = hv / 2 * const

    dT6 = rho * D * theta * n / 2

    return dT1 + dT2 + dT6

def b1(D, n, A0, CT, pi, sigma, ts, Q, P, tT, k1, CR, Av):
    ts_safe = max(ts + Q / P, 1e-8)

    term1 = (sigma / 2) * math.sqrt(ts_safe) * (math.sqrt(1 + k1**2) - k1)

    term2 = ((n - 1) * sigma / 2) * math.sqrt(tT + k1**2 * ts_safe)

    term3 = n * CR / pi

    return (D / n) * (A0 + n * CT + pi * (term1 + term2 + term3) + Av)

def b2(D, pi, sigma, ts, Q, P, tT, k1, n):
    ts_safe = ts + Q / P
    ts_safe = max(ts + Q / P, 1e-8)

    tT_safe = tT + k1 ** 2 * ts_safe
    tT_safe = max(tT + k1 ** 2 * ts_safe, 1e-8)

    term1 = (math.sqrt(1 + k1 ** 2) - k1) / math.sqrt(ts_safe)

    term2_1 = -1 / math.sqrt(ts_safe)
    term2 = (n - 1) * k1 * (term2_1+k1/math.sqrt(tT_safe))

    return (D * pi * sigma) / (4 * n * P) * (term1 + term2 )

def b3(hb, sigma, k1, ts, Q, P, hv, n, D, rho, theta):
    ts_safe = ts + Q / P
    ts_safe = max(ts + Q / P, 1e-8)

    term1 = hb * (0.5 + (k1 * sigma) / (2 * P * math.sqrt(ts_safe)))

    term2 = (hv / 2) * (n * (1 - D / P) - 1 + 2 * D / P)

    term3 = (rho * n * D * theta) / 2

    return term1 + term2 + term3



## hàm này tính Q* nè mấy thằng lồn
def giai_nghiem_daohamTc_theoQ(Q0, P, n, Av, theta,ts, tT, k1, CR, params):

    D = params["D"]
    A0 = params["A0"]
    CT = params["CT"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]
    hv = params["hv"]
    rho = params["rho"]

    Q = Q0
    for _ in range(100):
        beta1 = b1(D, n, A0, CT, pi, sigma, ts, Q, P, tT, k1, CR, Av)
        beta2 = b2(D, pi, sigma, ts, Q, P, tT, k1, n)
        beta3 = b3(hb, sigma, k1, ts, Q, P, hv, n, D, rho, theta)

        Q_new = beta1/(beta2+beta3*Q)
        if Q_new <= 0 or not np.isfinite(Q_new):
            Q_new = Q

        ## điều kiện hội tụ ( kèm Q_new vào ko nó nhảy sập lồn)
        if abs(Q_new - Q)/max(Q,1e-8) < 1e-6:
            return Q_new
        Q=Q_new


    ## không hội tụ trả về Q=Q0
    return Q

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


def daoham_Tc_theoP(P, b4_value, params):
    D = params["D"]
    xi1 = params["xi1"]

    return - b4_value / (P ** 2) + D * xi1


## tính P* nè mấy con gà
def giai_nghiem_daohamTc_theoP(b4, params):
    D = params["D"]
    xi1 = params["xi1"]
    Pmin = params["Pmin"]
    Pmax = params["Pmax"]

    if b4 <= 0:
        return Pmin


    P= math.sqrt(b4 / (D * xi1))

    if P < Pmin:
        P = Pmin
    elif P > Pmax:
        P = Pmax


    return P


## đạo hàm TC theo k1
def daoham_Tc_theoK1(k1, Q, P, n, ts, tT, params):
    D = params["D"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]

    ts_safe = ts + Q / P
    tT_safe = tT + k1 ** 2 * ts_safe

    term1_1 = k1/math.sqrt(1+k1**2) -1
    term1_2 = (n-1)*(k1*math.sqrt(ts_safe/tT_safe)-1)

    term1 = hb+(pi*D/(2*n*Q)) *(term1_1+term1_2)

    term2= sigma*math.sqrt(ts_safe)

    return term2*term1

## hàm tính t* (tử chia mẫu)

def capnhat_k1_theo_congthuc(k1, Q, P, n, ts, tT, params):
    D  = params["D"]
    hb = params["hb"]
    pi = params["pi"]

    u = ts + Q / P
    u = max(u, 1e-8)

    k1_safe= min(abs(k1),1e3)
    k1_safe = max(k1_safe,1e-6)
    v = tT + k1**2 * u
    v = max(v, 1e-8)


    tu = n * (D * pi - 2 * Q * hb)

    mau = D * pi * (
        1 / math.sqrt(1 + k1**2)
        + (n - 1) * math.sqrt(u / v)
    )

    # ===== CHỐNG CHIA CHO 0 =====
    if abs(mau) < 1e-10:
        return k1_safe  # giữ nguyên, không update

    k1_new = tu / mau

    # ===== ÉP MIỀN HỢP LỆ SAU UPDATE =====
    k1_new = max(min(k1_new, 1e3), 1e-6)

    return k1_new


##hàm này tìm k1* nè mấy thằng nhóc

def giai_nghiem_daohamTc_theoK1(Q, P, n, ts, tT, k1_old, params):
    k1 = k1_old
    for _ in range(100):   # cái in range là cái số vòng lặp tối đa (đây set mặc định 100, mốt tự chỉnh theo ý)
        k1_new = capnhat_k1_theo_congthuc(k1, Q, P, n, ts, tT, params)

        ## ghìm giá trị ,ko cho nó nhảy lung tung
        if abs(k1_new - k1) < 1e-6: #=> cais 1e-6 là tolerance
            return k1_new
        k1=k1_new
    return k1

##hàm này tính Av * nè mấy cu
def giai_nghiem_daohamTc_theoAv(Q,n,params):
    D = params["D"]
    B2 = params["B2"]
    return n*Q*B2/D
## hàm này giải nghiệm theta*
def giai_nghiem_daohamTc_theoTheta(Q,n,params):
    B1 = params["B1"]
    rho = params["rho"]
    D = params["D"]
    return 2*B1/(rho*n*D*Q)

print("\nMain --Test")

if __name__ == "__main__":

    # ===== MAP params -> BIẾN TRỰC TIẾP =====
    D = params["D"]
    Pmin = params["Pmin"]
    Pmax = params["Pmax"]

    Av0 = params["Av0"]
    theta0 = params["theta0"]
    tT = params["tT"]

    k1=1.0

    # sort thằng set C trước (nếu m chưa sort trong compute_CR)
    # (nên sort theo C tăng dần nếu paper yêu cầu)

    ## khởi tạo ts,j --> step 1
    ts_list = compute_ts(a, b)

    ## khởi tạo CR(ts)  --> step 1
    CR_list = compute_CR(ts_list, a, b, C)

    print("ts_list =", ts_list)
    print("CR_list =", CR_list)

    ## khởi tạo biến best_overral
    best_overall = {
        "TC": float("inf"),
        "n": None,
        "j": None,
        "Q": None,
        "P": None,
        "k1": None,
        "Av": None,
        "theta": None
    }

    best_solution = None
    n = 1

    Q = 200.0

    # số lặp tối đa
    Max_interation = 1000000

    ## khởi tạo TC min ban đầu cho mỗi n = dương vô cực
    TC_min_prev = float("inf")

    ## vòng lặp While (lặp cho đến khi tự dừng bằng điều kiện Tc(n) bắt đầu tăng trở lại
    while n<=Max_interation:

        TC_min_n = float("inf")

        ## boolean found best
        best_in_n = None

        for j in range(len(ts_list)):
            ts = ts_list[j]
            CR = CR_list[j]

            # STEP 2: INIT
            Q = 200.0
            P = (Pmin + Pmax) / 2
            k1 = 2.0
            Av = Av0
            theta = theta0



            # ===== STEP 4: GIẢI LẶP HỘI TỤ Q, P, k1, Av, theta =====

            for _ in range(Max_interation):

                Q_old, P_old, k1_old = Q, P, k1

                Q = giai_nghiem_daohamTc_theoQ(
                    Q, P, n, Av, theta, ts, tT, k1, CR, params
                )

                giatri_b4 = b4(ts, Q, P, n, params["hv"], tT, k1, params)
                print("Be4",giatri_b4)
                P = giai_nghiem_daohamTc_theoP(giatri_b4, params)







                k1 = giai_nghiem_daohamTc_theoK1(
                    Q, P, n, ts, tT, k1, params
                )
                # ===== CHẶN MIỀN HỢP LỆ THEO PAPER =====
                k1 = max(0.001, min(k1, 50.0))

                Av = giai_nghiem_daohamTc_theoAv(Q, n, params)
                theta = giai_nghiem_daohamTc_theoTheta(Q, n, params)

                # ===== ĐIỀU KIỆN HỘI TỤ =====
                if (
                    abs(Q - Q_old) / max(Q_old, 1e-8) < 1e-6 and
                    abs(P - P_old) / max(P_old, 1e-8) < 1e-6 and
                    abs(k1 - k1_old) < 1e-6
                ):
                    break

            # ===== STEP 6: TÍNH TC =====
            TC = compute_TC_total(Q, P, n, k1, Av, theta, params, ts, tT, CR)

            if TC < TC_min_n:
                TC_min_n = TC
                best_in_n = {
                    "TC": TC,
                    "n": n,
                    "j": j,
                    "Q": Q,
                    "P": P,
                    "k1": k1,
                    "Av": Av,
                    "theta": theta
                }

        print(f"[LOG n] n = {n}, TC_min_n = {TC_min_n}, TC_min_prev = {TC_min_prev}")

        # ===== ĐIỀU KIỆN DỪNG THEO n =====
        if TC_min_n >= TC_min_prev:
            break

        TC_min_prev = TC_min_n

        if best_in_n is not None and best_in_n["TC"] < best_overall["TC"]:
            best_overall = best_in_n

        n += 1


    print("\nResult")
    for k, v in best_overall.items():
        print(k, "=", v)

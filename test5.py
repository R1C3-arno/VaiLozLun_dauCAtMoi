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
    "sigma": 5.0,  ##nó là cái σ
    "rho": 1.0,  ## nó là cái ρ
    "hb": 19.234,
    "hv": 1.7,
    "pi": 80.0,  ## nó là π
    "xi1": 1.0 / 300.0,  ##ξ1 nó là cái này nè ξ1
    "xi2": 1.0 / 300.0,  ## nó là cái này nè ξ2
    "B1": 10000.0,
    "B2": 20.0,
    "theta0": 0.0002,  ## nó là cái này nè θ0
    "tT": 1.9,
}

a = [0.1, 0.15, 0.1]  ## cái set a1 a2 ae3
b = [0.05, 0.08, 0.04]  ## cái set b1 b2 b3
C = [10.0, 30.0, 70.0]  ## set C1 C2 C3
m = len(a)




m = len(a)

def compute_ts(a, b):
    m = len(a)
    ts_max = sum(a)
    ts_min = sum(b)
    ts_list = []

    for j in range(1, m + 1):  # j = 1..m (đúng như paper)
        sum_a = sum(a[i] for i in range(j, m))  # i từ j đến m-1 (trong Python)
        sum_b = sum(b[i] for i in range(0, j))  # i từ 0 đến j-1
        ts_j = sum_a - sum_b

        ## check miền b<ts<a
        if ts_min <= ts_j <= ts_max:
            ts_list.append(ts_j)
        else:
            print(f"⚠️ Bỏ ts_j={ts_j} (ngoài [{ts_min}, {ts_max}])")

    return ts_list


def compute_CR(ts_list, a, b, C):
    m = len(a)
    ts_max = sum(a)

    CR = []

    #sort Cr theo thứ tự tăng dần
    zip_all = sorted(zip(a, b, C), key=lambda x: x[2])
    a, b, C = map(list, zip(*zip_all))

    for j in range(1, len(ts_list) + 1):  # j = 1..m
        ts_prev = ts_list[j - 2] if j > 1 else ts_max  # t_{s,j-1}
        ts_curr = ts_list[j - 1]  # t_{s,j}

        term1 = C[j - 1] * max(ts_prev - ts_curr, 0)

        # sum i = 1 .. min(j+1, m)  (chú ý: Python index = i-1)
        last_i = min(j + 1, m)
        term2 = 0.0
        for i in range(1, last_i):  # i from 1 to j+1
            idx = i - 1  # python index
            term2 += C[idx] * (a[idx] - b[idx])

        CR.append(term1 + term2)

    return CR


def compute_TCv(Q, P, n, Av, theta, params):
    D = params["D"]
    hv = params["hv"]
    B1 = params["B1"]
    B2 = params["B2"]
    theta0 = params["theta0"]
    rho = params["rho"]
    Av0 = params["Av0"]
    xi1 = params["xi1"]
    xi2 = params["xi2"]


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
    D = params["D"]
    A0 = params["A0"]
    CT = params["CT"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]

    ts_safe = max(ts + Q / P, 1e-10)

    # --------- Term 1 ----------
    T1 = (A0 + n * CT) * D / (n * Q)

    # --------- Term 2 ----------
    T2 = hb * (Q / 2 + k1 * sigma * math.sqrt(ts_safe))

    # --------- E(x1 - R1)+ ----------
    Ex1 = (sigma / 2) * math.sqrt(ts_safe) * \
          (math.sqrt(1 + k1 ** 2) - k1)

    # --------- E(x2 - R2)+ ----------
    Ex2 = (sigma / 2) * math.sqrt(tT) * (
            math.sqrt(1 + k1 ** 2 * ts_safe / tT)
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
    # đạo hàm cái củ cặc D(A0+nCt)/nQ
    term1 = -((A0 + n * CT) * D) / (n * Q ** 2)
    # đạo hàm cái củ cặc hb/2ksigma/2Psqrt(ts+Q/p
    term2 = (hb / 2) + (hb * k1 * sigma) / (2 * P * math.sqrt(ts_safe))
    # đạo hàm cái củ cặc hb/2ksigma/2Psqrt(ts+Q/p
    term3_1 = (D * pi * sigma * (math.sqrt(1 + k1 ** 2) - k1)) / (4 * n * P * Q * math.sqrt(ts_safe))
    term3_2 = ((D * pi) / (n * Q ** 2)) * (sigma / 2) * (math.sqrt(ts_safe) * (math.sqrt(1 + k1 ** 2) - k1))
    term3 = term3_1 - term3_2

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

    term5 = -D * CR_ts / (Q ** 2)

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

    term1 = (sigma / 2) * math.sqrt(ts_safe) * (math.sqrt(1 + k1 ** 2) - k1)

    term2 = ((n - 1) * sigma / 2) * math.sqrt(tT + k1 ** 2 * ts_safe)

    term3 = n * CR / pi

    return (D / n) * (A0 + n * CT + pi * (term1 + term2 + term3) + Av)


def b2(D, pi, sigma, ts, Q, P, tT, k1, n):
    ts_safe = ts + Q / P
    ts_safe = max(ts + Q / P, 1e-8)

    tT_safe = tT + k1 ** 2 * ts_safe
    tT_safe = max(tT + k1 ** 2 * ts_safe, 1e-8)

    term1 = (math.sqrt(1 + k1 ** 2) - k1) / math.sqrt(ts_safe)

    term2_1 = -1 / math.sqrt(ts_safe)
    term2 = (n - 1) * k1 * (term2_1 + k1 / math.sqrt(tT_safe))

    return (D * pi * sigma) / (4 * n * P) * (term1 + term2)


def b3(hb, sigma, k1, ts, Q, P, hv, n, D, rho, theta):
    ts_safe = ts + Q / P
    ts_safe = max(ts + Q / P, 1e-8)

    term1 = hb * (0.5 + (k1 * sigma) / (2 * P * math.sqrt(ts_safe)))

    term2 = (hv / 2) * (n * (1 - D / P) - 1 + 2 * D / P)

    term3 = (rho * n * D * theta) / 2

    return term1 + term2 + term3


## hàm này tính Q* nè mấy thằng lồn

def giai_nghiem_daohamTc_theoQ(Q0, P, n, Av, theta, ts, tT, k1, CR, params):
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

        Q_new = beta1 / (beta2 + beta3 * Q)
        if not np.isfinite(Q_new):
            Q_new = Q
        Q_new = max(Q_new, 1e-3)

        ## điều kiện hội tụ ( kèm Q_new vào ko nó nhảy sập lồn)
        if abs(Q_new - Q) / max(Q, 1e-8) < 1e-6:
            return Q_new
        Q = Q_new

    ## không hội tụ trả về Q=Q0
    return Q

def b4(ts, Q, P, n, hv, tT, k1, params):
    D = params["D"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]
    xi2 = params["xi2"]

    ts_safe = ts + Q / P
    ts_sqrt = math.sqrt(ts_safe)

    tT_safe = tT + k1 ** 2 * ts_safe
    tT_sqrt = math.sqrt(tT_safe)

    term1 = hb * k1 * sigma * Q / (2 * ts_sqrt)

    trong_ngoac_1 = (math.sqrt(1 + k1 ** 2) - k1) / ts_sqrt
    trong_ngoac_2 = (n - 1) * (k1 /tT_sqrt - 1 / ts_sqrt)

    term2 = (D * pi * sigma / (4 * n)) * (trong_ngoac_1 + trong_ngoac_2)

    term3 = - hv * Q * D * (n - 2) / 2
    term4 = D * xi2

    return term1 + term2 + term3 + term4


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

    if b4 / (D * xi1) <= 0:
        print("Log b4 / (D * xi1) =  ",b4 / (D * xi1) )
        return Pmin

    P = math.sqrt(b4 / (D * xi1))

    return P


## đạo hàm TC theo k1
def daoham_Tc_theoK1(k1, Q, P, n, ts, tT, params):
    D = params["D"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]

    ts_safe = ts + Q / P
    tT_safe = tT + k1 ** 2 * ts_safe

    term1_1 = k1 / math.sqrt(1 + k1 ** 2) - 1
    term1_2 = (n - 1) * (k1 * math.sqrt(ts_safe / tT_safe) - 1)

    term1 = hb + (pi * D / (2 * n * Q)) * (term1_1 + term1_2)

    term2 = sigma * math.sqrt(ts_safe)

    return term2 * term1


## hàm tính t* (tử chia mẫu)

def capnhat_k1_theo_congthuc(k1, Q, P, n, ts, tT, params):
    D = params["D"]
    hb = params["hb"]
    pi = params["pi"]

    u = ts + Q / P

    v = tT + k1 ** 2 * u

    tu = n * (D * pi - 2 * Q * hb)

    mau = D * pi * (
            1 / math.sqrt(1 + k1 ** 2)
            + (n - 1) * math.sqrt(u / v)
    )

    k1_new = tu / mau

    return k1_new


##hàm này tìm k1* nè mấy thằng nhóc

def giai_nghiem_daohamTc_theoK1(Q, P, n, ts, tT, k1_old, params):


    k1 = capnhat_k1_theo_congthuc(k1_old, Q, P, n, ts, tT, params)

    return k1


##hàm này tính Av * nè mấy cu
def giai_nghiem_daohamTc_theoAv(Q, n, params):
    D = params["D"]
    B2 = params["B2"]
    return n * Q * B2 / D


## hàm này giải nghiệm theta*
def giai_nghiem_daohamTc_theoTheta(Q, n, params):
    B1 = params["B1"]
    rho = params["rho"]
    D = params["D"]
    return 2 * B1 / (rho * n * D * Q)


if __name__ == "__main__":

    print("\n========== PARAMS ==========")
    for k, v in params.items():
        print(f"{k:8s} = {v}")
    print("============================\n")

    # =========================
    # STEP 1: INITIALIZATION
    # =========================
    n = 1
    D = params["D"]
    Pmin = params["Pmin"]
    Pmax = params["Pmax"]
    Av0 = params["Av0"]
    theta0 = params["theta0"]
    tT = params["tT"]

    # Sort theo crashing cost (QUAN TRỌNG)
    zip_all = sorted(zip(a, b, C), key=lambda x: x[2])
    a_sorted, b_sorted, C_sorted = map(list, zip(*zip_all))

    ts_list = compute_ts(a_sorted, b_sorted)
    CR_list = compute_CR(ts_list, a_sorted, b_sorted, C_sorted)
    m = len(ts_list)

    print("=" * 80)
    print("STEP 1: INITIALIZATION")
    print("=" * 80)
    print(f"ts_list = {ts_list}")
    print(f"CR_list = {CR_list}")
    print(f"m = {m} components")

    best_TC_overall = float("inf")
    best_solution_overall = None

    MAX_N = 15
    TC_prev_n = float("inf")

    # =========================
    # STEP 5 OUTER LOOP: n
    # =========================
    while n <= MAX_N:
        print("\n" + "=" * 80)
        print(f"STEP 5: TRYING n = {n}")
        print("=" * 80)

        TC_best_for_n = float("inf")
        solution_best_for_n = None

        # =========================
        # STEP 2: FOR EACH ts_j
        # =========================
        for j in range(1, m + 1):
            ts = ts_list[j - 1]
            CR = CR_list[j - 1]

            print(f"\n{'─'*60}")
            print(f"STEP 2: n = {n}, j = {j}/{m}")
            print(f"ts = {ts:.4f}, CR = {CR:.4f}")
            print(f"{'─'*60}")

            # =========================
            # STEP 2a: INITIAL VALUES
            # =========================
            Q = 200.0
            P = 900.0
            k1 = 4.0
            Av = Av0
            theta = theta0

            # =====================================
            # 🔁 STEP 3 OUTER CONSTRAINT LOOP
            # =====================================
            MAX_CONSTRAINT_ITER = 50
            feasible = False

            for constraint_iter in range(MAX_CONSTRAINT_ITER):

                print(f"\n  ▶ Constraint iteration = {constraint_iter}")

                # =========================
                # STEP 2b–2e: CONVERGENCE LOOP
                # =========================
                MAX_ITER = 50
                tolerance = 1e-6

                for iteration in range(MAX_ITER):
                    Q_prev, P_prev, k1_prev = Q, P, k1
                    Av_prev, theta_prev = Av, theta

                    # ---------- STEP 2b ----------
                    beta4_val = b4(ts, Q, P, n, params["hv"], tT, k1, params)
                    P = giai_nghiem_daohamTc_theoP(beta4_val, params)

                    if P <= Pmin: P = Pmin
                    if P >= Pmax: P = Pmax

                    Q = giai_nghiem_daohamTc_theoQ(Q, P, n, Av, theta, ts, tT, k1, CR, params)

                    # ---------- STEP 2c ----------
                    k1 = giai_nghiem_daohamTc_theoK1(Q, P, n, ts, tT, k1, params)

                    # ---------- STEP 2d ----------
                    Av = giai_nghiem_daohamTc_theoAv(Q, n, params)
                    theta = giai_nghiem_daohamTc_theoTheta(Q, n, params)

                    # ---------- STEP 2e ----------
                    if (
                        abs(Q - Q_prev) < tolerance and
                        abs(P - P_prev) < tolerance and
                        abs(k1 - k1_prev) < tolerance and
                        abs(Av - Av_prev) < tolerance and
                        abs(theta - theta_prev) < tolerance
                    ):
                        break

                print(f"    Converged at iter {iteration+1}")
                print(f"    Q={Q:.4f}, P={P:.4f}, k1={k1:.4f}, Av={Av:.6f}, θ={theta:.8f}")

                # =========================
                # STEP 3: CHECK CONSTRAINTS (ĐÚNG FLOWCHART)
                # =========================
                print(f"    CHECK: Av={Av:.6f} vs Av0={Av0:.6f}")
                print(f"           θ={theta:.8f} vs θ0={theta0:.8f}")

                # ✅ STEP 3a
                if Av <= Av0 and theta <= theta0:
                    print("    ✅ BOTH SATISFIED → GO TO STEP 4")
                    feasible = True
                    break

                # ✅ STEP 3b
                elif Av > Av0 and theta <= theta0:
                    print("    ⚠ Av VIOLATED → SET Av = Av0 → GO BACK TO STEP 2")
                    Av = Av0
                    continue

                # ✅ STEP 3c
                elif Av <= Av0 and theta > theta0:
                    print("    ⚠ θ VIOLATED → SET θ = θ0 → GO BACK TO STEP 2")
                    theta = theta0
                    continue

                # ✅ STEP 3d
                else:
                    print("    ⚠ BOTH VIOLATED → SET Av = Av0, θ = θ0 → GO BACK TO STEP 2")
                    Av = Av0
                    theta = theta0
                    continue

            # Nếu không tìm được nghiệm hợp lệ
            if not feasible:
                print("  ❌ DISCARD THIS j")
                continue

            # =========================
            # STEP 4: COMPUTE TC
            # =========================
            TC_current = compute_TC_total(Q, P, n, k1, Av, theta, params, ts, tT, CR)

            print(f"\nSTEP 4: TC(n={n}, j={j}) = ${TC_current:.2f}")

            if TC_current < TC_best_for_n:
                TC_best_for_n = TC_current
                solution_best_for_n = {
                    'n': n, 'j': j, 'Q': Q, 'P': P, 'k1': k1,
                    'Av': Av, 'theta': theta, 'ts': ts, 'CR': CR,
                    'TC': TC_current
                }

        # =========================
        # STEP 5: COMPARE TC(n)
        # =========================
        print(f"\n{'='*80}")
        print(f"STEP 5: TC_best(n={n}) = ${TC_best_for_n:.2f}")
        print(f"        TC_previous = ${TC_prev_n:.2f}")

        if TC_best_for_n < TC_prev_n:
            print("  ✅ TC IMPROVED → CONTINUE")
            TC_prev_n = TC_best_for_n
            best_solution_overall = solution_best_for_n
            n += 1
        else:
            print("  ❌ TC INCREASED → STOP")
            break

    # =========================
    # STEP 6: FINAL SOLUTION
    # =========================
    print("\n" + "="*80)
    print("STEP 6: FINAL OPTIMAL SOLUTION")
    print("="*80)

    if best_solution_overall:
        sol = best_solution_overall
        print(f"n* = {sol['n']}")
        print(f"j* = {sol['j']}")
        print(f"Q* = {sol['Q']:.4f}")
        print(f"P* = {sol['P']:.4f}")
        print(f"k1* = {sol['k1']:.4f}")
        print(f"Av* = {sol['Av']:.6f}")
        print(f"θ* = {sol['theta']:.8f}")
        print(f"TC* = ${sol['TC']:.2f}")
    else:
        print("⚠ NO FEASIBLE SOLUTION")

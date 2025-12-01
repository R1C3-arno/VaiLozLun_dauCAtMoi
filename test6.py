import math
import numpy as np

# =========================
# INPUT PARAMETERS
# =========================
params = {
    "D": 600.0,
    "Pmin": 700.0,
    "Pmax": 1200.0,
    "Av0": 400.0,
    "A0": 150.0,
    "CT": 100.0,
    "sigma": 5.0,
    "rho": 1.0,
    "hb": 19.234,
    "hv": 1.7,
    "pi": 80.0,
    "xi1": 1.0 / 300.0,
    "xi2": 1.0 / 300.0,
    "B1": 10000.0,
    "B2": 20.0,
    "theta0": 0.0002,
    "tT": 1.9,
}

a = [0.1, 0.15, 0.1]
b = [0.05, 0.08, 0.04]
C = [10.0, 30.0, 70.0]


# =========================
# HELPER FUNCTIONS (GIỮ NGUYÊN)
# =========================
def compute_ts(a, b):
    m = len(a)
    ts_max = sum(a)
    ts_min = sum(b)
    ts_list = []
    for j in range(1, m + 1):
        sum_a = sum(a[i] for i in range(j, m))
        sum_b = sum(b[i] for i in range(0, j))
        ts_j = sum_a - sum_b
        if ts_min - 1e-9 <= ts_j <= ts_max + 1e-9:
            ts_list.append(ts_j)
    return ts_list


def compute_CR(ts_list, a, b, C):
    m = len(a)
    ts_max = sum(a)
    CR = []
    for j in range(1, len(ts_list) + 1):
        ts_prev = ts_list[j - 2] if j > 1 else ts_max
        ts_curr = ts_list[j - 1]
        term1 = C[j - 1] * max(ts_prev - ts_curr, 0)
        last_i = min(j + 1, m)
        term2 = 0.0
        for i in range(1, last_i):
            idx = i - 1
            term2 += C[idx] * (a[idx] - b[idx])
        CR.append(term1 + term2)
    return CR


def compute_TC_total(Q, P, n, k1, Av, theta, params, ts, tT, CR_ts):
    # Hàm tính tổng chi phí để so sánh kết quả
    D = params["D"]
    hb = params["hb"]
    hv = params["hv"]
    A0 = params["A0"]
    CT = params["CT"]
    pi = params["pi"]
    sigma = params["sigma"]
    rho = params["rho"]
    xi1 = params["xi1"]
    xi2 = params["xi2"]
    B1 = params["B1"]
    B2 = params["B2"]
    Av0 = params["Av0"]
    theta0 = params["theta0"]

    # 1. Buyer Cost
    ts_safe = max(ts + Q / P, 1e-10)
    T1 = (A0 + n * CT) * D / (n * Q)
    T2 = hb * (Q / 2 + k1 * sigma * math.sqrt(ts_safe))
    Ex1 = (sigma / 2) * math.sqrt(ts_safe) * (math.sqrt(1 + k1 ** 2) - k1)
    Ex2 = (sigma / 2) * math.sqrt(tT) * (math.sqrt(1 + k1 ** 2 * ts_safe / tT) - k1 * math.sqrt(ts_safe / tT))
    T3 = D * pi / (n * Q) * Ex1
    T4 = D * pi * (n - 1) / (n * Q) * Ex2
    T5 = D * n * CR_ts / (n * Q)
    TCb = T1 + T2 + T3 + T4 + T5

    # 2. Vendor Cost
    CP = xi1 * P + xi2 / P
    TV1 = Av * D / (n * Q)
    TV2 = hv * Q / 2 * (n * (1 - D / P) - 1 + 2 * D / P)
    TV3 = D * CP

    # Investment cost logic: Nếu theta < theta0 thì mới mất phí đầu tư
    # Tuy nhiên theo công thức tổng quát thì cứ cộng log vào, nếu theta=theta0 thì log(1)=0
    TV4 = B1 * math.log(theta0 / theta) if theta < theta0 else 0
    TV5 = B2 * math.log(Av0 / Av) if Av < Av0 else 0
    TV6 = rho * D * theta * n * Q / 2
    TCv = TV1 + TV2 + TV3 + TV4 + TV5 + TV6

    return TCb + TCv


# --- Các hàm Beta tính đạo hàm ---
def b1(D, n, A0, CT, pi, sigma, ts, Q, P, tT, k1, CR, Av):
    ts_safe = max(ts + Q / P, 1e-8)
    term1 = (sigma / 2) * math.sqrt(ts_safe) * (math.sqrt(1 + k1 ** 2) - k1)
    term2 = ((n - 1) * sigma / 2) * math.sqrt(tT + k1 ** 2 * ts_safe)
    term3 = n * CR / pi
    return (D / n) * (A0 + n * CT + pi * (term1 + term2 + term3) + Av)


def b2(D, pi, sigma, ts, Q, P, tT, k1, n):
    ts_safe = max(ts + Q / P, 1e-8)
    tT_safe = max(tT + k1 ** 2 * ts_safe, 1e-8)
    term1 = (math.sqrt(1 + k1 ** 2) - k1) / math.sqrt(ts_safe)
    term2 = (n - 1) * k1 * (-1 / math.sqrt(ts_safe) + k1 / math.sqrt(tT_safe))
    return (D * pi * sigma) / (4 * n * P) * (term1 + term2)


def b3(hb, sigma, k1, ts, Q, P, hv, n, D, rho, theta):
    ts_safe = max(ts + Q / P, 1e-8)
    term1 = hb * (0.5 + (k1 * sigma) / (2 * P * math.sqrt(ts_safe)))
    term2 = (hv / 2) * (n * (1 - D / P) - 1 + 2 * D / P)
    term3 = (rho * n * D * theta) / 2
    return term1 + term2 + term3


def b4(ts, Q, P, n, hv, tT, k1, params):
    D = params["D"]
    hb = params["hb"]
    sigma = params["sigma"]
    pi = params["pi"]
    xi2 = params["xi2"]
    ts_safe = ts + Q / P
    ts_sqrt = math.sqrt(ts_safe)
    tT_safe = tT + k1 ** 2 * ts_safe
    term1 = hb * k1 * sigma * Q / (2 * ts_sqrt)
    trong_ngoac = (math.sqrt(1 + k1 ** 2) - k1) / ts_sqrt + (n - 1) * (k1 / math.sqrt(tT_safe) - 1 / ts_sqrt)
    term2 = (D * pi * sigma / (4 * n)) * trong_ngoac
    term3 = - hv * Q * D * (n - 2) / 2
    term4 = D * xi2
    return term1 + term2 + term3 + term4


# --- Các hàm giải phương trình ---
def solve_P(b4_val, params):
    D = params["D"]
    xi1 = params["xi1"]
    if b4_val <= 0: return params["Pmin"]
    val = b4_val / (D * xi1)
    return math.sqrt(val)


def solve_Q(b1, b2, b3):
    # Solve: b3*Q^2 + b2*Q - b1 = 0
    delta = b2 ** 2 + 4 * b3 * b1
    return (-b2 + math.sqrt(delta)) / (2 * b3)


def solve_k1(k1_old, Q, P, n, ts, tT, params):
    D = params["D"]
    hb = params["hb"]
    pi = params["pi"]
    ts_safe = ts + Q / P
    tu = n * (D * pi - 2 * Q * hb)
    if tu <= 0: return 0.01
    tT_safe = tT + k1_old ** 2 * ts_safe
    mau = D * pi * (1 / math.sqrt(1 + k1_old ** 2) + (n - 1) * math.sqrt(ts_safe / tT_safe))
    if mau == 0: return k1_old
    return tu / mau


def solve_Av(Q, n, params):
    # Eq 13: Av* = n * Q * B2 / D
    return (n * Q * params["B2"]) / params["D"]


def solve_Theta(Q, n, params):
    # Eq 14: theta* = 2 * B1 / (rho * n * D * Q)
    return (2 * params["B1"]) / (params["rho"] * n * params["D"] * Q)


# =========================
# MAIN LOGIC
# =========================
if __name__ == "__main__":

    # 0. Sort Input theo C (Fix logic crashing)
    zip_all = sorted(zip(a, b, C), key=lambda x: x[2])
    a_sorted, b_sorted, C_sorted = map(list, zip(*zip_all))

    ts_list = compute_ts(a_sorted, b_sorted)
    CR_list = compute_CR(ts_list, a_sorted, b_sorted, C_sorted)
    m = len(ts_list)

    print(f"STEP 1: Init. ts={ts_list}, CR={CR_list}")

    best_solution_overall = None
    TC_prev_n = float("inf")  # Dùng để so sánh n vs n-1
    MAX_N = 15
    n = 1

    # STEP 5: Loop n
    while n <= MAX_N:
        print(f"\n{'=' * 60}\nSTEP 5: Checking n = {n}\n{'=' * 60}")

        TC_best_for_n = float("inf")
        solution_best_for_n = None

        # STEP 2: Loop j (crashing steps)
        for j in range(1, m + 1):
            ts = ts_list[j - 1]
            CR = CR_list[j - 1]

            # Khởi tạo STEP 2a
            Q = 200.0
            P = 900.0
            k1 = 4.0
            Av = params["Av0"]
            theta = params["theta0"]

            # 🛑 CỜ TRẠNG THÁI CHO STEP 3 (ACTIVE SET STRATEGY)
            fix_Av = False
            fix_theta = False

            # Vòng lặp "Go Back to Step 2" (Mô phỏng Flowchart)
            # Nó sẽ chạy cho đến khi thỏa mãn Constraints
            while True:
                # ==========================
                # STEP 2b-2e: Convergence Loop
                # ==========================
                for iter in range(50):
                    Q_old, P_old, k1_old = Q, P, k1

                    # 1. Tính Betas
                    # Lưu ý: Av và theta ở đây là giá trị "đang xét" (có thể là fix hoặc chưa)
                    b1_val, b2_val, b3_val, b4_val = b1(params["D"], n, params["A0"], params["CT"], params["pi"],
                                                        params["sigma"], ts, Q, P, params["tT"], k1, CR, Av), \
                        b2(params["D"], params["pi"], params["sigma"], ts, Q, P, params["tT"], k1, n), \
                        b3(params["hb"], params["sigma"], k1, ts, Q, P, params["hv"], n, params["D"], params["rho"],
                           theta), \
                        b4(ts, Q, P, n, params["hv"], params["tT"], k1, params)

                    # 2. Update P, Q, k1
                    P = solve_P(b4_val, params)
                    P = max(params["Pmin"], min(P, params["Pmax"]))

                    Q = solve_Q(b1_val, b2_val, b3_val)
                    Q = max(Q, 1.0)

                    k1 = solve_k1(k1, Q, P, n, ts, params["tT"], params)
                    k1 = max(0.01, min(k1, 20.0))

                    # 3. Update Av, Theta
                    # 🛑 QUAN TRỌNG: CHỈ TÍNH LẠI NẾU CHƯA BỊ KHÓA (FIX)
                    if not fix_Av:
                        Av = solve_Av(Q, n, params)
                        # Lưu ý: Av có thể > Av0 ở bước này, ta cứ để nó tính toán để tìm điểm hội tụ tự do

                    if not fix_theta:
                        theta = solve_Theta(Q, n, params)

                    # Check hội tụ
                    if max(abs(Q - Q_old) / Q, abs(P - P_old) / P) < 1e-5:
                        break

                # ==========================
                # STEP 3: Check Constraints & Decide
                # ==========================
                violation_found = False

                # Check Av
                if Av > params["Av0"] and not fix_Av:
                    # Case: Av unconstrained violates bound
                    print(f"    ⚠ Av violated ({Av:.2f} > {params['Av0']}). Fixing Av = Av0.")
                    Av = params["Av0"]
                    fix_Av = True  # Khóa lại
                    violation_found = True

                # Check Theta
                if theta > params["theta0"] and not fix_theta:
                    # Case: Theta unconstrained violates bound
                    print(f"    ⚠ Theta violated ({theta:.6f} > {params['theta0']}). Fixing Theta = Theta0.")
                    theta = params["theta0"]
                    fix_theta = True  # Khóa lại
                    violation_found = True

                # Flowchart Decision:
                if violation_found:
                    # "Go Back to Step 2" (với biến đã khóa)
                    continue
                else:
                    # Constraints thỏa mãn -> Sang Step 4
                    break

            # ==========================
            # STEP 4: Calculate TC
            # ==========================
            TC = compute_TC_total(Q, P, n, k1, Av, theta, params, ts, params["tT"], CR)
            print(f"  >> j={j} | TC=${TC:.2f} | Q={Q:.1f}, P={P:.1f}, Av={Av:.1f}, Th={theta:.6f}")

            if TC < TC_best_for_n:
                TC_best_for_n = TC
                solution_best_for_n = {
                    "n": n, "j": j, "TC": TC, "Q": Q, "P": P, "k1": k1, "Av": Av, "theta": theta, "ts": ts, "CR": CR
                }

        # ==========================
        # STEP 5 CHECK: Compare n with n-1
        # ==========================
        print(f"--> Best TC(n={n}) = {TC_best_for_n:.2f}")
        print(f"--> Best TC(n={n - 1}) = {TC_prev_n:.2f}")

        if TC_best_for_n < TC_prev_n:
            print("  ✓ Cost Decreased. Continue.")
            TC_prev_n = TC_best_for_n
            best_solution_overall = solution_best_for_n
            n += 1
        else:
            print("  ✗ Cost Increased. Stop.")
            break

    # ==========================
    # FINAL OUTPUT
    # ==========================
    if best_solution_overall:
        s = best_solution_overall
        print("\n" + "=" * 60)
        print("FINAL SOLUTION")
        print("=" * 60)
        print(f"n = {s['n']}")
        print(f"j = {s['j']} (ts={s['ts']:.4f})")
        print(f"Q = {s['Q']:.2f}")
        print(f"P = {s['P']:.2f}")
        print(f"k1 = {s['k1']:.4f}")
        print(f"Av = {s['Av']:.4f}")
        print(f"Theta = {s['theta']:.8f}")
        print(f"Total Cost = ${s['TC']:.2f}")
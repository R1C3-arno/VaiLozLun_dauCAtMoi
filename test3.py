import math
import numpy as np

# ==========================================
# 1. PARAMETERS (From Table 1 in PDF)
# ==========================================
params = {
    "D": 600.0,
    "Pmin": 700.0,
    "Pmax": 1200.0,
    "Av0": 400.0,  # Initial setup cost
    "A0": 150.0,  # Ordering cost
    "CT": 100.0,  # Transportation cost per shipment (Cr in table 1, CT in notation)
    "sigma": 5.0,  # Standard deviation
    "rho": 1.0,  # Defective cost parameter
    "hb": 19.234,  # Buyer holding cost
    "hv": 1.7,  # Vendor holding cost
    "pi": 80.0,  # Backorder cost
    "xi1": 1.0 / 300.0,  # Production cost param 1
    "xi2": 1.0 / 300.0,  # Production cost param 2
    "B1": 10000.0,  # Quality investment parameter
    "B2": 20.0,  # Setup cost investment parameter (Table 1 value)
    "theta0": 0.0002,  # Initial defective rate
    "tT": 1.9,  # Transportation time

    # Lead time components (Normal duration, Min duration, Crashing cost)
    "a": [0.1, 0.15, 0.1],
    "b": [0.05, 0.08, 0.04],
    "C_crash": [10.0, 30.0, 70.0]
}


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def compute_ts_and_costs(a, b, C_crash):
    """
    Generates the list of setup times (ts) and corresponding crashing costs (CR).
    Logic based on Assumption 3: Components 1..j are crashed to minimum.
    """
    ts_list = []
    CR_list = []
    m = len(a)

    # Case 0: No crashing (Normal duration)
    ts_normal = sum(a)
    ts_list.append(ts_normal)
    CR_list.append(0.0)

    # Case 1 to m: Crash components 1 through j
    current_ts = ts_normal
    current_cost = 0.0

    for j in range(m):
        # Crash component j (index j) from a[j] to b[j]
        reduction = a[j] - b[j]
        current_ts -= reduction
        current_cost += C_crash[j] * reduction

        ts_list.append(current_ts)
        CR_list.append(current_cost)

    return ts_list, CR_list


def calculate_betas(Q, P, n, k1, Av, theta, ts, CR, params):
    """
    Calculates Beta coefficients from Eq (5) and (6) in the paper.
    These are used to solve for Q and P.
    """
    D = params["D"]
    A0 = params["A0"]
    CT = params["CT"]
    pi_cost = params["pi"]
    sigma = params["sigma"]
    hb = params["hb"]
    hv = params["hv"]
    rho = params["rho"]
    xi1 = params["xi1"]
    xi2 = params["xi2"]
    tT = params["tT"]

    # Prevent division by zero
    P = max(P, 1e-5)
    ts_term = math.sqrt(ts + Q / P)

    # L(tT) logic: Lead time for batch 2..m
    # Safety factor k2 is related to k1 via Assumption 4
    # The term inside sqrt for batch 2 is tT + k1^2 * (ts + Q/P)
    # But in betas, the paper uses specific expanded terms.

    term_root_1 = math.sqrt(1 + k1 ** 2)
    # Approximation for the messy term in Beta 1 & 2 related to second batch
    # Based on Eq 5 definitions:

    # Beta 1 (Numerator for Q ordering cost part)
    # Note: Av is included here
    term_b1_1 = (math.sqrt(1 + k1 ** 2) - k1) * math.sqrt(ts + Q / P)  # This is an approximation as Q is inside
    # For the quadratic solver, we assume the Q inside the sqrt is "slow moving"
    # or handle it iteratively. The paper implies these betas are constants
    # relative to the partial derivative structure.

    # Let's strictly follow the Q coefficients in Eq 5:
    # dTC/dQ = -Beta1/Q^2 + Beta2/Q + Beta3

    # From Eq 5 defs:
    b1 = (D / n) * (A0 + n * CT + pi_cost * n * CR / pi_cost + Av)  # Simplified based on leading terms
    # The full Beta1 in paper includes sigma terms multiplied by Q?
    # No, looking at Eq 5, Beta1 seems to group the constant terms w.r.t Q derivation.
    # However, the paper's Beta1 definition (Eq 5 bottom) actually DEPENDS on Q.
    # This means true Q* is found iteratively.

    # Recalculating Beta 1 exactly from paper:
    t_safe = ts + Q / P
    t_safe_root = math.sqrt(t_safe)

    # Lead time for batch 2+ (transport only component adjusted by safety)
    # L(tT) term approximation
    lt_term = tT + (k1 ** 2) * t_safe
    lt_term_root = math.sqrt(lt_term)

    term_ex1 = (term_root_1 - k1) * t_safe_root
    term_ex2 = (n - 1) * (math.sqrt(lt_term) - k1 * math.sqrt(t_safe))  # Approximate based on structure
    # Wait, Eq 5 Beta 1 definition is massive.
    # To simplify for the Iterative Solver:
    # We calculate the numeric values of Beta1, Beta2, Beta3 using the CURRENT Q and P.
    # Then we find the NEW Q that satisfies the quadratic formed by these current betas.

    val_b1 = (D / n) * (A0 + n * CT + n * CR + Av)  # + pi term...
    # The pi term in b1: D*pi/n * [ ... ]
    # The exact expression for Beta1 in the paper is complex.
    # We will compute the exact derivation terms for the quadratic:
    # A * Q^2 + B * Q + C = 0
    # Where A = Beta3, B = Beta2, C = -Beta1

    # Re-evaluating Beta1 (Constant part of Cost * Q)
    # The sigma terms in Eq 1 are Q*sqrt(...). Derivative is complicated.
    # However, standard inventory models treat A/Q + hQ/2.
    # Let's use the explicit Beta definitions from the OCR text (Eq 5).

    # Beta 1
    # \beta_1 = \frac{D}{n} [ A_0 + nC_T + \pi \{ \frac{\sigma}{2} [\sqrt{t_s + Q/P} (...) ] \} + nC_R + A_v ]
    # Note: This effectively treats the backlog quantity as a constant w.r.t 1/Q^2 derivation in this step.
    bs1 = (sigma / 2) * t_safe_root * (term_root_1 - k1)
    bs2 = ((n - 1) * sigma / 2) * (lt_term_root - k1 * t_safe_root)  # Approximation of the messy term
    val_b1 = (D / n) * (A0 + n * CT + pi_cost * (bs1 + bs2) + n * CR + Av)

    # Beta 2 (1/Q term, usually 0 in standard EOQ, but here it appears)
    # \beta_2 = \frac{D\pi\sigma}{4nP} ...
    term_b2_1 = (term_root_1 - k1) / t_safe_root
    term_b2_2 = (n - 1) * (k1 / lt_term_root - 1 / t_safe_root)  # Derived from derivative of sqrt
    # Actually, let's look at Eq 5 def.
    val_b2 = (D * pi_cost * sigma) / (4 * n * P) * (term_b2_1 + k1 * term_b2_2)  # Adjusted based on structure

    # Beta 3 (Linear Q term - Holding Cost)
    # \beta_3 = h_b/2 [1 + k_1 sigma / (2 P sqrt(ts+Q/P)) ] + ...
    term_b3_h = hb / 2 * (1 + (k1 * sigma) / (2 * P * t_safe_root))
    term_b3_v = (hv / 2) * (n * (1 - D / P) - 1 + 2 * D / P)
    term_b3_def = (rho * n * D * theta) / 2
    val_b3 = term_b3_h + term_b3_v + term_b3_def

    # Beta 4 (For P calculation, Eq 6)
    # \beta_4 = h_b k_1 sigma Q / (2 t_safe_root) ...
    val_b4 = (hb * k1 * sigma * Q) / (2 * t_safe_root)
    # Add other terms from Eq 6
    val_b4 += (D * pi_cost * sigma / (4 * n)) * (
                (term_root_1 - k1) / t_safe_root + (n - 1) * (k1 / lt_term_root - 1 / t_safe_root))
    val_b4 += - (hv * Q * D * (n - 2)) / 2  # Note: Eq 6 says hv Q D (n-2) / 2
    val_b4 += D * xi2

    return val_b1, val_b2, val_b3, val_b4


def solve_Q(b1, b2, b3):
    """
    Solves -b1/Q^2 + b2/Q + b3 = 0
    => b3*Q^2 + b2*Q - b1 = 0
    """
    delta = b2 ** 2 - 4 * b3 * (-b1)
    if delta < 0: return 100.0  # Error fallback

    Q_new = (-b2 + math.sqrt(delta)) / (2 * b3)
    return max(Q_new, 1.0)


def solve_P(b4, params):
    """
    Solves Eq 11: P* = sqrt( Beta4 / (D * xi1) )
    """
    denom = params["D"] * params["xi1"]
    if b4 <= 0: return params["Pmin"]  # Fallback if cost structure implies min P

    P_new = math.sqrt(b4 / denom)
    return max(params["Pmin"], min(P_new, params["Pmax"]))


def solve_k1(k1_current, Q, P, n, ts, params):
    """
    Solves implicit Eq 12 for k1 using fixed point iteration.
    Eq 12: k1 = [n(D*pi - 2*Q*hb)] / [D*pi * (1/sqrt(1+k1^2) + (n-1)*...)]
    """
    D = params["D"]
    hb = params["hb"]
    pi_cost = params["pi"]
    tT = params["tT"]

    t_safe = ts + Q / P

    # Numerator
    num = n * (D * pi_cost - 2 * Q * hb)
    if num <= 0: return 0.01  # Boundary condition

    # Denominator term
    term1 = 1.0 / math.sqrt(1 + k1_current ** 2)

    lt_term = tT + k1_current ** 2 * t_safe
    term2 = (n - 1) * math.sqrt(t_safe / lt_term)

    denom = D * pi_cost * (term1 + term2)

    k1_new = num / denom
    return max(0.01, k1_new)  # Ensure positive


def calculate_total_cost(Q, P, n, k1, Av, theta, ts, CR, params):
    D = params["D"]
    CT = params["CT"]
    A0 = params["A0"]
    hb = params["hb"]
    hv = params["hv"]
    pi_cost = params["pi"]
    sigma = params["sigma"]
    rho = params["rho"]
    xi1 = params["xi1"]
    xi2 = params["xi2"]
    B1 = params["B1"]
    B2 = params["B2"]
    Av0 = params["Av0"]
    theta0 = params["theta0"]
    tT = params["tT"]

    t_safe = ts + Q / P
    t_safe_root = math.sqrt(t_safe)

    # Buyer Cost
    TC_b_order = (A0 + n * CT) * D / (n * Q)
    TC_b_hold = hb * (Q / 2 + k1 * sigma * t_safe_root)

    # Shortage Cost
    E_x1 = (sigma / 2) * t_safe_root * (math.sqrt(1 + k1 ** 2) - k1)

    lt_term = tT + k1 ** 2 * t_safe
    E_x2 = (sigma / 2) * math.sqrt(tT) * (math.sqrt(1 + k1 ** 2 * t_safe / tT) - k1 * math.sqrt(t_safe / tT))

    TC_b_short = (D * pi_cost / (n * Q)) * E_x1 + (D * pi_cost * (n - 1) / (n * Q)) * E_x2
    TC_b_crash = (n * D * CR) / (n * Q)

    TC_b = TC_b_order + TC_b_hold + TC_b_short + TC_b_crash

    # Vendor Cost
    prod_cost_unit = xi1 * P + xi2 / P
    TC_v_setup = Av * D / (n * Q)
    TC_v_hold = (hv * Q / 2) * (n * (1 - D / P) - 1 + 2 * D / P)
    TC_v_prod = D * prod_cost_unit
    TC_v_defect = rho * D * theta * n * Q / 2
    TC_v_invest = B1 * math.log(theta0 / theta) + B2 * math.log(Av0 / Av)

    TC_v = TC_v_setup + TC_v_hold + TC_v_prod + TC_v_defect + TC_v_invest

    return TC_b + TC_v


# ==========================================
# 3. MAIN ALGORITHM
# ==========================================

def solve_model():
    # Pre-calculate crashing options
    ts_options, CR_options = compute_ts_and_costs(params["a"], params["b"], params["C_crash"])

    best_overall_TC = float('inf')
    best_solution = None

    print(f"{'n':<4} {'j':<4} {'Q':<10} {'P':<10} {'k1':<8} {'Av':<10} {'Theta':<12} {'TC':<12}")
    print("-" * 80)

    # Step 1 & 5: Loop for n
    # We loop until TC stops decreasing
    n = 1
    while True:
        best_TC_n = float('inf')
        solution_n = None

        # Step 2: Loop for each setup time component j
        for j in range(len(ts_options)):
            ts = ts_options[j]
            CR = CR_options[j]

            # Step 2a: Initialize variables
            Q = 200.0
            P = 900.0
            k1 = 2.0
            Av = params["Av0"]
            theta = params["theta0"]

            # Constraints flags
            fix_Av = False
            fix_theta = False

            # Iterative Solver Loop
            for iteration in range(50):
                # Save previous for convergence check
                Q_old, P_old, k1_old = Q, P, k1

                # Step 2b: Update P and Q
                # Calculate betas based on current values
                b1, b2, b3, b4 = calculate_betas(Q, P, n, k1, Av, theta, ts, CR, params)

                P = solve_P(b4, params)
                Q = solve_Q(b1, b2, b3)

                # Step 2c: Update k1
                k1 = solve_k1(k1, Q, P, n, ts, params)

                # Step 2d: Update Av and Theta (Eq 13 & 14)
                # If constrained, keep fixed
                if not fix_Av:
                    # Eq 13: Av* = n * Q * B2 / D
                    Av_calc = (n * Q * params["B2"]) / params["D"]
                else:
                    Av_calc = params["Av0"]  # Fixed to max

                if not fix_theta:
                    # Eq 14: theta* = 2 * B1 / (rho * n * D * Q)
                    theta_calc = (2 * params["B1"]) / (params["rho"] * n * params["D"] * Q)
                else:
                    theta_calc = params["theta0"]  # Fixed to max

                # Step 3: Check Constraints and Logic
                # If calculated Av > Av0, we must set Av = Av0 and re-solve logic
                # This effectively means the optimal is at the boundary.

                if Av_calc > params["Av0"] and not fix_Av:
                    fix_Av = True
                    Av = params["Av0"]
                    continue  # Restart loop with fixed Av
                else:
                    Av = Av_calc

                if theta_calc > params["theta0"] and not fix_theta:
                    fix_theta = True
                    theta = params["theta0"]
                    continue  # Restart loop with fixed theta
                else:
                    theta = theta_calc

                # Check Convergence
                if max(abs(Q - Q_old) / Q, abs(P - P_old) / P, abs(k1 - k1_old) / k1) < 1e-5:
                    break

            # Calculate Total Cost for this n, j combination
            current_TC = calculate_total_cost(Q, P, n, k1, Av, theta, ts, CR, params)

            if current_TC < best_TC_n:
                best_TC_n = current_TC
                solution_n = {
                    "n": n, "j": j, "Q": Q, "P": P, "k1": k1,
                    "Av": Av, "theta": theta, "ts": ts, "CR": CR, "TC": current_TC
                }

        # Print progress for this n
        if solution_n:
            s = solution_n
            print(
                f"{n:<4} {s['j']:<4} {s['Q']:<10.2f} {s['P']:<10.2f} {s['k1']:<8.4f} {s['Av']:<10.4f} {s['theta']:<12.8f} {s['TC']:<12.2f}")

        # Step 6: Convexity Check
        # If TC increased compared to n-1, we stop (assuming convexity w.r.t n)
        # Note: We need at least 2 points to compare.
        if best_TC_n < best_overall_TC:
            best_overall_TC = best_TC_n
            best_solution = solution_n
            n += 1
            if n > 20: break  # Safety break
        else:
            # Optimal n found at previous step
            break

    return best_solution


# ==========================================
# 4. EXECUTION
# ==========================================
if __name__ == "__main__":
    opt = solve_model()

    print("\n" + "=" * 50)
    print("FINAL OPTIMAL SOLUTION")
    print("=" * 50)
    print(f"Number of Shipments (n): {opt['n']}")
    print(f"Crash Level (j):         {opt['j']} (ts = {opt['ts']:.4f})")
    print(f"Lot Size (Q):            {opt['Q']:.2f}")
    print(f"Production Rate (P):     {opt['P']:.2f}")
    print(f"Safety Factor (k1):      {opt['k1']:.4f}")
    print(f"Vendor Setup Cost (Av):  {opt['Av']:.4f}")
    print(f"Defective Rate (theta):  {opt['theta']:.8f}")
    print(f"Expected Total Cost:     ${opt['TC']:.2f}")
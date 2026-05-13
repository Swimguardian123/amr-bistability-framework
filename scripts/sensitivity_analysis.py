"""
COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS (v3 — Final Corrected)
===================================================================

PURPOSE
-------
Determines how the transcritical point I*2 depends on model parameters.
Includes THREE analyses:
  1. ONE-AT-A-TIME: Varies each parameter individually (main effects on I*2).
  2. TWO-PARAMETER INTERACTION: Varies correlated pairs jointly:
       a) b vs b_R (pharmacological correlation)
       b) eta vs mu (structural bistability mechanism)
  3. BASELINE CROSS-CHECK: Verifies baseline I*2 matches corrected compendium v4.

CORRECTED BIFURCATION CONTEXT
-----------------------------
The full 3D model exhibits SEQUENTIAL regime transitions:
  • I*1 ≈ 4.7:  Extinction becomes stable (monostable → bistable TYPE 1)
  • I*2 ≈ 11.6: Interior merges with p=1 boundary (transcritical, TYPE 1 → TYPE 2)
  • I*3 ≈ 34.5: p=1 boundary disappears via saddle-node (bistable → monostable)

This script computes I*2 — the transcritical point where interior coexistence
ceases to exist. The interior equilibrium remains STABLE up to I*2; it simply
merges with the p=1 boundary and vanishes. This is NOT a stability-loss
bifurcation (the eigenvalues stay negative), but a structural disappearance.

METHODOLOGY NOTES
-----------------
One-at-a-time sensitivity captures MAIN EFFECTS but misses INTERACTIONS.
For example, b (kill rate susceptible) and b_R (kill rate resistant) are
likely correlated in practice (both depend on the same pharmacodynamic assay).
The two-parameter heatmaps test whether joint uncertainty produces wider I*2
ranges than individual variation suggests.

The eta vs mu interaction is structurally critical: bistability requires
3D + endogenous drug feedback (eta > 0). If eta → 0 or mu → ∞, the drug
dynamics decouple and bistability collapses to monostability.

POST-HOC VALIDATION FRAMEWORK
-----------------------------
Sensitivity analysis, not parameter estimation. All parameters fixed as priors.
The sensitivity index S = (ΔI*2/I*2_base)/(Δp/p_base) bounds PRACTICAL
IDENTIFIABILITY: large |S| → wide uncertainty in transcritical point given
parameter uncertainty ranges.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================================
# BASE PARAMETERS — Confirmed bistable set (Compendium v4, Sec 3.1)
# ============================================================
BASE_PARAMS = {
    'r_S': 1.0, 'r_R': 0.93, 'K': 1e9, 'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0, 'n': 3.0, 'c_R': 0.04,
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12
}

def hill(C, MIC, n):
    if C <= 0: return 0.0
    return C**n / (C**n + MIC**n)

def growth_rates(N, p, C, p_dict):
    r_S = p_dict['r_S']; r_R = p_dict['r_R']; c_R = p_dict['c_R']; K = p_dict['K']
    b = p_dict['b']; b_R = p_dict['b_R']
    MIC_S = p_dict['MIC_S']; MIC_R = p_dict['MIC_R']; n = p_dict['n']
    f_S = hill(C, MIC_S, n)
    f_R = hill(C, MIC_R, n)
    g_S = r_S * (1 - N / K) - b * f_S
    g_R = r_R * (1 - N / K) - c_R - b_R * f_R
    g_bar = (1 - p) * g_S + p * g_R
    return g_S, g_R, g_bar

def residuals(state, I_val, p_dict):
    N, p, C = state
    N = max(N, 0.0)
    p = np.clip(p, 0.0, 1.0)
    g_S, g_R, g_bar = growth_rates(N, p, C, p_dict)
    dN = N * g_bar
    dp = p * (1 - p) * (g_R - g_S + p_dict['gamma'] * N)
    dC = I_val - p_dict['mu'] * C - p_dict['eta'] * N * p * C
    return np.array([dN, dp, dC])

def jacobian(state, I_val, p_dict):
    N, p, C = state
    r_S = p_dict['r_S']; r_R = p_dict['r_R']; c_R = p_dict['c_R']; K = p_dict['K']
    b = p_dict['b']; b_R = p_dict['b_R']
    MIC_S = p_dict['MIC_S']; MIC_R = p_dict['MIC_R']; n = p_dict['n']
    mu = p_dict['mu']; eta = p_dict['eta']; gamma = p_dict['gamma']
    g_S, g_R, g_bar = growth_rates(N, p, C, p_dict)
    if C > 0:
        df_S_dC = n * (MIC_S**n) * C**(n-1) / (C**n + MIC_S**n)**2
        df_R_dC = n * (MIC_R**n) * C**(n-1) / (C**n + MIC_R**n)**2
    else:
        df_S_dC = df_R_dC = 0.0
    dgS_dN = -r_S / K
    dgR_dN = -r_R / K
    dgS_dC = -b * df_S_dC
    dgR_dC = -b_R * df_R_dC
    Delta_g = g_R - g_S
    J11 = g_bar + N * ((1 - p) * dgS_dN + p * dgR_dN)
    J12 = N * Delta_g
    J13 = N * ((1 - p) * dgS_dC + p * dgR_dC)
    J21 = p * (1 - p) * (dgR_dN - dgS_dN + gamma)
    J22 = (1 - 2*p) * (Delta_g + gamma * N)
    J23 = p * (1 - p) * (dgR_dC - dgS_dC)
    J31 = -eta * p * C
    J32 = -eta * N * C
    J33 = -mu - eta * N * p
    return np.array([[J11, J12, J13],
                     [J21, J22, J23],
                     [J31, J32, J33]])

def is_stable(state, I_val, p_dict, tol=-1e-10):
    J = jacobian(state, I_val, p_dict)
    eigs = eigvals(J)
    return np.all(eigs.real < tol)

def has_stable_coexistence(I_val, p_dict, p_tol=1e-4, res_tol=1e-5, prev_eq=None):
    """
    Check if stable interior coexistence exists at dosing rate I_val.
    Uses QSSA-aware initial guesses + optional continuation from prev_eq.
    
    RETURNS: (found_bool, eq_state_or_None)
      found_bool: True if stable interior equilibrium exists
      eq_state:   The equilibrium state [N, p, C] for continuation, or None
    """
    K = p_dict['K']
    seeds = []
    
    # CONTINUATION: use previous equilibrium as primary seed
    if prev_eq is not None:
        seeds.append(list(prev_eq))
        seeds.append([prev_eq[0]*0.99, prev_eq[1], prev_eq[2]])
        seeds.append([prev_eq[0]*1.01, prev_eq[1]*0.99, prev_eq[2]])
    
    for N_frac in [0.5, 0.7, 0.85, 0.95, 0.99]:
        N_guess = K * N_frac
        for p_frac in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
            C_guess = I_val / (p_dict['mu'] + p_dict['eta'] * N_guess * p_frac + 1e-12)
            seeds.append([N_guess, p_frac, C_guess])
    for N_frac in [0.8, 0.95]:
        N_guess = K * N_frac
        for p_frac in [0.25, 0.5, 0.75]:
            C_guess = I_val / (p_dict['mu'] + p_dict['eta'] * N_guess * p_frac + 1e-12)
            seeds.append([N_guess, p_frac, C_guess * 0.5])
    
    best_eq = None
    best_res = 1e10
    
    for seed in seeds:
        try:
            sol, info, ier, mesg = fsolve(
                residuals, seed, args=(I_val, p_dict),
                xtol=1e-12, maxfev=2000, full_output=True
            )
            if ier != 1: continue
            N, p, C = float(sol[0]), float(np.clip(sol[1], 0, 1)), float(sol[2])
            if N <= 0 or C <= 0: continue
            res = residuals(sol, I_val, p_dict)
            scales = np.array([max(abs(N), 1e5), 1.0, max(abs(C), 1.0)])
            norm_res = np.linalg.norm(res / scales)
            if norm_res > res_tol: continue
            if p <= p_tol or p >= 1 - p_tol: continue
            
            if norm_res < best_res:
                best_res = norm_res
                best_eq = np.array([N, p, C])
            
            if is_stable(sol, I_val, p_dict):
                return True, np.array([N, p, C])
        except: continue
    
    return False, best_eq

def find_I_transcritical(p_dict, I_min=0.5, I_max=30.0, n_coarse=40, bisection_tol=0.05, max_extend=3):
    """
    Find I*2 — the transcritical point where interior coexistence merges
    with the p=1 boundary and ceases to exist.
    Uses CONTINUATION: tracks equilibrium from low I to high I.
    """
    I_vals = np.linspace(I_min, I_max, n_coarse)
    flags = []
    prev_eq = None
    
    for I_val in I_vals:
        found, eq = has_stable_coexistence(I_val, p_dict, prev_eq=prev_eq)
        flags.append(found)
        if eq is not None:
            prev_eq = eq
    
    if not any(flags):
        I_low_test = np.linspace(0.1, I_min, 10)
        for I_val in I_low_test:
            found, eq = has_stable_coexistence(I_val, p_dict)
            if found:
                return I_val, 'success'
        return None, 'no_coexistence'
    
    if flags[-1]:
        for ext in range(max_extend):
            I_new_max = I_max * 2
            I_test = np.linspace(I_max, I_new_max, 20)
            new_flags = []
            for I in I_test:
                found, eq = has_stable_coexistence(I, p_dict, prev_eq=prev_eq)
                new_flags.append(found)
                if eq is not None:
                    prev_eq = eq
            I_vals = np.concatenate([I_vals, I_test])
            flags = flags + new_flags
            I_max = I_new_max
            if not new_flags[-1]:
                break
        if flags[-1]:
            return I_max, 'extends_beyond_range'
    
    valid_idx = [i for i, f in enumerate(flags) if f]
    last_valid = valid_idx[-1]
    if last_valid >= len(I_vals) - 1:
        return I_vals[last_valid], 'extends_beyond_range'
    
    I_low = I_vals[last_valid]
    I_high = I_vals[last_valid + 1]
    for _ in range(25):
        if I_high - I_low < bisection_tol:
            break
        I_mid = (I_low + I_high) / 2.0
        found, eq = has_stable_coexistence(I_mid, p_dict, prev_eq=prev_eq)
        if found:
            I_low = I_mid
            if eq is not None:
                prev_eq = eq
        else:
            I_high = I_mid
    
    return I_low, 'success'

def find_I_extinction_stable(p_dict, I_min=0.1, I_max=10.0, n_test=100):
    """
    Find I*1 — the threshold where extinction becomes stable.
    """
    I_vals = np.linspace(I_min, I_max, n_test)
    for I_val in I_vals:
        C = I_val / p_dict['mu']
        gS_0 = p_dict['r_S'] - p_dict['b'] * hill(C, p_dict['MIC_S'], p_dict['n'])
        gR_0 = p_dict['r_R'] - p_dict['c_R'] - p_dict['b_R'] * hill(C, p_dict['MIC_R'], p_dict['n'])
        if gS_0 < 0 and gR_0 < 0:
            return I_val
    return None

SENSITIVITY_CONFIG = [
    {'name': 'r_S',     'baseline': 1.0,   'range': [0.5, 0.75, 1.0, 1.25, 1.5],       'log': False, 'unit': '1/hr'},
    {'name': 'r_R',     'baseline': 0.93,  'range': [0.5, 0.75, 0.93, 1.0, 1.25],      'log': False, 'unit': '1/hr'},
    {'name': 'b',       'baseline': 2.0,   'range': [1.0, 1.5, 2.0, 2.5, 3.0],         'log': False, 'unit': '1/hr'},
    {'name': 'b_R',     'baseline': 1.5,   'range': [0.5, 1.0, 1.5, 2.0, 2.5],         'log': False, 'unit': '1/hr'},
    {'name': 'MIC_S',   'baseline': 2.0,   'range': [1.0, 2.0, 4.0],                   'log': True,  'unit': 'mg/L'},
    {'name': 'MIC_R',   'baseline': 4.0,   'range': [2.0, 4.0, 8.0, 16.0, 32.0],       'log': True,  'unit': 'mg/L'},
    {'name': 'n',       'baseline': 3.0,   'range': [1.0, 2.0, 3.0, 4.0, 5.0],         'log': False, 'unit': ''},
    {'name': 'c_R',     'baseline': 0.04,  'range': [0.0, 0.02, 0.04, 0.08, 0.12, 0.16],'log': False, 'unit': '1/hr'},
    {'name': 'K',       'baseline': 1e9,   'range': [1e8, 5e8, 1e9, 5e9, 1e10],        'log': True,  'unit': 'cells'},
    {'name': 'mu',      'baseline': 1.0,   'range': [0.5, 1.0, 2.0],                   'log': False, 'unit': '1/hr'},
    {'name': 'eta',     'baseline': 2e-8,  'range': [1e-12, 1e-10, 1e-9, 1e-8, 1e-7],  'log': True,  'unit': 'L/(cell*hr)'},
]


# =============================================================================
# MAIN EXECUTION
# =============================================================================

print("="*80)
print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS (v3 — Final Corrected)")
print("="*80)

print("\n[1/3] Computing baseline regime boundaries...")
t0 = time.time()
baseline_I2, baseline_status = find_I_transcritical(BASE_PARAMS, I_min=0.5, I_max=25.0)
t_baseline = time.time() - t0

baseline_I1 = find_I_extinction_stable(BASE_PARAMS)

if baseline_I2 is None:
    print(f"  BASELINE I*2: No coexistence found ({baseline_status})")
else:
    print(f"  BASELINE I*2 (transcritical): {baseline_I2:.4f} mg/L/hr  [{baseline_status}]  (took {t_baseline:.1f}s)")
    print(f"    → Interior coexistence exists for I < {baseline_I2:.2f}")
    print(f"    → TYPE 1 bistability for I*1 < I < I*2")

if baseline_I1 is not None:
    print(f"  BASELINE I*1 (extinction stable): {baseline_I1:.4f} mg/L/hr")
    print(f"    → Extinction becomes stable at I > {baseline_I1:.2f}")
    if baseline_I2 is not None:
        print(f"  BISTABLE TYPE 1 RANGE: I ∈ [{baseline_I1:.2f}, {baseline_I2:.2f}]")
else:
    print(f"  BASELINE I*1: Not found in scanned range")

if baseline_I2 is not None:
    expected_I2 = 11.6
    discrepancy = abs(baseline_I2 - expected_I2) / expected_I2 * 100
    print(f"  CROSS-CHECK: Corrected compendium v4 predicts I*2 ≈ {expected_I2:.1f}")
    print(f"  Discrepancy: {discrepancy:.1f}%")
    if discrepancy > 15:
        print("  WARNING: Discrepancy >15%. Continuation may have lost the branch.")

print(f"\n[2/3] Running one-at-a-time sensitivity sweeps ({len(SENSITIVITY_CONFIG)} parameters)...")
print("-"*80)

results = {}
for cfg in SENSITIVITY_CONFIG:
    name = cfg["name"]
    values = cfg["range"]
    I2_vals = []
    statuses = []
    print(f"\nSweeping {name:8s} (baseline={cfg['baseline']})...")
    for val in values:
        p_dict = BASE_PARAMS.copy()
        p_dict[name] = val
        t0 = time.time()
        I2, status = find_I_transcritical(p_dict, I_min=0.5, I_max=25.0)
        dt = time.time() - t0
        I2_vals.append(I2)
        statuses.append(status)
        if I2 is None:
            print(f"  {name}={val:12.4e}  ->  I*2=N/A  [{status:22s}]  ({dt:.1f}s)")
        else:
            if baseline_I2 is not None and baseline_I2 > 0 and cfg["baseline"] != 0:
                dI = (I2 - baseline_I2) / baseline_I2
                dp = (val - cfg["baseline"]) / cfg["baseline"]
                if abs(dp) > 1e-10:
                    sens_idx = dI / dp
                else:
                    sens_idx = np.nan
            else:
                sens_idx = np.nan
            print(f"  {name}={val:12.4e}  ->  I*2={I2:8.3f}  [{status:10s}]  S={sens_idx:+.2f}  ({dt:.1f}s)")
    results[name] = {
        "values": values, "I2": I2_vals, "statuses": statuses,
        "log": cfg["log"], "unit": cfg["unit"], "baseline": cfg["baseline"]
    }

print("\n[3/3] Running two-parameter interaction analyses...")
print("-"*80)

# Analysis 3a: b vs b_R
print("\n  [3a] b vs b_R interaction...")
b_range = [1.0, 1.5, 2.0, 2.5, 3.0]
bR_range = [0.5, 1.0, 1.5, 2.0, 2.5]
interaction_matrix_bbR = np.zeros((len(b_range), len(bR_range)))
interaction_status_bbR = np.empty((len(b_range), len(bR_range)), dtype=object)
for i, b_val in enumerate(b_range):
    for j, bR_val in enumerate(bR_range):
        p_dict = BASE_PARAMS.copy()
        p_dict["b"] = b_val
        p_dict["b_R"] = bR_val
        crit, status = find_I_transcritical(p_dict, I_min=0.5, I_max=25.0)
        interaction_matrix_bbR[i, j] = crit if crit is not None else np.nan
        interaction_status_bbR[i, j] = status
        crit_str = f"{crit:.2f}" if crit is not None else "N/A"
        print(f"    b={b_val:.1f}, b_R={bR_val:.1f} -> I*2={crit_str} [{status}]")

# Analysis 3b: eta vs mu (FIXED: centered around baseline, NaN = lightgray)
print("\n  [3b] eta vs mu interaction (structural)...")
eta_range = [1e-10, 1e-9, 2e-8, 1e-7, 1e-6]  # Centered around baseline 2e-8
mu_range = [0.5, 1.0, 2.0, 5.0, 10.0]
interaction_matrix_etamu = np.zeros((len(eta_range), len(mu_range)))
interaction_status_etamu = np.empty((len(eta_range), len(mu_range)), dtype=object)
for i, eta_val in enumerate(eta_range):
    for j, mu_val in enumerate(mu_range):
        p_dict = BASE_PARAMS.copy()
        p_dict["eta"] = eta_val
        p_dict["mu"] = mu_val
        crit, status = find_I_transcritical(p_dict, I_min=0.5, I_max=50.0)
        interaction_matrix_etamu[i, j] = crit if crit is not None else np.nan
        interaction_status_etamu[i, j] = status
        crit_str = f"{crit:.2f}" if crit is not None else "N/A"
        print(f"    eta={eta_val:.0e}, mu={mu_val:.1f} -> I*2={crit_str} [{status}]")

# PLOTTING
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
n_params = len(results)
for idx, (name, data) in enumerate(results.items()):
    row = idx // 4
    col = idx % 4
    ax = fig.add_subplot(gs[row, col])
    vals = np.array(data["values"], dtype=float)
    I2s = np.array(data["I2"], dtype=float)
    statuses = data["statuses"]
    x_s = []; y_s = []
    x_e = []; y_e = []
    x_f = []; y_f = []
    for v, t, s in zip(vals, I2s, statuses):
        if s == "success": x_s.append(v); y_s.append(t)
        elif s == "extends_beyond_range": x_e.append(v); y_e.append(t)
        else: x_f.append(v); y_f.append(t)
    vm = np.array([s == "success" or s == "extends_beyond_range" for s in statuses])
    if np.any(vm):
        if data["log"]: ax.semilogx(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
        else: ax.plot(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
    if x_s: ax.scatter(x_s, y_s, c="blue", s=40, zorder=3, label="Converged")
    if x_e: ax.scatter(x_e, y_e, c="green", s=40, marker="^", zorder=3, label="Extends")
    if x_f: ax.scatter(x_f, y_f, c="red", s=40, marker="x", zorder=3, label="No coexistence")
    if baseline_I2 is not None: ax.axhline(y=baseline_I2, color="crimson", linestyle="--", alpha=0.6, linewidth=1, label="Baseline I*2")
    ax.axvline(x=data["baseline"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel(f"{name} ({data['unit']})", fontsize=9)
    ax.set_ylabel("I*2 (transcritical)", fontsize=9)
    ax.set_title(f"{name}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

# Heatmap 1: b vs b_R
ax = fig.add_subplot(gs[2, 2])
im1 = ax.imshow(interaction_matrix_bbR, cmap="RdYlGn_r", aspect="auto", 
                vmin=np.nanmin(interaction_matrix_bbR), vmax=np.nanmax(interaction_matrix_bbR))
ax.set_xticks(range(len(bR_range)))
ax.set_xticklabels([f"{v:.1f}" for v in bR_range], fontsize=8)
ax.set_yticks(range(len(b_range)))
ax.set_yticklabels([f"{v:.1f}" for v in b_range], fontsize=8)
ax.set_xlabel("b_R", fontsize=10)
ax.set_ylabel("b", fontsize=10)
ax.set_title("Interaction: b vs b_R\n(pharmacological)", fontsize=10, fontweight="bold")
cbar1 = plt.colorbar(im1, ax=ax, shrink=0.8)
cbar1.set_label("I*2", fontsize=9)
for i in range(len(b_range)):
    for j in range(len(bR_range)):
        if not np.isnan(interaction_matrix_bbR[i, j]):
            ax.text(j, i, f"{interaction_matrix_bbR[i,j]:.1f}", ha="center", va="center", fontsize=7, color="black")

# Heatmap 2: eta vs mu (FIXED: NaN = lightgray, centered range)
ax = fig.add_subplot(gs[2, 3])
cmap = plt.cm.RdYlGn_r.copy()
cmap.set_bad('lightgray')  # NaN cells = no bistability region (structurally correct)
im2 = ax.imshow(interaction_matrix_etamu, cmap=cmap, aspect="auto",
                vmin=np.nanmin(interaction_matrix_etamu), vmax=np.nanmax(interaction_matrix_etamu))
ax.set_xticks(range(len(mu_range)))
ax.set_xticklabels([f"{v:.1f}" for v in mu_range], fontsize=8)
ax.set_yticks(range(len(eta_range)))
ax.set_yticklabels([f"{v:.0e}" for v in eta_range], fontsize=7)
ax.set_xlabel("mu", fontsize=10)
ax.set_ylabel("eta", fontsize=10)
ax.set_title("Interaction: eta vs mu\n(structural mechanism)", fontsize=10, fontweight="bold")
cbar2 = plt.colorbar(im2, ax=ax, shrink=0.8)
cbar2.set_label("I*2", fontsize=9)
for i in range(len(eta_range)):
    for j in range(len(mu_range)):
        if not np.isnan(interaction_matrix_etamu[i, j]):
            ax.text(j, i, f"{interaction_matrix_etamu[i,j]:.1f}", ha="center", va="center", fontsize=7, color="black")

plt.suptitle("Parameter Sensitivity: Main Effects + Two-Parameter Interactions (v3 Final)", 
             fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("comprehensive_sensitivity_v3_final.png", dpi=300, bbox_inches="tight")
plt.show()

# SUMMARY TABLE
print("\n" + "="*80)
print("SUMMARY TABLE — Transcritical Point I*2 Sensitivity")
print("="*80)
print("Param    Value         I*2       Status               SensIdx")
print("-"*80)
for name, data in results.items():
    for val, I2, status in zip(data["values"], data["I2"], data["statuses"]):
        I2_str = f"{I2:.3f}" if I2 is not None else "N/A"
        if baseline_I2 is not None and baseline_I2 > 0 and data["baseline"] != 0 and I2 is not None:
            dI = (I2 - baseline_I2) / baseline_I2
            dp = (val - data["baseline"]) / data["baseline"]
            if abs(dp) > 1e-10: sens_idx = f"{dI/dp:+.2f}"
            else: sens_idx = "--"
        else: sens_idx = "--"
        print(f"{name:<8} {val:<14.4e} {I2_str:<10} {status:<20} {sens_idx:<10}")
print("="*80)
print("\nLEGEND:")
print("  * Blue circles   : Transcritical point precisely determined")
print("  * Green triangles: Coexistence persists beyond scanned I range")
print("  * Red crosses    : No stable coexistence found in scanned range")
print("  * SensIdx        : Normalized sensitivity index")
print("  * Dashed red line: Baseline transcritical point I*2")
print("  * Dotted gray line: Baseline parameter value")
print("\nNOTE: One-at-a-time analysis captures main effects but misses parameter")
print("interactions. The two heatmaps test joint uncertainty in correlated pairs.")
print("\nCORRECTED BIFURCATION CONTEXT:")
print(f"  I*1 (extinction stable) ≈ {baseline_I1:.2f} mg/L/hr" if baseline_I1 else "  I*1: not computed")
print(f"  I*2 (transcritical)     ≈ {baseline_I2:.2f} mg/L/hr" if baseline_I2 else "  I*2: not computed")
print(f"  TYPE 1 bistable range: I ∈ [{baseline_I1:.2f}, {baseline_I2:.2f}]" if baseline_I1 and baseline_I2 else "  TYPE 1 range: see above")
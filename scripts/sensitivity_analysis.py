"""
COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS (v2)
================================================

PURPOSE
-------
Determines how the critical dosing rate I* depends on model parameters.
Includes THREE analyses:
  1. ONE-AT-A-TIME: Varies each parameter individually (main effects).
  2. TWO-PARAMETER INTERACTION: Varies correlated pairs jointly (e.g., b vs b_R)
     to catch parameter interactions missed by one-at-a-time analysis.
  3. BASELINE CROSS-CHECK: Verifies baseline I* matches compendium v4 value.

METHODOLOGY NOTES
-----------------
One-at-a-time sensitivity captures MAIN EFFECTS but misses INTERACTIONS.
For example, b (kill rate susceptible) and b_R (kill rate resistant) are
likely correlated in practice (both depend on the same pharmacodynamic assay).
The two-parameter heatmap (Analysis 2) tests whether their joint uncertainty
produces wider I* ranges than individual variation suggests.

POST-HOC VALIDATION FRAMEWORK
-----------------------------
Sensitivity analysis, not parameter estimation. All parameters fixed as priors.
The sensitivity index S = (ΔI*/I*_base)/(Δp/p_base) bounds PRACTICAL
IDENTIFIABILITY: large |S| → wide uncertainty in tipping point given parameter
uncertainty ranges.
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
# Parameter   Value        Source / Context                  Uncertainty   Role
# r_S         1.0 /gen     P. aeruginosa chemostat           ±5%           Susceptible growth
# r_R         0.93 /gen    Plasmid burden (And10)            ±5%           Resistant growth
# c_R         0.04         Plasmid fitness cost (And10)      ±20%          Resistance cost
# K           1e9 cells/mL Carrying capacity                 ±20%          Population scale
# b           2.0 /gen     Time-kill Emax (Regoes04)         ±30%          Max kill susceptible
# b_R         1.5 /gen     Partial resistance (Hal19)        ±30%          Max kill resistant
# MIC_S       2.0 mg/L     EUCAST breakpoint anchor          ±1 dilution   Susceptible MIC
# MIC_R       4.0 mg/L     Partial resistance regime         ±1 dilution   Resistant MIC
# n           3.0          PK/PD sigmoidicity (Regoes04)     ±15%          Hill coefficient
# mu          1.0 /hr      Drug clearance (Gat22-like)       ±20%          PK elimination
# eta         2e-8 /cell/hr Collective degradation (Bar83)   ±50%          Drug depletion
# gamma       1e-12        HGT rate (Levin 1997)             ±1 order      Conjugation
# ------------------------------------------------------------------------------
# CAVEAT: Literature priors from heterogeneous sources. Structural theorem
# (compendium Sec 2): bistability requires 3D + endogenous drug feedback.
# ------------------------------------------------------------------------------

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

def is_stable(state, I_val, p_dict, tol=-1e-8):
    J = jacobian(state, I_val, p_dict)
    eigs = eigvals(J)
    return np.all(eigs.real < tol)

def has_stable_coexistence(I_val, p_dict, p_tol=1e-4, res_tol=1e-5):
    K = p_dict['K']
    C_approx = I_val / p_dict['mu']
    seeds = []
    for N_frac in [0.5, 0.7, 0.85, 0.95, 0.99]:
        for p_frac in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
            seeds.append([K * N_frac, p_frac, C_approx])
    for N_frac in [0.8, 0.95]:
        for p_frac in [0.25, 0.5, 0.75]:
            seeds.append([K * N_frac, p_frac, C_approx * 0.3])
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
            if is_stable(sol, I_val, p_dict):
                return True
        except: continue
    return False

def find_critical_I(p_dict, I_min=0.5, I_max=30.0, n_coarse=40, bisection_tol=0.05, max_extend=3):
    I_vals = np.linspace(I_min, I_max, n_coarse)
    flags = []
    for I_val in I_vals:
        flags.append(has_stable_coexistence(I_val, p_dict))
    if not any(flags):
        I_low_test = np.linspace(0.1, I_min, 10)
        for I_val in I_low_test:
            if has_stable_coexistence(I_val, p_dict):
                return I_val, 'success'
        return None, 'no_coexistence'
    if flags[-1]:
        for ext in range(max_extend):
            I_new_max = I_max * 2
            I_test = np.linspace(I_max, I_new_max, 20)
            new_flags = [has_stable_coexistence(I, p_dict) for I in I_test]
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
        if has_stable_coexistence(I_mid, p_dict):
            I_low = I_mid
        else:
            I_high = I_mid
    return I_low, 'success'

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
print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS (v2)")
print("="*80)

print("\n[1/3] Computing baseline critical I*...")
t0 = time.time()
baseline_crit, baseline_status = find_critical_I(BASE_PARAMS, I_min=0.5, I_max=25.0)
t_baseline = time.time() - t0

if baseline_crit is None:
    print(f"  BASELINE: No coexistence found ({baseline_status})")
else:
    print(f"  BASELINE: I* = {baseline_crit:.4f} mg/L/hr  [{baseline_status}]  (took {t_baseline:.1f}s)")
    compendium_I_star = 7.84
    discrepancy = abs(baseline_crit - compendium_I_star) / compendium_I_star * 100
    print(f"  CROSS-CHECK: Compendium v4 predicts I* ≈ {compendium_I_star:.2f}")
    print(f"  Discrepancy: {discrepancy:.1f}%")
    if discrepancy > 10:
        print("  WARNING: Discrepancy >10%. Check parameter values or numerical tolerance.")

print(f"\n[2/3] Running one-at-a-time sensitivity sweeps ({len(SENSITIVITY_CONFIG)} parameters)...")
print("-"*80)

results = {}
for cfg in SENSITIVITY_CONFIG:
    name = cfg["name"]
    values = cfg["range"]
    tips = []
    statuses = []
    print(f"\nSweeping {name:8s} (baseline={cfg['baseline']})...")
    for val in values:
        p_dict = BASE_PARAMS.copy()
        p_dict[name] = val
        t0 = time.time()
        crit, status = find_critical_I(p_dict, I_min=0.5, I_max=25.0)
        dt = time.time() - t0
        tips.append(crit)
        statuses.append(status)
        if crit is None:
            print(f"  {name}={val:12.4e}  ->  I*=N/A  [{status:22s}]  ({dt:.1f}s)")
        else:
            if baseline_crit is not None and baseline_crit > 0 and cfg["baseline"] != 0:
                dI = (crit - baseline_crit) / baseline_crit
                dp = (val - cfg["baseline"]) / cfg["baseline"]
                if abs(dp) > 1e-10:
                    sens_idx = dI / dp
                else:
                    sens_idx = np.nan
            else:
                sens_idx = np.nan
            print(f"  {name}={val:12.4e}  ->  I*={crit:8.3f}  [{status:10s}]  S={sens_idx:+.2f}  ({dt:.1f}s)")
    results[name] = {
        "values": values, "tipping": tips, "statuses": statuses,
        "log": cfg["log"], "unit": cfg["unit"], "baseline": cfg["baseline"]
    }

print("\n[3/3] Running two-parameter interaction analysis (b vs b_R)...")
print("-"*80)
b_range = [1.0, 1.5, 2.0, 2.5, 3.0]
bR_range = [0.5, 1.0, 1.5, 2.0, 2.5]
interaction_matrix = np.zeros((len(b_range), len(bR_range)))
interaction_status = np.empty((len(b_range), len(bR_range)), dtype=object)
for i, b_val in enumerate(b_range):
    for j, bR_val in enumerate(bR_range):
        p_dict = BASE_PARAMS.copy()
        p_dict["b"] = b_val
        p_dict["b_R"] = bR_val
        crit, status = find_critical_I(p_dict, I_min=0.5, I_max=25.0)
        interaction_matrix[i, j] = crit if crit is not None else np.nan
        interaction_status[i, j] = status
        crit_str = f"{crit:.2f}" if crit is not None else "N/A"
        print(f"  b={b_val:.1f}, b_R={bR_val:.1f} -> I*={crit_str} [{status}]")

# PLOTTING
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.35)
n_params = len(results)
for idx, (name, data) in enumerate(results.items()):
    row = idx // 4
    col = idx % 4
    ax = fig.add_subplot(gs[row, col])
    vals = np.array(data["values"], dtype=float)
    tips = np.array(data["tipping"], dtype=float)
    statuses = data["statuses"]
    x_s = []; y_s = []
    x_e = []; y_e = []
    x_f = []; y_f = []
    for v, t, s in zip(vals, tips, statuses):
        if s == "success": x_s.append(v); y_s.append(t)
        elif s == "extends_beyond_range": x_e.append(v); y_e.append(t)
        else: x_f.append(v); y_f.append(t)
    vm = np.array([s == "success" or s == "extends_beyond_range" for s in statuses])
    if np.any(vm):
        if data["log"]: ax.semilogx(vals[vm], tips[vm], "-", color="steelblue", linewidth=2, zorder=1)
        else: ax.plot(vals[vm], tips[vm], "-", color="steelblue", linewidth=2, zorder=1)
    if x_s: ax.scatter(x_s, y_s, c="blue", s=40, zorder=3, label="Converged")
    if x_e: ax.scatter(x_e, y_e, c="green", s=40, marker="^", zorder=3, label="Extends")
    if x_f: ax.scatter(x_f, y_f, c="red", s=40, marker="x", zorder=3, label="No coexistence")
    if baseline_crit is not None: ax.axhline(y=baseline_crit, color="crimson", linestyle="--", alpha=0.6, linewidth=1)
    ax.axvline(x=data["baseline"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.set_xlabel(f"{name} ({data['unit']})", fontsize=9)
    ax.set_ylabel("I*", fontsize=9)
    ax.set_title(f"{name}", fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)
ax = fig.add_subplot(gs[2, 3])
im = ax.imshow(interaction_matrix, cmap="RdYlGn_r", aspect="auto", vmin=np.nanmin(interaction_matrix), vmax=np.nanmax(interaction_matrix))
ax.set_xticks(range(len(bR_range)))
ax.set_xticklabels([f"{v:.1f}" for v in bR_range], fontsize=8)
ax.set_yticks(range(len(b_range)))
ax.set_yticklabels([f"{v:.1f}" for v in b_range], fontsize=8)
ax.set_xlabel("b_R", fontsize=10)
ax.set_ylabel("b", fontsize=10)
ax.set_title("Two-parameter interaction\n(b vs b_R)", fontsize=11, fontweight="bold")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("I*", fontsize=9)
for i in range(len(b_range)):
    for j in range(len(bR_range)):
        if not np.isnan(interaction_matrix[i, j]):
            ax.text(j, i, f"{interaction_matrix[i,j]:.1f}", ha="center", va="center", fontsize=7, color="black")
plt.suptitle("Parameter Sensitivity: Main Effects + Two-Parameter Interaction", fontsize=13, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig("comprehensive_sensitivity_v2.png", dpi=300, bbox_inches="tight")
plt.show()

# SUMMARY TABLE
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print("Param    Value         I*        Status               SensIdx")
print("-"*80)
for name, data in results.items():
    for val, tip, status in zip(data["values"], data["tipping"], data["statuses"]):
        tip_str = f"{tip:.3f}" if tip is not None else "N/A"
        if baseline_crit is not None and baseline_crit > 0 and data["baseline"] != 0 and tip is not None:
            dI = (tip - baseline_crit) / baseline_crit
            dp = (val - data["baseline"]) / data["baseline"]
            if abs(dp) > 1e-10: sens_idx = f"{dI/dp:+.2f}"
            else: sens_idx = "--"
        else: sens_idx = "--"
        print(f"{name:<8} {val:<14.4e} {tip_str:<10} {status:<20} {sens_idx:<10}")
print("="*80)
print("\nLEGEND:")
print("  * Blue circles   : Stable coexistence found, critical I precisely determined")
print("  * Green triangles: Coexistence persists beyond scanned I range")
print("  * Red crosses    : No stable coexistence found in scanned range")
print("  * SensIdx        : Normalized sensitivity index")
print("  * Dashed red line: Baseline critical I*")
print("  * Dotted gray line: Baseline parameter value")
print("\nNOTE: One-at-a-time analysis captures main effects but misses parameter")
print("interactions. The b vs b_R heatmap (Analysis 3) tests joint uncertainty.")
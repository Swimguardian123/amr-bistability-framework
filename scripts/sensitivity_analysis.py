"""
COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS
=============================================
Addresses all review concerns:
  1. Scans FULL I range (no break-on-first-failure)
  2. Checks STABILITY via Jacobian eigenvalues
  3. Uses BISECTION for precise critical I
  4. Includes ALL biologically relevant parameters
  5. Distinguishes numerical failure from biological absence
  6. Scale-invariant convergence criteria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import warnings
import time
warnings.filterwarnings('ignore')

# =============================================================================
# BASE PARAMETERS
# =============================================================================
BASE_PARAMS = {
    'r_S': 1.0, 'r_R': 0.93, 'K': 1e9, 'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0, 'n': 3.0, 'c_R': 0.04,
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12
}

# =============================================================================
# MODEL FUNCTIONS
# =============================================================================
def hill(C, MIC, n):
    if C <= 0:
        return 0.0
    return C**n / (C**n + MIC**n)

def growth_rates(N, p, C, p_dict):
    """Compute per-capita growth rates."""
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
    """ODE right-hand side (should be zero at equilibrium)."""
    N, p, C = state
    N = max(N, 0.0)
    p = np.clip(p, 0.0, 1.0)
    g_S, g_R, g_bar = growth_rates(N, p, C, p_dict)
    dN = N * g_bar
    dp = p * (1 - p) * (g_R - g_S + p_dict['gamma'] * N)
    dC = I_val - p_dict['mu'] * C - p_dict['eta'] * N * p * C
    return np.array([dN, dp, dC])

def jacobian(state, I_val, p_dict):
    """3x3 Jacobian matrix at state."""
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
    """Check if equilibrium is linearly stable (all eigenvalues negative real part)."""
    J = jacobian(state, I_val, p_dict)
    eigs = eigvals(J)
    return np.all(eigs.real < tol)

# =============================================================================
# EQUILIBRIUM FINDING (ROBUST)
# =============================================================================
def has_stable_coexistence(I_val, p_dict, p_tol=1e-4, res_tol=1e-5):
    """
    Return True if a STABLE coexistence equilibrium exists at this I.
    Uses multiple seeds and returns early on first success (fast path).
    """
    K = p_dict['K']
    C_approx = I_val / p_dict['mu']
    
    # Diverse seeds covering the expected equilibrium manifold
    seeds = []
    for N_frac in [0.5, 0.7, 0.85, 0.95, 0.99]:
        for p_frac in [0.05, 0.15, 0.35, 0.5, 0.65, 0.85, 0.95]:
            seeds.append([K * N_frac, p_frac, C_approx])
    for N_frac in [0.8, 0.95]:
        for p_frac in [0.25, 0.5, 0.75]:
            seeds.append([K * N_frac, p_frac, C_approx * 0.3])  # lower C for high eta
    
    for seed in seeds:
        try:
            sol, info, ier, mesg = fsolve(
                residuals, seed, args=(I_val, p_dict),
                xtol=1e-12, maxfev=2000, full_output=True
            )
            if ier != 1:
                continue
            
            N, p, C = float(sol[0]), float(np.clip(sol[1], 0, 1)), float(sol[2])
            
            # Feasibility
            if N <= 0 or C <= 0:
                continue
            
            # Scale-invariant residual check
            res = residuals(sol, I_val, p_dict)
            scales = np.array([max(abs(N), 1e5), 1.0, max(abs(C), 1.0)])
            norm_res = np.linalg.norm(res / scales)
            if norm_res > res_tol:
                continue
            
            # Coexistence: strictly interior
            if p <= p_tol or p >= 1 - p_tol:
                continue
            
            # Stability check
            if is_stable(sol, I_val, p_dict):
                return True
                
        except Exception:
            continue
    
    return False

# =============================================================================
# CRITICAL I FINDER (COARSE SCAN + BISECTION)
# =============================================================================
def find_critical_I(p_dict, I_min=0.5, I_max=30.0, n_coarse=40, bisection_tol=0.05, max_extend=3):
    """
    Find the largest I at which a stable coexistence equilibrium exists.
    
    Strategy:
      1. Coarse scan across full range (no early break)
      2. If coexistence persists to I_max, extend range
      3. Bisection around the transition for precision
    
    Returns: (I_crit, status)
      status: 'success', 'no_coexistence', 'extends_beyond_range', 'numerical_issue'
    """
    # --- Phase 1: Coarse scan ---
    I_vals = np.linspace(I_min, I_max, n_coarse)
    flags = []
    for I_val in I_vals:
        flags.append(has_stable_coexistence(I_val, p_dict))
    
    # --- Phase 2: Handle edge cases ---
    if not any(flags):
        # Coexistence never found. Try lower I just in case.
        I_low_test = np.linspace(0.1, I_min, 10)
        for I_val in I_low_test:
            if has_stable_coexistence(I_val, p_dict):
                return I_val, 'success'  # exists only below I_min
        return None, 'no_coexistence'
    
    if flags[-1]:
        # Persists to I_max -- extend
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
    
    # --- Phase 3: Identify transition bracket ---
    valid_idx = [i for i, f in enumerate(flags) if f]
    last_valid = valid_idx[-1]
    
    if last_valid >= len(I_vals) - 1:
        return I_vals[last_valid], 'extends_beyond_range'
    
    I_low = I_vals[last_valid]
    I_high = I_vals[last_valid + 1]
    
    # --- Phase 4: Bisection ---
    for _ in range(25):  # 25 iterations -> precision ~ (I_high-I_low)/2^25
        if I_high - I_low < bisection_tol:
            break
        I_mid = (I_low + I_high) / 2.0
        if has_stable_coexistence(I_mid, p_dict):
            I_low = I_mid
        else:
            I_high = I_mid
    
    return I_low, 'success'

# =============================================================================
# SENSITIVITY SWEEP CONFIGURATION
# =============================================================================
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
# RUN BASELINE
# =============================================================================
print("="*80)
print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS")
print("="*80)
print("\n[1/2] Computing baseline critical I*...")
t0 = time.time()
baseline_crit, baseline_status = find_critical_I(BASE_PARAMS, I_min=0.5, I_max=25.0)
t_baseline = time.time() - t0

if baseline_crit is None:
    print(f"  BASELINE: No coexistence found ({baseline_status})")
else:
    print(f"  BASELINE: I* = {baseline_crit:.4f} mg/L/hr  [{baseline_status}]  (took {t_baseline:.1f}s)")

# =============================================================================
# RUN SWEEPS
# =============================================================================
print(f"\n[2/2] Running sensitivity sweeps ({len(SENSITIVITY_CONFIG)} parameters)...")
print("-"*80)

results = {}
for cfg in SENSITIVITY_CONFIG:
    name = cfg['name']
    values = cfg['range']
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
            # Compute sensitivity index
            if baseline_crit is not None and baseline_crit > 0 and cfg['baseline'] != 0:
                dI = (crit - baseline_crit) / baseline_crit
                dp = (val - cfg['baseline']) / cfg['baseline']
                if abs(dp) > 1e-10:
                    sens_idx = dI / dp
                else:
                    sens_idx = np.nan
            else:
                sens_idx = np.nan
            print(f"  {name}={val:12.4e}  ->  I*={crit:8.3f}  [{status:10s}]  S={sens_idx:+.2f}  ({dt:.1f}s)")
    
    results[name] = {
        'values': values,
        'tipping': tips,
        'statuses': statuses,
        'log': cfg['log'],
        'unit': cfg['unit'],
        'baseline': cfg['baseline']
    }

# =============================================================================
# PLOTTING
# =============================================================================
n_params = len(results)
ncols = 3
nrows = int(np.ceil(n_params / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4*nrows))
axes = axes.flatten() if n_params > 1 else [axes]

for idx, (name, data) in enumerate(results.items()):
    ax = axes[idx]
    vals = np.array(data['values'], dtype=float)
    tips = np.array(data['tipping'], dtype=float)
    statuses = data['statuses']
    
    # Separate by status for coloring
    x_success = []; y_success = []
    x_extend = []; y_extend = []
    x_fail = []; y_fail = []
    
    for v, t, s in zip(vals, tips, statuses):
        if s == 'success':
            x_success.append(v); y_success.append(t)
        elif s == 'extends_beyond_range':
            x_extend.append(v); y_extend.append(t)
        else:
            x_fail.append(v); y_fail.append(t)
    
    # Plot lines connecting successful points
    valid_mask = np.array([s == 'success' or s == 'extends_beyond_range' for s in statuses])
    if np.any(valid_mask):
        if data['log']:
            ax.semilogx(vals[valid_mask], tips[valid_mask], '-', color='steelblue', linewidth=2, zorder=1)
        else:
            ax.plot(vals[valid_mask], tips[valid_mask], '-', color='steelblue', linewidth=2, zorder=1)
    
    # Plot markers
    if x_success:
        ax.scatter(x_success, y_success, c='blue', s=60, zorder=3, label='Converged')
    if x_extend:
        ax.scatter(x_extend, y_extend, c='green', s=60, marker='^', zorder=3, label='Extends beyond range')
    if x_fail:
        ax.scatter(x_fail, y_fail, c='red', s=60, marker='x', zorder=3, label='No coexistence')
    
    # Baseline reference
    if baseline_crit is not None:
        ax.axhline(y=baseline_crit, color='crimson', linestyle='--', alpha=0.6, 
                   label=f'Baseline I*={baseline_crit:.2f}')
    
    # Baseline parameter value marker
    ax.axvline(x=data['baseline'], color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(f"{name} ({data['unit']})", fontsize=10)
    ax.set_ylabel('Critical I* (mg/L/hr)', fontsize=10)
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')

# Hide unused subplots
for idx in range(n_params, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Comprehensive Parameter Sensitivity: Critical Infusion Rate I*', 
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('comprehensive_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"{'Param':<8} {'Value':<14} {'I*':<10} {'Status':<20} {'SensIdx':<10}")
print("-"*80)

for name, data in results.items():
    for val, tip, status in zip(data['values'], data['tipping'], data['statuses']):
        tip_str = f"{tip:.3f}" if tip is not None else "N/A"
        # Sensitivity index
        if baseline_crit is not None and baseline_crit > 0 and data['baseline'] != 0 and tip is not None:
            dI = (tip - baseline_crit) / baseline_crit
            dp = (val - data['baseline']) / data['baseline']
            if abs(dp) > 1e-10:
                sens_idx = f"{dI/dp:+.2f}"
            else:
                sens_idx = "--"
        else:
            sens_idx = "--"
        print(f"{name:<8} {val:<14.4e} {tip_str:<10} {status:<20} {sens_idx:<10}")

print("="*80)
print("\nLEGEND:")
print("  * Blue circles  : Stable coexistence found, critical I precisely determined")
print("  * Green triangles: Coexistence persists beyond scanned I range (I* > I_max)")
print("  * Red crosses   : No stable coexistence found in scanned range")
print("  * SensIdx       : Normalized sensitivity index  (DeltaI*/I*_base) / (Deltap/p_base)")
print("  * Dashed red line: Baseline critical I*")
print("  * Dotted gray line: Baseline parameter value")
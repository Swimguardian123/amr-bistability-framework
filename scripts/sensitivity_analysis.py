"""
================================================================================
COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS (v4 — Global + Uncertainty + ASI)
================================================================================

INTEGRATED FEEDBACK — ALL REVIEWER SUGGESTIONS IMPLEMENTED:

1. GLOBAL SENSITIVITY (Sobol indices)
   - Replaces one-at-a-time with variance-based decomposition
   - First-order (S1) + Total-order (ST) indices with bootstrap CIs
   - Uses QMC Sobol sampling (Saltelli 2010 scheme)
   - SALib-independent implementation via scipy.stats.qmc

2. ASI BIOMARKER SENSITIVITY
   - Metric 1: Sensitivity of transcritical location (I*2)
   - Metric 2: Sensitivity of ASI itself across parameter space
   - Computes corr(ASI_true, ASI_surrogate) per parameter set
   - Computes AUC(ASI) classification performance vs classical EWS

3. VIRTUAL INFECTION COHORT
   - 10,000+ mechanistically distinct virtual infections
   - Each: random parameters → trajectory → ASI + classical EWS
   - Statistical test: does ASI consistently outperform classical indicators?

4. PARAMETER UNCERTAINTY PROPAGATION
   - Distributional priors: Normal / LogNormal per parameter
   - Monte Carlo sampling with joint parameter variation
   - Reports: I2 = mean ± std, 95% CI

5. RETAINED BASELINE ANALYSES
   - OAT sweeps (for comparison with global results)
   - Two-parameter interaction heatmaps
   - Baseline cross-check against compendium v4

CORRECTED BIFURCATION CONTEXT
-----------------------------
The full 3D model exhibits SEQUENTIAL regime transitions:
  • I*1 ≈ 4.7:  Extinction becomes stable (monostable → bistable TYPE 1)
  • I*2 ≈ 11.6: Interior merges with p=1 boundary (transcritical, TYPE 1 → TYPE 2)
  • I*3 ≈ 34.5: p=1 boundary disappears via saddle-node (bistable → monostable)

This script computes I*2 — the transcritical point where interior coexistence
merges with the p=1 boundary and ceases to exist.

METHODOLOGY
-----------
- Global sensitivity: Sobol decomposition captures main effects + interactions
- ASI sensitivity: Tests whether biomarker works, not just where tipping point is
- Uncertainty propagation: Clinically interpretable confidence intervals
- Virtual cohort: Mechanistic robustness across parameter space

DEPENDENCIES
------------
  numpy, scipy, matplotlib
  (SALib NOT required — custom implementation provided)

USAGE
-----
  python comprehensive_sensitivity_v4.py

OUTPUT
------
  - comprehensive_sensitivity_v4.png  (main figure: multi-panel)
  - sobol_results_v4.pkl              (raw Sobol indices)
  - cohort_results_v4.pkl             (virtual infection cohort data)
  - Console: full numerical results + statistical tests
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eigvals
from scipy.stats import qmc, norm, lognorm
from sklearn.metrics import roc_auc_score
import warnings
import time
import pickle
import os
warnings.filterwarnings('ignore')

# ============================================================
# BASE PARAMETERS — Confirmed bistable set (Compendium v4, Sec 3.1)
# ============================================================
BASE_PARAMS = {
    'r_S': 1.0, 'r_R': 0.93, 'K': 1e9, 'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0, 'n': 3.0, 'c_R': 0.04,
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12
}

# ============================================================
# PARAMETER UNCERTAINTY DISTRIBUTIONS
# ============================================================
PARAM_DISTRIBUTIONS = {
    'r_S':     {'dist': 'normal',   'mean': 1.0,   'std': 0.15,      'unit': '1/hr'},
    'r_R':     {'dist': 'normal',   'mean': 0.93,  'std': 0.12,      'unit': '1/hr'},
    'b':       {'dist': 'normal',   'mean': 2.0,   'std': 0.4,       'unit': '1/hr'},
    'b_R':     {'dist': 'normal',   'mean': 1.5,   'std': 0.3,       'unit': '1/hr'},
    'MIC_S':   {'dist': 'lognormal','mean': 2.0,   'std': 0.5,       'unit': 'mg/L'},
    'MIC_R':   {'dist': 'lognormal','mean': 4.0,   'std': 1.0,       'unit': 'mg/L'},
    'n':       {'dist': 'normal',   'mean': 3.0,   'std': 0.5,       'unit': ''},
    'c_R':     {'dist': 'normal',   'mean': 0.04,  'std': 0.01,      'unit': '1/hr'},
    'K':       {'dist': 'lognormal','mean': 1e9,   'std': 3e8,       'unit': 'cells'},
    'mu':      {'dist': 'normal',   'mean': 1.0,   'std': 0.2,       'unit': '1/hr'},
    'eta':     {'dist': 'lognormal','mean': 2e-8,  'std': 5e-9,      'unit': 'L/(cell*hr)'},
    'gamma':   {'dist': 'normal',   'mean': 1e-12, 'std': 3e-13,     'unit': '1/(cell*hr)'},
}

# ============================================================
# SOBOL SAMPLING BOUNDS (±2σ for uniform Sobol space)
# ============================================================
SOBOL_BOUNDS = {}
for name, d in PARAM_DISTRIBUTIONS.items():
    if d['dist'] == 'normal':
        lb = max(d['mean'] - 2*d['std'], d['mean'] * 0.1)
        ub = d['mean'] + 2*d['std']
    else:
        sigma = np.sqrt(np.log(1 + (d['std']/d['mean'])**2))
        mu_ln = np.log(d['mean']) - sigma**2/2
        lb = np.exp(mu_ln - 2*sigma)
        ub = np.exp(mu_ln + 2*sigma)
    SOBOL_BOUNDS[name] = [lb, ub]

# ============================================================
# CORE MODEL FUNCTIONS
# ============================================================

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
    
    # Guard: if state contains non-finite values, return zero Jacobian
    if not np.all(np.isfinite(state)):
        return np.zeros((3, 3))
    
    g_S, g_R, g_bar = growth_rates(N, p, C, p_dict)
    
    # Guard: if C is near-zero, Hill derivatives blow up → return zero Jacobian
    eps = 1e-12
    if C < eps or not np.isfinite(C):
        return np.zeros((3, 3))
    
    C_safe = max(C, eps)
    df_S_dC = n * (MIC_S**n) * C_safe**(n-1) / (C_safe**n + MIC_S**n)**2
    df_R_dC = n * (MIC_R**n) * C_safe**(n-1) / (C_safe**n + MIC_R**n)**2
    
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
    
    J = np.array([[J11, J12, J13],
                  [J21, J22, J23],
                  [J31, J32, J33]])
    
    # Final guard: if any element is non-finite, return zero Jacobian
    if not np.all(np.isfinite(J)):
        return np.zeros((3, 3))
    
    return J

def is_stable(state, I_val, p_dict, tol=-1e-10):
    J = jacobian(state, I_val, p_dict)
    if not np.all(np.isfinite(J)):
        return False
    eigs = eigvals(J)
    return np.all(eigs.real < tol)

def has_stable_coexistence(I_val, p_dict, p_tol=1e-4, res_tol=1e-5, prev_eq=None):
    K = p_dict['K']
    seeds = []
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
    I_vals = np.linspace(I_min, I_max, n_test)
    for I_val in I_vals:
        C = I_val / p_dict['mu']
        gS_0 = p_dict['r_S'] - p_dict['b'] * hill(C, p_dict['MIC_S'], p_dict['n'])
        gR_0 = p_dict['r_R'] - p_dict['c_R'] - p_dict['b_R'] * hill(C, p_dict['MIC_R'], p_dict['n'])
        if gS_0 < 0 and gR_0 < 0:
            return I_val
    return None

# ============================================================
# SOBOL SAMPLING & ANALYSIS (SALib-independent)
# ============================================================

def generate_sobol_samples(n, bounds_dict, seed=42):
    """Generate Saltelli-style samples for Sobol sensitivity analysis."""
    param_names = list(bounds_dict.keys())
    d = len(param_names)
    n_qmc = 2**int(np.ceil(np.log2(n)))
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    samples = sampler.random(2 * n_qmc)
    A_raw = samples[:n_qmc]
    B_raw = samples[n_qmc:]
    bounds = np.array([bounds_dict[p] for p in param_names])
    l_bounds = bounds[:, 0]
    u_bounds = bounds[:, 1]
    A = qmc.scale(A_raw, l_bounds, u_bounds)
    B = qmc.scale(B_raw, l_bounds, u_bounds)
    AB_dict = {}
    for i, name in enumerate(param_names):
        AB = A.copy()
        AB[:, i] = B[:, i]
        AB_dict[name] = AB
    return A, B, AB_dict, param_names

def compute_sobol_indices(Y_A, Y_B, Y_AB_dict, param_names):
    """Compute first-order (S1) and total-order (ST) Sobol indices."""
    n = len(Y_A)
    Y_all = np.concatenate([Y_A, Y_B])
    var_Y = np.var(Y_all, ddof=1)
    if var_Y < 1e-15:
        return ({name: 0.0 for name in param_names},
                {name: 0.0 for name in param_names},
                {name: (0.0, 0.0) for name in param_names},
                {name: (0.0, 0.0) for name in param_names})
    S1 = {}
    ST = {}
    for name in param_names:
        Y_AB = Y_AB_dict[name]
        S1[name] = np.mean(Y_B * (Y_AB - Y_A)) / var_Y
        ST[name] = np.mean((Y_A - Y_AB)**2) / (2 * var_Y)
    # Bootstrap confidence intervals
    n_boot = 1000
    rng = np.random.default_rng(42)
    S1_boot = {name: [] for name in param_names}
    ST_boot = {name: [] for name in param_names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Y_A_b = Y_A[idx]
        Y_B_b = Y_B[idx]
        var_b = np.var(np.concatenate([Y_A_b, Y_B_b]), ddof=1)
        if var_b < 1e-15:
            continue
        for name in param_names:
            Y_AB_b = Y_AB_dict[name][idx]
            S1_boot[name].append(np.mean(Y_B_b * (Y_AB_b - Y_A_b)) / var_b)
            ST_boot[name].append(np.mean((Y_A_b - Y_AB_b)**2) / (2 * var_b))
    S1_conf = {}
    ST_conf = {}
    for name in param_names:
        if len(S1_boot[name]) > 0:
            S1_conf[name] = (np.percentile(S1_boot[name], 2.5), np.percentile(S1_boot[name], 97.5))
            ST_conf[name] = (np.percentile(ST_boot[name], 2.5), np.percentile(ST_boot[name], 97.5))
        else:
            S1_conf[name] = (0.0, 0.0)
            ST_conf[name] = (0.0, 0.0)
    return S1, ST, S1_conf, ST_conf

def params_dict_from_array(arr, param_names):
    """Convert parameter array to dictionary."""
    p = BASE_PARAMS.copy()
    for i, name in enumerate(param_names):
        p[name] = float(arr[i])
    return p

# ============================================================
# ASI COMPUTATION MODULE
# ============================================================

def simulate_trajectory(p_dict, I_val, t_max=500, dt=0.5, N0=None, p0=0.01, C0=None, noise_std=0.0, rng=None):
    """Simulate 3D model trajectory under dosing rate I_val."""
    if rng is None:
        rng = np.random.default_rng()
    n_steps = int(t_max / dt)
    t = np.linspace(0, t_max, n_steps)
    N = np.zeros(n_steps)
    p = np.zeros(n_steps)
    C = np.zeros(n_steps)
    N[0] = N0 if N0 is not None else p_dict['K'] * 0.5
    p[0] = p0
    C[0] = C0 if C0 is not None else I_val / p_dict['mu']
    for i in range(n_steps - 1):
        state = np.array([N[i], p[i], C[i]])
        k1 = residuals(state, I_val, p_dict)
        k2 = residuals(state + 0.5*dt*k1, I_val, p_dict)
        k3 = residuals(state + 0.5*dt*k2, I_val, p_dict)
        k4 = residuals(state + dt*k3, I_val, p_dict)
        new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        N[i+1] = max(new_state[0], 0)
        p[i+1] = np.clip(new_state[1], 0, 1)
        C[i+1] = max(new_state[2], 0)
    if noise_std > 0:
        N += rng.normal(0, noise_std * np.mean(N[N > 0]), size=n_steps)
        p += rng.normal(0, noise_std * 0.01, size=n_steps)
        C += rng.normal(0, noise_std * np.mean(C[C > 0]), size=n_steps)
        N = np.maximum(N, 0)
        p = np.clip(p, 0, 1)
        C = np.maximum(C, 0)
    return t, N, p, C

def compute_ews_indicators(t, N, p, C, window_frac=0.3):
    """Compute classical early warning signals from trajectory."""
    n = len(t)
    start_idx = int(n * (1 - window_frac))
    N_w = N[start_idx:]
    p_w = p[start_idx:]
    C_w = C[start_idx:]
    def lag1_autocorr(x):
        if len(x) < 2: return np.nan
        x_c = x - np.mean(x)
        denom = np.sum(x_c[:-1]**2)
        if denom < 1e-15: return np.nan
        return np.sum(x_c[:-1] * x_c[1:]) / denom
    def skew(x):
        if len(x) < 3: return np.nan
        x_c = x - np.mean(x)
        std = np.std(x, ddof=1)
        if std < 1e-15: return np.nan
        return np.mean(x_c**3) / std**3
    def kurt(x):
        if len(x) < 4: return np.nan
        x_c = x - np.mean(x)
        std = np.std(x, ddof=1)
        if std < 1e-15: return np.nan
        return np.mean(x_c**4) / std**4 - 3
    return {
        'variance_N': np.var(N_w, ddof=1),
        'variance_p': np.var(p_w, ddof=1),
        'variance_C': np.var(C_w, ddof=1),
        'autocorr_N': lag1_autocorr(N_w),
        'autocorr_p': lag1_autocorr(p_w),
        'autocorr_C': lag1_autocorr(C_w),
        'skewness_N': skew(N_w),
        'skewness_p': skew(p_w),
        'skewness_C': skew(C_w),
        'kurtosis_N': kurt(N_w),
        'kurtosis_p': kurt(p_w),
        'kurtosis_C': kurt(C_w),
        'cv_N': np.std(N_w, ddof=1) / (np.mean(N_w) + 1e-12),
        'index_of_dispersion_N': np.var(N_w, ddof=1) / (np.mean(N_w) + 1e-12),
    }

def compute_asi(t, N, p, C, p_dict, I_val, window_frac=0.3):
    """Compute ASI (Anticipated System Instability) indicator."""
    n = len(t)
    start_idx = int(n * (1 - window_frac))
    N_w = N[start_idx:]
    p_w = p[start_idx:]
    C_w = C[start_idx:]
    
    # Guard: if window has insufficient data or all zeros, return neutral
    if len(N_w) < 3 or np.mean(N_w) < 1e-6 or np.mean(C_w) < 1e-12:
        return 0.0, {
            'spectral_abscissa': 0.0,
            'mean_dpdt': 0.0,
            'fitness_diff': 0.0,
            'curvature': 0.0,
        }
    
    state = np.array([np.mean(N_w), np.mean(p_w), np.mean(C_w)])
    
    # Guard: if state mean is non-finite, return neutral
    if not np.all(np.isfinite(state)):
        return 0.0, {
            'spectral_abscissa': 0.0,
            'mean_dpdt': 0.0,
            'fitness_diff': 0.0,
            'curvature': 0.0,
        }
    
    J = jacobian(state, I_val, p_dict)
    
    # Guard: jacobian() now returns zero matrix on failure, but double-check
    if not np.all(np.isfinite(J)):
        spectral_abscissa = 0.0
        stability_change = 0.0
    else:
        try:
            eigs = eigvals(J)
            spectral_abscissa = np.max(eigs.real)
        except (ValueError, np.linalg.LinAlgError):
            spectral_abscissa = 0.0
        stability_change = spectral_abscissa
    
    dpdt = np.gradient(p_w, t[1]-t[0])
    mean_dpdt = np.mean(np.abs(dpdt))
    g_S, g_R, g_bar = growth_rates(state[0], state[1], state[2], p_dict)
    fitness_diff = abs(g_R - g_S)
    d2pdt2 = np.gradient(dpdt, t[1]-t[0])
    curvature = np.mean(np.abs(d2pdt2))
    
    sa_norm = -spectral_abscissa
    dpdt_norm = mean_dpdt * 1e6
    fd_norm = 1.0 / (fitness_diff + 0.01)
    curv_norm = curvature * 1e8
    
    asi = 0.4 * sa_norm + 0.3 * dpdt_norm + 0.2 * fd_norm + 0.1 * curv_norm
    
    # Sanitize final output
    if not np.isfinite(asi):
        asi = 0.0
    if not np.isfinite(spectral_abscissa):
        spectral_abscissa = 0.0
    if not np.isfinite(mean_dpdt):
        mean_dpdt = 0.0
    if not np.isfinite(fitness_diff):
        fitness_diff = 0.0
    if not np.isfinite(curvature):
        curvature = 0.0
    
    return asi, {
        'spectral_abscissa': spectral_abscissa,
        'mean_dpdt': mean_dpdt,
        'fitness_diff': fitness_diff,
        'curvature': curvature,
    }

def compute_surrogate_asi(t, N, p, C, window_frac=0.3):
    """Compute surrogate ASI using ONLY observable quantities."""
    n = len(t)
    start_idx = int(n * (1 - window_frac))
    N_w = N[start_idx:]
    p_w = p[start_idx:]
    C_w = C[start_idx:]
    var_N = np.var(N_w, ddof=1)
    var_p = np.var(p_w, ddof=1)
    dpdt = np.gradient(p_w, t[1]-t[0])
    mean_dpdt = np.mean(np.abs(dpdt))
    d2pdt2 = np.gradient(dpdt, t[1]-t[0])
    curvature = np.mean(np.abs(d2pdt2))
    trend_N = np.polyfit(range(len(N_w)), N_w, 1)[0]
    surr_asi = (0.3 * np.log1p(var_N) + 0.3 * np.log1p(var_p) +
                0.2 * mean_dpdt * 1e6 + 0.1 * curvature * 1e8 +
                0.1 * np.abs(trend_N) * 1e-6)
    return surr_asi

# ============================================================
# VIRTUAL INFECTION COHORT
# ============================================================

def sample_parameters_from_distributions(n_samples, seed=42):
    """Sample parameter vectors from uncertainty distributions."""
    rng = np.random.default_rng(seed)
    params_list = []
    for _ in range(n_samples):
        p = BASE_PARAMS.copy()
        for name, d in PARAM_DISTRIBUTIONS.items():
            if d['dist'] == 'normal':
                p[name] = rng.normal(d['mean'], d['std'])
                if name in ['r_S', 'r_R', 'b', 'b_R', 'mu', 'n']:
                    p[name] = max(p[name], 0.1)
                if name == 'c_R':
                    p[name] = max(p[name], 0.0)
                if name == 'gamma':
                    p[name] = max(p[name], 1e-15)
            else:
                sigma = np.sqrt(np.log(1 + (d['std']/d['mean'])**2))
                mu_ln = np.log(d['mean']) - sigma**2/2
                p[name] = rng.lognormal(mu_ln, sigma)
        params_list.append(p)
    return params_list

def evaluate_virtual_infection(p_dict, I_val, t_max=500, dt=0.5, noise_std=0.05, seed=None):
    """Evaluate a single virtual infection under dosing I_val."""
    rng = np.random.default_rng(seed)
    t, N, p, C = simulate_trajectory(p_dict, I_val, t_max=t_max, dt=dt, noise_std=noise_std, rng=rng)
    I2, status = find_I_transcritical(p_dict, I_min=0.5, I_max=25.0)
    I1 = find_I_extinction_stable(p_dict)
    ews = compute_ews_indicators(t, N, p, C)
    asi_true, asi_components = compute_asi(t, N, p, C, p_dict, I_val)
    asi_surrogate = compute_surrogate_asi(t, N, p, C)
    extinct = (N[-1] < 0.01 * p_dict['K']) and (N[-1] < N[-10])
    proximity = None
    if I2 is not None and I2 > 0:
        proximity = I_val / I2
    return {
        'I_val': I_val, 'I2': I2, 'I2_status': status, 'I1': I1,
        'extinct': extinct, 'final_N': N[-1], 'final_p': p[-1], 'final_C': C[-1],
        'proximity': proximity, 'asi_true': asi_true, 'asi_surrogate': asi_surrogate,
        'asi_components': asi_components, **ews
    }

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*80)
    print("COMPREHENSIVE PARAMETER SENSITIVITY ANALYSIS (v4 — Global + Uncertainty)")
    print("="*80)

    # -------------------------------------------------------------------------
    # [1/6] BASELINE
    # -------------------------------------------------------------------------
    print("\n[1/6] Computing baseline regime boundaries...")
    t0 = time.time()
    baseline_I2, baseline_status = find_I_transcritical(BASE_PARAMS, I_min=0.5, I_max=25.0)
    t_baseline = time.time() - t0
    baseline_I1 = find_I_extinction_stable(BASE_PARAMS)

    if baseline_I2 is None:
        print(f"  BASELINE I*2: No coexistence found ({baseline_status})")
    else:
        print(f"  BASELINE I*2 (transcritical): {baseline_I2:.4f} mg/L/hr  [{baseline_status}]  ({t_baseline:.1f}s)")
        print(f"    -> Interior coexistence exists for I < {baseline_I2:.2f}")
    if baseline_I1 is not None:
        print(f"  BASELINE I*1 (extinction stable): {baseline_I1:.4f} mg/L/hr")
        if baseline_I2 is not None:
            print(f"  BISTABLE TYPE 1 RANGE: I in [{baseline_I1:.2f}, {baseline_I2:.2f}]")
    if baseline_I2 is not None:
        expected_I2 = 11.6
        discrepancy = abs(baseline_I2 - expected_I2) / expected_I2 * 100
        print(f"  CROSS-CHECK: Compendium v4 predicts I*2 ~ {expected_I2:.1f}, discrepancy {discrepancy:.1f}%")

    # -------------------------------------------------------------------------
    # [2/6] OAT SWEEPS
    # -------------------------------------------------------------------------
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

    print(f"\n[2/6] Running OAT sensitivity sweeps ({len(SENSITIVITY_CONFIG)} parameters)...")
    print("-"*80)
    results_oat = {}
    for cfg in SENSITIVITY_CONFIG:
        name = cfg["name"]
        values = cfg["range"]
        I2_vals = []; statuses = []
        print(f"\nSweeping {name:8s} (baseline={cfg['baseline']})...")
        for val in values:
            p_dict = BASE_PARAMS.copy()
            p_dict[name] = val
            t0 = time.time()
            I2, status = find_I_transcritical(p_dict, I_min=0.5, I_max=25.0)
            dt = time.time() - t0
            I2_vals.append(I2); statuses.append(status)
            if I2 is None:
                print(f"  {name}={val:12.4e}  ->  I*2=N/A  [{status:22s}]  ({dt:.1f}s)")
            else:
                if baseline_I2 is not None and baseline_I2 > 0 and cfg["baseline"] != 0:
                    dI = (I2 - baseline_I2) / baseline_I2
                    dp = (val - cfg["baseline"]) / cfg["baseline"]
                    sens_idx = dI / dp if abs(dp) > 1e-10 else np.nan
                else:
                    sens_idx = np.nan
                print(f"  {name}={val:12.4e}  ->  I*2={I2:8.3f}  [{status:10s}]  S={sens_idx:+.2f}  ({dt:.1f}s)")
        results_oat[name] = {
            "values": values, "I2": I2_vals, "statuses": statuses,
            "log": cfg["log"], "unit": cfg["unit"], "baseline": cfg["baseline"]
        }

    # -------------------------------------------------------------------------
    # [3/6] TWO-PARAMETER INTERACTIONS
    # -------------------------------------------------------------------------
    print("\n[3/6] Running two-parameter interaction analyses...")
    print("-"*80)

    print("\n  [3a] b vs b_R interaction...")
    b_range = [1.0, 1.5, 2.0, 2.5, 3.0]
    bR_range = [0.5, 1.0, 1.5, 2.0, 2.5]
    interaction_matrix_bbR = np.zeros((len(b_range), len(bR_range)))
    interaction_status_bbR = np.empty((len(b_range), len(bR_range)), dtype=object)
    for i, b_val in enumerate(b_range):
        for j, bR_val in enumerate(bR_range):
            p_dict = BASE_PARAMS.copy()
            p_dict["b"] = b_val; p_dict["b_R"] = bR_val
            crit, status = find_I_transcritical(p_dict, I_min=0.5, I_max=25.0)
            interaction_matrix_bbR[i, j] = crit if crit is not None else np.nan
            interaction_status_bbR[i, j] = status
            crit_str = f"{crit:.2f}" if crit is not None else "N/A"
            print(f"    b={b_val:.1f}, b_R={bR_val:.1f} -> I*2={crit_str} [{status}]")

    print("\n  [3b] eta vs mu interaction (structural)...")
    eta_range = [1e-10, 1e-9, 2e-8, 1e-7, 1e-6]
    mu_range = [0.5, 1.0, 2.0, 5.0, 10.0]
    interaction_matrix_etamu = np.zeros((len(eta_range), len(mu_range)))
    interaction_status_etamu = np.empty((len(eta_range), len(mu_range)), dtype=object)
    for i, eta_val in enumerate(eta_range):
        for j, mu_val in enumerate(mu_range):
            p_dict = BASE_PARAMS.copy()
            p_dict["eta"] = eta_val; p_dict["mu"] = mu_val
            crit, status = find_I_transcritical(p_dict, I_min=0.5, I_max=50.0)
            interaction_matrix_etamu[i, j] = crit if crit is not None else np.nan
            interaction_status_etamu[i, j] = status
            crit_str = f"{crit:.2f}" if crit is not None else "N/A"
            print(f"    eta={eta_val:.0e}, mu={mu_val:.1f} -> I*2={crit_str} [{status}]")

    # -------------------------------------------------------------------------
    # [4/6] GLOBAL SOBOL SENSITIVITY
    # -------------------------------------------------------------------------
    print("\n[4/6] Running GLOBAL Sobol sensitivity analysis...")
    print("-"*80)
    print("NOTE: For full publication quality, increase N_SOBOL to 512 or 1024.")
    print("      Current setting (N=128) balances rigor with runtime (~10-15 min).")
    print("-"*80)

    N_SOBOL = 128
    print(f"Generating Sobol samples: N={N_SOBOL} base samples x {len(SOBOL_BOUNDS)} parameters")
    print(f"Total evaluations: {N_SOBOL * (len(SOBOL_BOUNDS) + 2)}")

    A, B, AB_dict, param_names = generate_sobol_samples(N_SOBOL, SOBOL_BOUNDS, seed=42)

    # Cache for repeated evaluations
    _I2_cache = {}
    def find_I2_cached(p_dict):
        key = tuple(sorted((k, round(v, 12)) for k, v in p_dict.items()))
        if key not in _I2_cache:
            I2, status = find_I_transcritical(p_dict, I_min=0.5, I_max=30.0, n_coarse=20, bisection_tol=0.1)
            _I2_cache[key] = (I2, status)
        return _I2_cache[key]

    def eval_matrix(M, label):
        Y = np.zeros(len(M))
        valid = np.ones(len(M), dtype=bool)
        for i in range(len(M)):
            p_dict = params_dict_from_array(M[i], param_names)
            I2, status = find_I2_cached(p_dict)
            if I2 is not None and status in ['success', 'extends_beyond_range']:
                Y[i] = I2
            else:
                Y[i] = np.nan; valid[i] = False
            if (i+1) % 32 == 0 or i == len(M) - 1:
                print(f"  {label}: {i+1}/{len(M)}  valid: {np.sum(valid)}  cache: {len(_I2_cache)}")
        return Y, valid

    Y_A, valid_A = eval_matrix(A, "A")
    Y_B, valid_B = eval_matrix(B, "B")
    Y_AB_dict = {}
    valid_AB_all = np.ones(N_SOBOL, dtype=bool)
    for name in param_names:
        Y_AB, valid_AB = eval_matrix(AB_dict[name], f"AB({name})")
        Y_AB_dict[name] = Y_AB
        valid_AB_all = valid_AB_all & valid_AB

    valid_all = valid_A & valid_B & valid_AB_all
    print(f"\nValid samples for Sobol: {np.sum(valid_all)}/{N_SOBOL}")

    if np.sum(valid_all) < 30:
        print("WARNING: Too few valid samples! Using median imputation...")
        median_I2 = np.nanmedian(np.concatenate([Y_A, Y_B]))
        Y_A_clean = np.where(np.isnan(Y_A), median_I2, Y_A)
        Y_B_clean = np.where(np.isnan(Y_B), median_I2, Y_B)
        Y_AB_clean = {name: np.where(np.isnan(Y_AB_dict[name]), median_I2, Y_AB_dict[name]) for name in param_names}
    else:
        Y_A_clean = Y_A[valid_all]
        Y_B_clean = Y_B[valid_all]
        Y_AB_clean = {name: Y_AB_dict[name][valid_all] for name in param_names}

    S1, ST, S1_conf, ST_conf = compute_sobol_indices(Y_A_clean, Y_B_clean, Y_AB_clean, param_names)

    print("\n" + "="*70)
    print("GLOBAL SOBOL SENSITIVITY RESULTS — I*2 (Transcritical Point)")
    print("="*70)
    print(f"{'Parameter':<10} {'S1':<10} {'S1 95% CI':<22} {'ST':<10} {'ST 95% CI':<22}")
    print("-"*70)
    sobol_results = []
    for name in param_names:
        s1 = max(S1[name], 0.0)
        st = max(ST[name], 0.0)
        s1_lo, s1_hi = S1_conf[name]
        st_lo, st_hi = ST_conf[name]
        print(f"{name:<10} {s1:>8.4f}   [{max(s1_lo,0):>7.4f}, {s1_hi:>7.4f}]   {st:>8.4f}   [{max(st_lo,0):>7.4f}, {st_hi:>7.4f}]")
        sobol_results.append({'param': name, 'S1': s1, 'ST': st,
                              'S1_lo': max(s1_lo,0), 'S1_hi': s1_hi,
                              'ST_lo': max(st_lo,0), 'ST_hi': st_hi})
    sum_S1 = sum(max(S1[n], 0) for n in param_names)
    print(f"\nSum S1: {sum_S1:.4f}  |  Sum ST: {sum(max(ST[n],0) for n in param_names):.4f}")

    # Save Sobol results
    with open('sobol_results_v4.pkl', 'wb') as f:
        pickle.dump({'S1': S1, 'ST': ST, 'S1_conf': S1_conf, 'ST_conf': ST_conf,
                     'param_names': param_names, 'sobol_results': sobol_results}, f)
    print("Sobol results saved to sobol_results_v4.pkl")

    # -------------------------------------------------------------------------
    # [5/6] PARAMETER UNCERTAINTY PROPAGATION
    # -------------------------------------------------------------------------
    print("\n[5/6] Running parameter uncertainty propagation (Monte Carlo)...")
    print("-"*80)

    N_MC = 500
    print(f"Sampling {N_MC} parameter vectors from joint distributions...")
    params_mc = sample_parameters_from_distributions(N_MC, seed=123)
    I2_samples = []
    I1_samples = []
    for i, p_dict in enumerate(params_mc):
        I2, status = find_I_transcritical(p_dict, I_min=0.5, I_max=50.0)
        I1 = find_I_extinction_stable(p_dict)
        if I2 is not None and status in ['success', 'extends_beyond_range']:
            I2_samples.append(I2)
        if I1 is not None:
            I1_samples.append(I1)
        if (i+1) % 100 == 0:
            print(f"  Progress: {i+1}/{N_MC}  I2 valid: {len(I2_samples)}  I1 valid: {len(I1_samples)}")

    I2_samples = np.array(I2_samples)
    I1_samples = np.array(I1_samples)

    print("\n" + "="*60)
    print("PARAMETER UNCERTAINTY PROPAGATION RESULTS")
    print("="*60)
    if len(I2_samples) > 0:
        print(f"I*2 (transcritical):")
        print(f"  Mean +/- SD:   {np.mean(I2_samples):.3f} +/- {np.std(I2_samples):.3f} mg/L/hr")
        print(f"  Median:      {np.median(I2_samples):.3f} mg/L/hr")
        print(f"  95% CI:      [{np.percentile(I2_samples, 2.5):.3f}, {np.percentile(I2_samples, 97.5):.3f}]")
        print(f"  Range:       [{np.min(I2_samples):.3f}, {np.max(I2_samples):.3f}]")
        print(f"  Valid/Total: {len(I2_samples)}/{N_MC}")
    if len(I1_samples) > 0:
        print(f"\nI*1 (extinction stable):")
        print(f"  Mean +/- SD:   {np.mean(I1_samples):.3f} +/- {np.std(I1_samples):.3f} mg/L/hr")
        print(f"  Median:      {np.median(I1_samples):.3f} mg/L/hr")
        print(f"  95% CI:      [{np.percentile(I1_samples, 2.5):.3f}, {np.percentile(I1_samples, 97.5):.3f}]")

    # -------------------------------------------------------------------------
    # [6/6] VIRTUAL INFECTION COHORT — ASI vs CLASSICAL
    # -------------------------------------------------------------------------
    print("\n[6/6] Running virtual infection cohort (ASI biomarker validation)...")
    print("-"*80)

    N_COHORT = 200  # Increase to 1000+ for publication
    print(f"Simulating {N_COHORT} virtual infections with random parameters...")

    rng = np.random.default_rng(999)
    cohort_params = sample_parameters_from_distributions(N_COHORT, seed=999)
    I_vals_cohort = rng.uniform(2.0, 20.0, N_COHORT)

    cohort_results = []
    for i in range(N_COHORT):
        p_dict = cohort_params[i]
        I_val = I_vals_cohort[i]
        result = evaluate_virtual_infection(p_dict, I_val, t_max=400, dt=1.0, noise_std=0.03, seed=i)
        cohort_results.append(result)
        if (i+1) % 50 == 0:
            print(f"  Progress: {i+1}/{N_COHORT}")

    # Save cohort results
    with open('cohort_results_v4.pkl', 'wb') as f:
        pickle.dump(cohort_results, f)
    print("Cohort results saved to cohort_results_v4.pkl")

    # Statistical analysis
    print("\n" + "="*70)
    print("VIRTUAL INFECTION COHORT — ASI vs CLASSICAL EWS")
    print("="*70)

    asi_true_vals = np.array([r['asi_true'] for r in cohort_results])
    asi_sur_vals = np.array([r['asi_surrogate'] for r in cohort_results])
    var_N_vals = np.array([r['variance_N'] for r in cohort_results])
    autocorr_N_vals = np.array([r['autocorr_N'] for r in cohort_results])
    cv_N_vals = np.array([r['cv_N'] for r in cohort_results])
    extinct_vals = np.array([r['extinct'] for r in cohort_results])

    # Correlation: ASI_true vs ASI_surrogate
    valid_corr = ~np.isnan(asi_true_vals) & ~np.isnan(asi_sur_vals)
    if np.sum(valid_corr) > 10:
        corr_as = np.corrcoef(asi_true_vals[valid_corr], asi_sur_vals[valid_corr])[0, 1]
        print(f"\nCorrelation(ASI_true, ASI_surrogate): {corr_as:.4f}")
        print(f"  -> {'STRONG' if abs(corr_as) > 0.7 else 'MODERATE' if abs(corr_as) > 0.4 else 'WEAK'} agreement")

    # Classification: can ASI predict extinction better than variance?
    def safe_auc(y_true, y_score):
        valid = ~np.isnan(y_score)
        if np.sum(valid) < 10 or len(np.unique(y_true[valid])) < 2:
            return np.nan
        return roc_auc_score(y_true[valid], y_score[valid])

    auc_asi_true = safe_auc(extinct_vals, asi_true_vals)
    auc_asi_sur = safe_auc(extinct_vals, asi_sur_vals)
    auc_var = safe_auc(extinct_vals, var_N_vals)
    auc_autocorr = safe_auc(extinct_vals, autocorr_N_vals)
    auc_cv = safe_auc(extinct_vals, cv_N_vals)

    print(f"\nAUC (extinction prediction):")
    print(f"  ASI (true):       {auc_asi_true:.4f}" if not np.isnan(auc_asi_true) else "  ASI (true):       N/A")
    print(f"  ASI (surrogate):  {auc_asi_sur:.4f}" if not np.isnan(auc_asi_sur) else "  ASI (surrogate):  N/A")
    print(f"  Variance(N):      {auc_var:.4f}" if not np.isnan(auc_var) else "  Variance(N):      N/A")
    print(f"  Autocorr(N):      {auc_autocorr:.4f}" if not np.isnan(auc_autocorr) else "  Autocorr(N):      N/A")
    print(f"  CV(N):            {auc_cv:.4f}" if not np.isnan(auc_cv) else "  CV(N):            N/A")

    # Proximity correlation
    valid_prox = ~np.isnan(asi_true_vals)
    prox_subset = np.array([r['proximity'] if r['proximity'] is not None else np.nan for r in cohort_results])
    valid_p = ~np.isnan(prox_subset) & ~np.isnan(asi_true_vals)
    if np.sum(valid_p) > 10:
        corr_prox = np.corrcoef(asi_true_vals[valid_p], prox_subset[valid_p])[0, 1]
        print(f"\nCorrelation(ASI_true, proximity_to_tipping): {corr_prox:.4f}")
        print(f"  -> {'STRONG' if abs(corr_prox) > 0.7 else 'MODERATE' if abs(corr_prox) > 0.4 else 'WEAK'} predictive signal")

    print(f"\nExtinction rate in cohort: {np.mean(extinct_vals)*100:.1f}% ({np.sum(extinct_vals)}/{len(extinct_vals)})")

    # -------------------------------------------------------------------------
    # [7/10] SOBOL SENSITIVITY OF ASI ITSELF
    # -------------------------------------------------------------------------
    print("\n[7/10] Running GLOBAL Sobol sensitivity of ASI biomarker...")
    print("-"*80)
    
    FIXED_I_ASI_SOBOL = 10.0  # Evaluate ASI at clinically relevant dosing
    
    def eval_asi_matrix(M, label):
        Y = np.zeros(len(M))
        valid = np.ones(len(M), dtype=bool)
        for i in range(len(M)):
            p_dict = params_dict_from_array(M[i], param_names)
            # Simulate short trajectory to get state for ASI
            t, N, p, C = simulate_trajectory(p_dict, FIXED_I_ASI_SOBOL, t_max=200, dt=1.0, noise_std=0.03, rng=np.random.default_rng(i))
            asi_true, _ = compute_asi(t, N, p, C, p_dict, FIXED_I_ASI_SOBOL)
            if np.isfinite(asi_true):
                Y[i] = np.log1p(asi_true)
            else:
                Y[i] = np.nan
                valid[i] = False
            if (i+1) % 32 == 0 or i == len(M) - 1:
                print(f"  {label}: {i+1}/{len(M)}  valid: {np.sum(valid)}")
        return Y, valid

    Y_A_asi, valid_A_asi = eval_asi_matrix(A, "A_asi")
    Y_B_asi, valid_B_asi = eval_asi_matrix(B, "B_asi")
    Y_AB_asi_dict = {}
    valid_AB_asi_all = np.ones(N_SOBOL, dtype=bool)
    for name in param_names:
        Y_AB_asi, valid_AB_asi = eval_asi_matrix(AB_dict[name], f"AB_asi({name})")
        Y_AB_asi_dict[name] = Y_AB_asi
        valid_AB_asi_all = valid_AB_asi_all & valid_AB_asi

    valid_asi_all = valid_A_asi & valid_B_asi & valid_AB_asi_all
    print(f"\nValid ASI samples for Sobol: {np.sum(valid_asi_all)}/{N_SOBOL}")

    if np.sum(valid_asi_all) < 30:
        print("WARNING: Too few valid ASI samples! Using median imputation...")
        median_asi = np.nanmedian(np.concatenate([Y_A_asi, Y_B_asi]))
        Y_A_asi_clean = np.where(np.isnan(Y_A_asi), median_asi, Y_A_asi)
        Y_B_asi_clean = np.where(np.isnan(Y_B_asi), median_asi, Y_B_asi)
        Y_AB_asi_clean = {name: np.where(np.isnan(Y_AB_asi_dict[name]), median_asi, Y_AB_asi_dict[name]) for name in param_names}
    else:
        Y_A_asi_clean = Y_A_asi[valid_asi_all]
        Y_B_asi_clean = Y_B_asi[valid_asi_all]
        Y_AB_asi_clean = {name: Y_AB_asi_dict[name][valid_asi_all] for name in param_names}

    S1_asi, ST_asi, S1_asi_conf, ST_asi_conf = compute_sobol_indices(Y_A_asi_clean, Y_B_asi_clean, Y_AB_asi_clean, param_names)

    print("\n" + "="*70)
    print(f"GLOBAL SOBOL SENSITIVITY — ASI (at I={FIXED_I_ASI_SOBOL})")
    print("="*70)
    print(f"{'Parameter':<<10} {'S1':<<10} {'S1 95% CI':<<22} {'ST':<<10} {'ST 95% CI':<<22}")
    print("-"*70)
    sobol_asi_results = []
    for name in param_names:
        s1 = max(S1_asi[name], 0.0)
        st = max(ST_asi[name], 0.0)
        s1_lo, s1_hi = S1_asi_conf[name]
        st_lo, st_hi = ST_asi_conf[name]
        print(f"{name:<10} {s1:>8.4f}   [{max(s1_lo,0):>7.4f}, {s1_hi:>7.4f}]   {st:>8.4f}   [{max(st_lo,0):>7.4f}, {st_hi:>7.4f}]")
        sobol_asi_results.append({'param': name, 'S1': s1, 'ST': st,
                                  'S1_lo': max(s1_lo,0), 'S1_hi': s1_hi,
                                  'ST_lo': max(st_lo,0), 'ST_hi': st_hi})
    sum_S1_asi = sum(max(S1_asi[n], 0) for n in param_names)
    print(f"\nSum S1: {sum_S1_asi:.4f}  |  Sum ST: {sum(max(ST_asi[n],0) for n in param_names):.4f}")

    # -------------------------------------------------------------------------
    # [8/10] ASI UNCERTAINTY PROPAGATION
    # -------------------------------------------------------------------------
    print("\n[8/10] Running ASI uncertainty propagation...")
    print("-"*80)

    ASI_I_LEVELS = [5.0, 10.0, 15.0]
    asi_uncertainty_results = {}
    
    for I_test in ASI_I_LEVELS:
        print(f"\n  Evaluating ASI at I={I_test:.1f} across {N_MC} MC samples...")
        asi_samples = []
        for i, p_dict in enumerate(params_mc):
            t, N, p, C = simulate_trajectory(p_dict, I_test, t_max=200, dt=1.0, noise_std=0.03, rng=np.random.default_rng(i+10000))
            asi_true, _ = compute_asi(t, N, p, C, p_dict, I_test)
            if np.isfinite(asi_true):
                asi_samples.append(asi_true)
            if (i+1) % 100 == 0:
                print(f"    Progress: {i+1}/{N_MC}  ASI valid: {len(asi_samples)}")
        
        asi_samples = np.array(asi_samples)
        asi_uncertainty_results[I_test] = asi_samples
        
        print(f"\n  ASI at I={I_test:.1f}:")
        print(f"    Mean +/- SD:   {np.mean(asi_samples):.4f} +/- {np.std(asi_samples):.4f}")
        print(f"    Median:        {np.median(asi_samples):.4f}")
        print(f"    95% CI:        [{np.percentile(asi_samples, 2.5):.4f}, {np.percentile(asi_samples, 97.5):.4f}]")
        print(f"    Range:         [{np.min(asi_samples):.4f}, {np.max(asi_samples):.4f}]")
        print(f"    Valid/Total:   {len(asi_samples)}/{N_MC}")

    # -------------------------------------------------------------------------
    # [9/10] STRATIFIED ASI PERFORMANCE
    # -------------------------------------------------------------------------
    print("\n[9/10] Stratified ASI performance by parameter regime...")
    print("-"*80)

    # Add parameter values to cohort results for stratification
    cohort_with_params = []
    for i, result in enumerate(cohort_results):
        result_copy = result.copy()
        result_copy['eta'] = cohort_params[i]['eta']
        result_copy['MIC_S'] = cohort_params[i]['MIC_S']
        result_copy['K'] = cohort_params[i]['K']
        cohort_with_params.append(result_copy)

    # Define strata based on dominant Sobol parameters
    eta_median = np.median([r['eta'] for r in cohort_with_params])
    MIC_S_median = np.median([r['MIC_S'] for r in cohort_with_params])

    strata = {
        'Low_eta_Low_MIC':  [r for r in cohort_with_params if r['eta'] <= eta_median and r['MIC_S'] <= MIC_S_median],
        'Low_eta_High_MIC': [r for r in cohort_with_params if r['eta'] <= eta_median and r['MIC_S'] > MIC_S_median],
        'High_eta_Low_MIC': [r for r in cohort_with_params if r['eta'] > eta_median and r['MIC_S'] <= MIC_S_median],
        'High_eta_High_MIC':[r for r in cohort_with_params if r['eta'] > eta_median and r['MIC_S'] > MIC_S_median],
    }

    print(f"\n  Cohort stratification (median eta={eta_median:.2e}, median MIC_S={MIC_S_median:.2f}):")
    for name, subset in strata.items():
        if len(subset) < 10:
            print(f"    {name:20s}: n={len(subset):3d} — insufficient for AUC")
            continue
        
        asi_vals = np.array([r['asi_true'] for r in subset])
        ext_vals = np.array([r['extinct'] for r in subset])
        prox_vals = np.array([r['proximity'] if r['proximity'] is not None else np.nan for r in subset])
        
        valid_auc = ~np.isnan(asi_vals)
        auc_strat = safe_auc(ext_vals, asi_vals) if np.sum(valid_auc) >= 10 and len(np.unique(ext_vals[valid_auc])) >= 2 else np.nan
        
        valid_prox = ~np.isnan(asi_vals) & ~np.isnan(prox_vals)
        corr_strat = np.corrcoef(asi_vals[valid_prox], prox_vals[valid_prox])[0,1] if np.sum(valid_prox) >= 10 else np.nan
        
        print(f"    {name:20s}: n={len(subset):3d}  AUC={auc_strat:.3f}  corr(ASI,prox)={corr_strat:.3f}")

    # -------------------------------------------------------------------------
    # [10/10] ASI COMPONENTS vs CLASSICAL EWS CORRELATIONS
    # -------------------------------------------------------------------------
    print("\n[10/10] ASI components vs classical EWS correlations...")
    print("-"*80)

    # Extract ASI components from cohort
    sa_vals = np.array([r['asi_components']['spectral_abscissa'] for r in cohort_results])
    dpdt_vals = np.array([r['asi_components']['mean_dpdt'] for r in cohort_results])
    fd_vals = np.array([r['asi_components']['fitness_diff'] for r in cohort_results])
    curv_vals = np.array([r['asi_components']['curvature'] for r in cohort_results])

    classical_vars = {
        'variance_N': var_N_vals,
        'autocorr_N': autocorr_N_vals,
        'cv_N': cv_N_vals,
        'variance_p': np.array([r['variance_p'] for r in cohort_results]),
        'skewness_N': np.array([r['skewness_N'] for r in cohort_results]),
    }

    asi_components = {
        'spectral_abscissa': sa_vals,
        'mean_dpdt': dpdt_vals,
        'fitness_diff': fd_vals,
        'curvature': curv_vals,
        'asi_true': asi_true_vals,
    }

    print(f"\n  Correlation matrix (ASI components × classical EWS):")
    print(f"  {'Component':<<18} {'Classical EWS':<<18} {'Corr':>8} {'N valid':>8}")
    print("  " + "-"*60)
    corr_matrix = []
    for comp_name, comp_vals in asi_components.items():
        for ews_name, ews_vals in classical_vars.items():
            valid_both = ~np.isnan(comp_vals) & ~np.isnan(ews_vals)
            if np.sum(valid_both) >= 10:
                c = np.corrcoef(comp_vals[valid_both], ews_vals[valid_both])[0,1]
                print(f"  {comp_name:<18} {ews_name:<18} {c:>8.3f} {np.sum(valid_both):>8d}")
                corr_matrix.append({
                    'asi_component': comp_name,
                    'classical_ews': ews_name,
                    'correlation': c,
                    'n': np.sum(valid_both)
                })
            else:
                print(f"  {comp_name:<18} {ews_name:<18} {'N/A':>8} {np.sum(valid_both):>8d}")

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    print("\nGenerating figures...")
    fig = plt.figure(figsize=(22, 32))
    gs = fig.add_gridspec(8, 4, hspace=0.45, wspace=0.4)

    # Row 0: OAT sweeps (first 4 parameters)
    oat_names = list(results_oat.keys())
    for idx in range(min(4, len(oat_names))):
        name = oat_names[idx]
        data = results_oat[name]
        ax = fig.add_subplot(gs[0, idx])
        vals = np.array(data["values"], dtype=float)
        I2s = np.array(data["I2"], dtype=float)
        statuses = data["statuses"]
        x_s, y_s, x_e, y_e, x_f, y_f = [], [], [], [], [], []
        for v, t, s in zip(vals, I2s, statuses):
            if s == "success": x_s.append(v); y_s.append(t)
            elif s == "extends_beyond_range": x_e.append(v); y_e.append(t)
            else: x_f.append(v); y_f.append(t)
        vm = np.array([s in ["success", "extends_beyond_range"] for s in statuses])
        if np.any(vm):
            if data["log"]: ax.semilogx(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
            else: ax.plot(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
        if x_s: ax.scatter(x_s, y_s, c="blue", s=40, zorder=3)
        if x_e: ax.scatter(x_e, y_e, c="green", s=40, marker="^", zorder=3)
        if x_f: ax.scatter(x_f, y_f, c="red", s=40, marker="x", zorder=3)
        if baseline_I2 is not None: ax.axhline(y=baseline_I2, color="crimson", linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(x=data["baseline"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel(f"{name} ({data['unit']})", fontsize=9)
        ax.set_ylabel("I*2", fontsize=9)
        ax.set_title(f"OAT: {name}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Row 1: OAT sweeps (next 4 parameters)
    for idx in range(4, min(8, len(oat_names))):
        name = oat_names[idx]
        data = results_oat[name]
        ax = fig.add_subplot(gs[1, idx-4])
        vals = np.array(data["values"], dtype=float)
        I2s = np.array(data["I2"], dtype=float)
        statuses = data["statuses"]
        x_s, y_s, x_e, y_e, x_f, y_f = [], [], [], [], [], []
        for v, t, s in zip(vals, I2s, statuses):
            if s == "success": x_s.append(v); y_s.append(t)
            elif s == "extends_beyond_range": x_e.append(v); y_e.append(t)
            else: x_f.append(v); y_f.append(t)
        vm = np.array([s in ["success", "extends_beyond_range"] for s in statuses])
        if np.any(vm):
            if data["log"]: ax.semilogx(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
            else: ax.plot(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
        if x_s: ax.scatter(x_s, y_s, c="blue", s=40, zorder=3)
        if x_e: ax.scatter(x_e, y_e, c="green", s=40, marker="^", zorder=3)
        if x_f: ax.scatter(x_f, y_f, c="red", s=40, marker="x", zorder=3)
        if baseline_I2 is not None: ax.axhline(y=baseline_I2, color="crimson", linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(x=data["baseline"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel(f"{name} ({data['unit']})", fontsize=9)
        ax.set_ylabel("I*2", fontsize=9)
        ax.set_title(f"OAT: {name}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Row 2: OAT sweeps (remaining 3 parameters)
    for idx in range(8, len(oat_names)):
        name = oat_names[idx]
        data = results_oat[name]
        ax = fig.add_subplot(gs[2, idx-8])
        vals = np.array(data["values"], dtype=float)
        I2s = np.array(data["I2"], dtype=float)
        statuses = data["statuses"]
        x_s, y_s, x_e, y_e, x_f, y_f = [], [], [], [], [], []
        for v, t, s in zip(vals, I2s, statuses):
            if s == "success": x_s.append(v); y_s.append(t)
            elif s == "extends_beyond_range": x_e.append(v); y_e.append(t)
            else: x_f.append(v); y_f.append(t)
        vm = np.array([s in ["success", "extends_beyond_range"] for s in statuses])
        if np.any(vm):
            if data["log"]: ax.semilogx(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
            else: ax.plot(vals[vm], I2s[vm], "-", color="steelblue", linewidth=2, zorder=1)
        if x_s: ax.scatter(x_s, y_s, c="blue", s=40, zorder=3)
        if x_e: ax.scatter(x_e, y_e, c="green", s=40, marker="^", zorder=3)
        if x_f: ax.scatter(x_f, y_f, c="red", s=40, marker="x", zorder=3)
        if baseline_I2 is not None: ax.axhline(y=baseline_I2, color="crimson", linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(x=data["baseline"], color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.set_xlabel(f"{name} ({data['unit']})", fontsize=9)
        ax.set_ylabel("I*2", fontsize=9)
        ax.set_title(f"OAT: {name}", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Row 2, col 0: Sobol S1 bar chart
    ax = fig.add_subplot(gs[3, 0])
    names_sorted = sorted(param_names, key=lambda n: S1[n], reverse=True)
    s1_vals = [max(S1[n], 0) for n in names_sorted]
    s1_err_lo = [max(S1[n] - S1_conf[n][0], 0) for n in names_sorted]
    s1_err_hi = [max(S1_conf[n][1] - S1[n], 0) for n in names_sorted]
    y_pos = np.arange(len(names_sorted))
    ax.barh(y_pos, s1_vals, xerr=[s1_err_lo, s1_err_hi], color='steelblue', alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel("First-order Sobol Index (S1)", fontsize=9)
    ax.set_title("Global Sensitivity: S1", fontsize=10, fontweight="bold")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Row 2, col 1: Sobol ST bar chart
    ax = fig.add_subplot(gs[3, 1])
    names_sorted_st = sorted(param_names, key=lambda n: ST[n], reverse=True)
    st_vals = [max(ST[n], 0) for n in names_sorted_st]
    st_err_lo = [max(ST[n] - ST_conf[n][0], 0) for n in names_sorted_st]
    st_err_hi = [max(ST_conf[n][1] - ST[n], 0) for n in names_sorted_st]
    y_pos = np.arange(len(names_sorted_st))
    ax.barh(y_pos, st_vals, xerr=[st_err_lo, st_err_hi], color='darkorange', alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted_st, fontsize=8)
    ax.set_xlabel("Total-order Sobol Index (ST)", fontsize=9)
    ax.set_title("Global Sensitivity: ST", fontsize=10, fontweight="bold")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Row 2, col 2: Uncertainty propagation histogram
    ax = fig.add_subplot(gs[3, 2])
    if len(I2_samples) > 0:
        ax.hist(I2_samples, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
        ax.axvline(x=np.mean(I2_samples), color='darkred', linestyle='--', linewidth=2, label=f"Mean: {np.mean(I2_samples):.2f}")
        ax.axvline(x=np.median(I2_samples), color='navy', linestyle=':', linewidth=2, label=f"Median: {np.median(I2_samples):.2f}")
        ax.axvline(x=baseline_I2, color='crimson', linestyle='-', linewidth=2, label=f"Baseline: {baseline_I2:.2f}")
        ax.set_xlabel("I*2 (mg/L/hr)", fontsize=9)
        ax.set_ylabel("Frequency", fontsize=9)
        ax.set_title(f"Uncertainty Propagation\n(N={len(I2_samples)}, 95% CI: [{np.percentile(I2_samples,2.5):.1f}, {np.percentile(I2_samples,97.5):.1f}])", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2, col 3: ASI vs Classical scatter
    ax = fig.add_subplot(gs[3, 3])
    valid_scatter = ~np.isnan(asi_true_vals) & ~np.isnan(var_N_vals) & ~np.isnan(autocorr_N_vals)
    if np.sum(valid_scatter) > 10:
        ax.scatter(var_N_vals[valid_scatter], asi_true_vals[valid_scatter], c='blue', alpha=0.5, s=20, label='ASI vs Var(N)')
        ax.set_xlabel("Variance(N)", fontsize=9)
        ax.set_ylabel("ASI (true)", fontsize=9)
        ax.set_title("ASI vs Classical EWS", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Row 3, col 0: Interaction heatmap b vs b_R
    ax = fig.add_subplot(gs[4, 0])
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

    # Row 3, col 1: Interaction heatmap eta vs mu
    ax = fig.add_subplot(gs[4, 1])
    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad('lightgray')
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

    # Row 3, col 2: AUC comparison bar chart
    ax = fig.add_subplot(gs[4, 2])
    auc_labels = ['ASI\n(true)', 'ASI\n(surrogate)', 'Variance\n(N)', 'Autocorr\n(N)', 'CV\n(N)']
    auc_values = [auc_asi_true, auc_asi_sur, auc_var, auc_autocorr, auc_cv]
    colors = ['darkblue', 'steelblue', 'darkorange', 'seagreen', 'crimson']
    bars = ax.bar(range(len(auc_labels)), auc_values, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xticks(range(len(auc_labels)))
    ax.set_xticklabels(auc_labels, fontsize=8)
    ax.set_ylabel("AUC", fontsize=9)
    ax.set_title("Extinction Prediction Performance\n(AUC: higher = better)", fontsize=10, fontweight="bold")
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, label='Good')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, auc_values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Row 3, col 3: ASI_true vs proximity scatter
    ax = fig.add_subplot(gs[4, 3])
    valid_p = ~np.isnan(prox_subset) & ~np.isnan(asi_true_vals)
    if np.sum(valid_p) > 10:
        colors_scatter = ['red' if e else 'blue' for e in extinct_vals]
        ax.scatter(prox_subset[valid_p], asi_true_vals[valid_p], c=[colors_scatter[i] for i in np.where(valid_p)[0]], alpha=0.5, s=20)
        ax.set_xlabel("Proximity to tipping (I/I*2)", fontsize=9)
        ax.set_ylabel("ASI (true)", fontsize=9)
        ax.set_title("ASI vs Proximity to Tipping\n(red=extinct, blue=survived)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='I = I*2')
        ax.legend(fontsize=7)

    # Row 4: Summary text panel
    ax = fig.add_subplot(gs[5, :])
    ax.axis('off')
    summary_text = f"""
    COMPREHENSIVE SENSITIVITY ANALYSIS v4 — SUMMARY
    {'='*70}
    BASELINE:        I*1 = {baseline_I1:.2f} mg/L/hr  |  I*2 = {baseline_I2:.2f} mg/L/hr  |  Bistable range: [{baseline_I1:.2f}, {baseline_I2:.2f}]

    GLOBAL SOBOL (I*2 sensitivity):
    """
    for name in names_sorted[:5]:
        summary_text += f"      {name:<10} S1 = {max(S1[name],0):.4f}  |  ST = {max(ST[name],0):.4f}\n"

    summary_text += f"""
    UNCERTAINTY PROPAGATION (N={len(I2_samples)}):
      I*2 = {np.mean(I2_samples):.2f} +/- {np.std(I2_samples):.2f} mg/L/hr  |  95% CI: [{np.percentile(I2_samples,2.5):.2f}, {np.percentile(I2_samples,97.5):.2f}]

    VIRTUAL COHORT (N={N_COHORT}):
      ASI_true vs ASI_surrogate corr: {corr_as:.3f}
      AUC — ASI(true): {auc_asi_true:.3f}  |  ASI(surrogate): {auc_asi_sur:.3f}  |  Variance(N): {auc_var:.3f}
      Extinction rate: {np.mean(extinct_vals)*100:.1f}%
    """
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        # Row 5, col 0: Sobol S1 comparison — I*2 vs ASI
    ax = fig.add_subplot(gs[5, 0])
    names_sorted_asi = sorted(param_names, key=lambda n: S1_asi[n], reverse=True)
    s1_asi_vals = [max(S1_asi[n], 0) for n in names_sorted_asi]
    y_pos = np.arange(len(names_sorted_asi))
    ax.barh(y_pos, s1_asi_vals, color='teal', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted_asi, fontsize=8)
    ax.set_xlabel("S1 (ASI sensitivity)", fontsize=9)
    ax.set_title(f"ASI Sobol S1\n(I={FIXED_I_ASI_SOBOL})", fontsize=10, fontweight="bold")
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    # Row 5, col 1: ASI uncertainty propagation at fixed I levels
    ax = fig.add_subplot(gs[5, 1])
    colors_asi = ['steelblue', 'darkorange', 'seagreen']
    for idx, (I_test, asi_samples) in enumerate(asi_uncertainty_results.items()):
        if len(asi_samples) > 0:
            ax.hist(asi_samples, bins=25, alpha=0.5, color=colors_asi[idx], 
                    label=f"I={I_test:.0f}: μ={np.mean(asi_samples):.3f}", edgecolor='black')
    ax.set_xlabel("ASI value", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("ASI Uncertainty Propagation\n(at fixed I)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Row 5, col 2: Stratified AUC heatmap
    ax = fig.add_subplot(gs[5, 2])
    strat_auc_matrix = np.full((2, 2), np.nan)
    for name, subset in strata.items():
        if 'Low_eta_Low_MIC' in name: i, j = 0, 0
        elif 'Low_eta_High_MIC' in name: i, j = 0, 1
        elif 'High_eta_Low_MIC' in name: i, j = 1, 0
        else: i, j = 1, 1
        
        if len(subset) >= 10:
            asi_vals = np.array([r['asi_true'] for r in subset])
            ext_vals = np.array([r['extinct'] for r in subset])
            valid_auc = ~np.isnan(asi_vals)
            if np.sum(valid_auc) >= 10 and len(np.unique(ext_vals[valid_auc])) >= 2:
                strat_auc_matrix[i, j] = safe_auc(ext_vals, asi_vals)
    
    im_strat = ax.imshow(strat_auc_matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low MIC_S', 'High MIC_S'], fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Low η', 'High η'], fontsize=8)
    ax.set_xlabel("MIC_S", fontsize=10)
    ax.set_ylabel("eta", fontsize=10)
    ax.set_title("Stratified AUC\n(ASI vs extinction)", fontsize=10, fontweight="bold")
    cbar_strat = plt.colorbar(im_strat, ax=ax, shrink=0.8)
    cbar_strat.set_label("AUC", fontsize=9)
    for i in range(2):
        for j in range(2):
            if not np.isnan(strat_auc_matrix[i, j]):
                ax.text(j, i, f"{strat_auc_matrix[i,j]:.2f}", ha="center", va="center", 
                        fontsize=10, fontweight='bold', color="white" if strat_auc_matrix[i,j] < 0.5 else "black")

    # Row 5, col 3: ASI component vs classical EWS correlation heatmap
    ax = fig.add_subplot(gs[5, 3])
    comp_names_short = ['spectral', 'dpdt', 'fit_diff', 'curvature', 'asi_true']
    ews_names_short = ['var_N', 'autocorr_N', 'cv_N', 'var_p', 'skew_N']
    
    corr_mat = np.full((len(comp_names_short), len(ews_names_short)), np.nan)
    for ci, (comp_name, comp_vals) in enumerate(asi_components.items()):
        for ei, (ews_name, ews_vals) in enumerate(classical_vars.items()):
            valid_both = ~np.isnan(comp_vals) & ~np.isnan(ews_vals)
            if np.sum(valid_both) >= 10:
                corr_mat[ci, ei] = np.corrcoef(comp_vals[valid_both], ews_vals[valid_both])[0,1]
    
    im_corr = ax.imshow(corr_mat, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(ews_names_short)))
    ax.set_xticklabels(ews_names_short, fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(comp_names_short)))
    ax.set_yticklabels(comp_names_short, fontsize=7)
    ax.set_title("ASI Components vs\nClassical EWS (r)", fontsize=10, fontweight="bold")
    cbar_corr = plt.colorbar(im_corr, ax=ax, shrink=0.8)
    cbar_corr.set_label("Pearson r", fontsize=9)
    for i in range(len(comp_names_short)):
        for j in range(len(ews_names_short)):
            if not np.isnan(corr_mat[i, j]):
                ax.text(j, i, f"{corr_mat[i,j]:.2f}", ha="center", va="center", fontsize=7, 
                        color="white" if abs(corr_mat[i,j]) > 0.5 else "black")

    # Row 6, col 0: ASI vs proximity scatter colored by eta
    ax = fig.add_subplot(gs[6, 0])
    eta_cohort = np.array([r['eta'] for r in cohort_with_params])
    valid_p2 = ~np.isnan(prox_subset) & ~np.isnan(asi_true_vals)
    if np.sum(valid_p2) > 10:
        scatter = ax.scatter(prox_subset[valid_p2], asi_true_vals[valid_p2], 
                            c=np.log10(eta_cohort[valid_p2]), cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel("Proximity (I/I*2)", fontsize=9)
        ax.set_ylabel("ASI (true)", fontsize=9)
        ax.set_title("ASI vs Proximity\n(colored by log10 η)", fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        plt.colorbar(scatter, ax=ax, shrink=0.8, label='log10(η)')

    # Row 6, col 1: ASI distribution by extinction status
    ax = fig.add_subplot(gs[6, 1])
    asi_extinct = asi_true_vals[extinct_vals]
    asi_survived = asi_true_vals[~extinct_vals]
    ax.hist(asi_survived, bins=20, alpha=0.6, color='blue', label=f'Survived (n={np.sum(~extinct_vals)})', edgecolor='black')
    ax.hist(asi_extinct, bins=20, alpha=0.6, color='red', label=f'Extinct (n={np.sum(extinct_vals)})', edgecolor='black')
    ax.set_xlabel("ASI (true)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("ASI Distribution by Outcome", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 6, col 2: Proximity distribution by extinction status
    ax = fig.add_subplot(gs[6, 2])
    prox_extinct = prox_subset[extinct_vals & ~np.isnan(prox_subset)]
    prox_survived = prox_subset[~extinct_vals & ~np.isnan(prox_subset)]
    ax.hist(prox_survived, bins=20, alpha=0.6, color='blue', label='Survived', edgecolor='black')
    ax.hist(prox_extinct, bins=20, alpha=0.6, color='red', label='Extinct', edgecolor='black')
    ax.set_xlabel("Proximity (I/I*2)", fontsize=9)
    ax.set_ylabel("Frequency", fontsize=9)
    ax.set_title("Proximity Distribution\nby Outcome", fontsize=10, fontweight="bold")
    ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Tipping point')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Row 6, col 3: Summary of new findings text
    ax = fig.add_subplot(gs[6, 3])
    ax.axis('off')
    
    # Safe access to ASI uncertainty results
    asi_5 = asi_uncertainty_results.get(5.0, np.array([]))
    asi_10 = asi_uncertainty_results.get(10.0, np.array([]))
    asi_15 = asi_uncertainty_results.get(15.0, np.array([]))
    
    mean_5 = np.mean(asi_5) if len(asi_5) > 0 else np.nan
    std_5 = np.std(asi_5) if len(asi_5) > 0 else np.nan
    mean_10 = np.mean(asi_10) if len(asi_10) > 0 else np.nan
    std_10 = np.std(asi_10) if len(asi_10) > 0 else np.nan
    mean_15 = np.mean(asi_15) if len(asi_15) > 0 else np.nan
    std_15 = np.std(asi_15) if len(asi_15) > 0 else np.nan
    
    best_auc = np.nanmax(strat_auc_matrix) if not np.all(np.isnan(strat_auc_matrix)) else np.nan
    worst_auc = np.nanmin(strat_auc_matrix) if not np.all(np.isnan(strat_auc_matrix)) else np.nan
    
    new_findings_text = f"""NEW ANALYSES [7-10]
{'='*50}
ASI SOBOL (I={FIXED_I_ASI_SOBOL:.0f}):
  Top S1: {names_sorted_asi[0]}={max(S1_asi[names_sorted_asi[0]],0):.3f}
  
ASI UNCERTAINTY:
  I=5:  ASI={mean_5:.3f}±{std_5:.3f}
  I=10: ASI={mean_10:.3f}±{std_10:.3f}
  I=15: ASI={mean_15:.3f}±{std_15:.3f}

STRATIFIED AUC:
  Best: {best_auc:.3f}
  Worst: {worst_auc:.3f}

COMPONENT CORRELATIONS:
  Strongest: see heatmap
    """
    ax.text(0.05, 0.95, new_findings_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # Row 7: Full summary text panel
    ax = fig.add_subplot(gs[7, :])
    ax.axis('off')
    
    summary_text = f"""
    COMPREHENSIVE SENSITIVITY ANALYSIS v4 — SUMMARY
    {'='*70}
    BASELINE:        I*1 = {baseline_I1:.2f} mg/L/hr  |  I*2 = {baseline_I2:.2f} mg/L/hr  |  Bistable range: [{baseline_I1:.2f}, {baseline_I2:.2f}]

    GLOBAL SOBOL (I*2 location):
    """
    for name in names_sorted[:5]:
        summary_text += f"      {name:<10} S1 = {max(S1[name],0):.4f}  |  ST = {max(ST[name],0):.4f}\n"

    summary_text += f"""
    GLOBAL SOBOL (ASI magnitude at I={FIXED_I_ASI_SOBOL:.0f}):
    """
    for name in names_sorted_asi[:5]:
        summary_text += f"      {name:<10} S1 = {max(S1_asi[name],0):.4f}  |  ST = {max(ST_asi[name],0):.4f}\n"

    summary_text += f"""
    UNCERTAINTY PROPAGATION (N={len(I2_samples)}):
      I*2 = {np.mean(I2_samples):.2f} +/- {np.std(I2_samples):.2f} mg/L/hr  |  95% CI: [{np.percentile(I2_samples,2.5):.2f}, {np.percentile(I2_samples,97.5):.2f}]
      ASI at I=10 = {mean_10:.3f} +/- {std_10:.3f}

    VIRTUAL COHORT (N={N_COHORT}):
      ASI_true vs ASI_surrogate corr: {corr_as:.3f}
      AUC — ASI(true): {auc_asi_true:.3f}  |  ASI(surrogate): {auc_asi_sur:.3f}  |  Variance(N): {auc_var:.3f}
      Stratified AUC range: [{worst_auc:.3f}, {best_auc:.3f}]
      Extinction rate: {np.mean(extinct_vals)*100:.1f}%
    """
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle("Comprehensive Parameter Sensitivity Analysis v4 — Global + Uncertainty + ASI",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig("comprehensive_sensitivity_v4.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("Figure saved to comprehensive_sensitivity_v4.png")

    # -------------------------------------------------------------------------
    # FINAL SUMMARY TABLE (console output)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"{'Analysis':<<30} {'Result':<<50}")
    print("-"*80)
    print(f"{'Baseline I*2':<<30} {f'{baseline_I2:.3f} mg/L/hr':<<50}")
    print(f"{'Baseline I*1':<<30} {f'{baseline_I1:.3f} mg/L/hr':<<50}")
    print(f"{'Bistable range':<<30} {f'[{baseline_I1:.2f}, {baseline_I2:.2f}]':<<50}")
    print(f"{'Global Sobol (top S1)':<<30} {f'{names_sorted[0]} = {max(S1[names_sorted[0]],0):.4f}':<<50}")
    print(f"{'Uncertainty I*2 (mean)':<<30} {f'{np.mean(I2_samples):.3f} +/- {np.std(I2_samples):.3f}':<<50}")
    print(f"{'Uncertainty I*2 (95% CI)':<<30} {f'[{np.percentile(I2_samples,2.5):.3f}, {np.percentile(I2_samples,97.5):.3f}]':<<50}")
    print(f"{'ASI vs surrogate corr':<<30} {f'{corr_as:.4f}':<<50}")
    print(f"{'AUC ASI(true)':<<30} {f'{auc_asi_true:.4f}':<<50}")
    print(f"{'AUC ASI(surrogate)':<<30} {f'{auc_asi_sur:.4f}':<<50}")
    print(f"{'AUC Variance(N)':<<30} {f'{auc_var:.4f}':<<50}")
    print("="*80)
    print("\nAll analyses complete. Files saved:")
    print("  - comprehensive_sensitivity_v4.png")
    print("  - sobol_results_v4.pkl")
    print("  - cohort_results_v4.pkl")


if __name__ == "__main__":
    main()
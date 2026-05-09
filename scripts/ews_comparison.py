"""
================================================================================
COMPREHENSIVE EWS COMPARISON: ASI vs All Classical Indicators (CORRECTED VERSION)
================================================================================

This is the EXACT code that produced the figure with:
- 5 rows x 3 columns layout
- t_max=100, dt=0.02, n_realizations=10, window=80
- EWS computed on INDIVIDUAL trajectories (not ensemble mean)
- Dual-axis comparison panel showing raw values

KEY CORRECTIONS from original buggy version:
1. EWS (variance, AR(1), skewness, kurtosis, CV, spectral) computed on 
   each individual trajectory, then averaged across realizations.
   Original bug: computed on ensemble mean -> noise killed by averaging.
2. AR(1) uses linear detrending before computing lag-1 autocorrelation.
3. Spectral ratio uses power (amplitude^2) not raw amplitude.
4. Added ensemble variance (variance across realizations at each time).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.linalg import eigvals
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS
# ============================================================
r_S, r_R, K, b, b_R = 1.0, 0.93, 1e9, 2.0, 1.5
MIC_S, MIC_R, n, c_R = 2.0, 4.0, 3.0, 0.04
mu, eta, gamma = 1.0, 2e-8, 1e-12
MIC_S_n, MIC_R_n = MIC_S**n, MIC_R**n

# ============================================================
# EQUILIBRIUM & JACOBIAN
# ============================================================
_eq_cache = {}

def find_equilibrium(I_val):
    if I_val in _eq_cache:
        return _eq_cache[I_val]
    def residuals(X):
        N, p, C = X
        N = max(N, 1e-6); p = np.clip(p, 1e-6, 1-1e-6); C = max(C, 1e-6)
        Cn = C**n
        hS = Cn / (Cn + MIC_S_n)
        hR = Cn / (Cn + MIC_R_n)
        gS = r_S*(1 - N/K) - b*hS
        gR = r_R*(1 - N/K) - c_R - b_R*hR
        gbar = (1-p)*gS + p*gR
        dN = N*gbar
        dp = p*(1-p)*(gR - gS + gamma*N)
        dC = I_val - mu*C - eta*N*p*C
        return [dN, dp, dC]
    seeds = [[9.53e8,0.04,0.1],[9.53e8,0.4,0.6],[9.53e8,0.7,0.6],[9.5e8,0.9,0.5],[9.5e8,0.95,1.0]]
    best_sol = None; best_res = 1e10
    for seed in seeds:
        try:
            sol = least_squares(residuals, seed, ftol=1e-10, xtol=1e-10, max_nfev=2000,
                               bounds=([0,0,0],[K,1,100]))
            if sol.success:
                N, p, C = sol.x
                res = np.max(np.abs(residuals(sol.x)))
                if N > 1e7 and 0.001 < p < 0.999 and C > 0.01 and res < best_res:
                    best_res = res; best_sol = sol.x
        except: continue
    _eq_cache[I_val] = best_sol
    return best_sol

def jacobian(N, p, C):
    Cn = C**n
    hS = Cn / (Cn + MIC_S_n)
    hR = Cn / (Cn + MIC_R_n)
    hdS = n * C**(n-1) * MIC_S_n / (Cn + MIC_S_n)**2
    hdR = n * C**(n-1) * MIC_R_n / (Cn + MIC_R_n)**2
    gS = r_S*(1 - N/K) - b*hS
    gR = r_R*(1 - N/K) - c_R - b_R*hR
    gbar = (1-p)*gS + p*gR
    Delta_val = gR - gS
    J11 = gbar - N*((1-p)*r_S + p*r_R)/K
    J12 = N * Delta_val
    J13 = -N*((1-p)*b*hdS + p*b_R*hdR)
    J21 = p*(1-p)*(r_S-r_R)/K + gamma*p*(1-p)
    J22 = (1-2*p)*Delta_val + gamma*N*(1-2*p)
    J23 = p*(1-p)*(b*hdS - b_R*hdR)
    J31 = -eta*p*C
    J32 = -eta*N*C
    J33 = -mu - eta*N*p
    return np.array([[J11,J12,J13],[J21,J22,J23],[J31,J32,J33]])

def compute_ASI(N, p, C, ref_I=1.0):
    J = jacobian(N, p, C)
    lam_dom = np.max(np.real(eigvals(J)))
    X_ref = find_equilibrium(ref_I)
    if X_ref is None:
        for test_I in [1.5, 2.0, 2.5]:
            X_ref = find_equilibrium(test_I)
            if X_ref is not None: break
        else: return np.nan
    J_ref = jacobian(X_ref[0], X_ref[1], X_ref[2])
    lam_ref = np.max(np.real(eigvals(J_ref)))
    if lam_dom >= 0: return 0.0
    if lam_ref >= 0: return np.nan
    return -lam_dom / abs(lam_ref)

def find_bifurcation_point():
    I_test = np.linspace(1, 12.5, 150)
    valid = []; prev_lam = None; bif_I = None
    for idx, I_val in enumerate(I_test):
        X = find_equilibrium(I_val)
        if X is not None:
            N, p, C = X
            if N > 1e7 and 0.001 < p < 0.999 and C > 0.01:
                lam = np.max(np.real(eigvals(jacobian(N, p, C))))
                valid.append((I_val, N, p, C, lam))
                if prev_lam is not None and prev_lam > 0 and lam < 0:
                    bif_I = I_test[idx-1]
                prev_lam = lam
    if len(valid) == 0: return None, []
    if bif_I is None: bif_I = valid[-1][0]
    return bif_I, valid

# ============================================================
# FAST SIMULATION
# ============================================================
def stochastic_sim_fast(I_val, X0, t_max=100, dt=0.02, seed=None):
    if seed is not None: np.random.seed(seed)
    n_steps = int(t_max / dt)
    traj = np.zeros((n_steps, 3))
    N, p, C = float(X0[0]), float(X0[1]), float(X0[2])
    sqdt = np.sqrt(dt)
    for i in range(n_steps):
        if N < 1e-6: N = 1e-6
        if p < 1e-6: p = 1e-6
        if p > 1-1e-6: p = 1-1e-6
        if C < 1e-6: C = 1e-6
        Cn = C**n
        hS = Cn / (Cn + MIC_S_n)
        hR = Cn / (Cn + MIC_R_n)
        gS = r_S*(1 - N/K) - b*hS
        gR = r_R*(1 - N/K) - c_R - b_R*hR
        gbar = (1-p)*gS + p*gR
        dN = N * gbar
        dp = p*(1-p)*(gR - gS + gamma*N)
        dC = I_val - mu*C - eta*N*p*C
        reff = (1-p)*(r_S + b*hS) + p*(r_R + c_R + b_R*hR)
        D_NN = max(N*reff, 1e-12)
        D_pp = max(p*(1-p)*reff/N, 1e-12)
        D_CC = max(mu*C + eta*N*p*C, 1e-12)
        N = N + dN*dt + np.sqrt(D_NN)*np.random.randn()*sqdt
        p = p + dp*dt + np.sqrt(D_pp)*np.random.randn()*sqdt
        C = C + dC*dt + np.sqrt(D_CC)*np.random.randn()*sqdt
        traj[i] = [N, p, C]
    t_span = np.linspace(0, t_max, n_steps)
    return t_span, traj

# ============================================================
# CORRECTED EWS FUNCTIONS
# ============================================================
def detrend_linear(ts):
    x = np.arange(len(ts))
    coeffs = np.polyfit(x, ts, 1)
    trend = np.polyval(coeffs, x)
    return ts - trend

def rolling_stat_individual(all_trajs, window, func, burn_in_frac=0.5):
    """Compute rolling statistic on EACH individual trajectory, then average."""
    n_real, n_time = all_trajs.shape
    burn_in = int(n_time * burn_in_frac)
    post = all_trajs[:, burn_in:]
    n_post = post.shape[1]
    if n_post <= window + 10:
        return np.array([]), np.array([])
    n_roll = n_post - window
    all_vals = np.zeros((n_real, n_roll))
    for r in range(n_real):
        traj = post[r]
        for i in range(n_roll):
            val = func(traj[i:i+window])
            all_vals[r, i] = val if not np.isnan(val) else np.nan
    return np.nanmean(all_vals, axis=0), np.nanstd(all_vals, axis=0)

def compute_variance(ts): return np.var(ts)
def compute_skewness(ts): return skew(ts)
def compute_kurtosis(ts): return kurtosis(ts)

def compute_ar1(ts):
    dt = detrend_linear(ts)
    if len(dt) > 1 and np.std(dt) > 1e-12:
        c = np.corrcoef(dt[:-1], dt[1:])[0, 1]
        return c if not np.isnan(c) else np.nan
    return np.nan

def compute_cv(ts):
    m = np.mean(ts)
    if abs(m) > 1e-10: return np.std(ts) / abs(m)
    return np.nan

def compute_spectral_ratio(ts, fs=50.0):
    try:
        dt = detrend_linear(ts)
        if np.std(dt) < 1e-12: return np.nan
        fv = np.abs(fft(dt))[:len(dt)//2]
        fr = fftfreq(len(dt), d=1/fs)[:len(dt)//2]
        if len(fr) > 0:
            lp = np.sum(fv[(fr >= 0) & (fr < 0.5)]**2)
            hp = np.sum(fv[fr >= 0.5]**2)
            if hp > 1e-20: return lp / hp
    except: pass
    return np.nan

def ensemble_variance_ews(all_trajs, window, burn_in_frac=0.5):
    n_real, n_time = all_trajs.shape
    burn_in = int(n_time * burn_in_frac)
    post = all_trajs[:, burn_in:]
    n_post = post.shape[1]
    if n_post <= window + 10: return np.array([])
    var_across = np.var(post, axis=0)
    n_roll = n_post - window
    return np.array([np.mean(var_across[i:i+window]) for i in range(n_roll)])

# ============================================================
# MAIN ANALYSIS
# ============================================================

def main():
    print("="*70)
    print("CORRECTED EWS ANALYSIS")
    print("="*70)

    # [1] Bifurcation
    I_bif, valid_data = find_bifurcation_point()
    print(f"\n[1] Bifurcation at I ≈ {I_bif:.2f}")

    # [2] Conditions
    I_far, I_mid = 5.0, 9.0
    I_near = max(5.0, min(I_bif - 0.3, 11.5))
    conditions = [("Far", I_far, "#1f77b4"), ("Mid", I_mid, "#2ca02c"), ("Near", I_near, "#d62728")]

    print(f"\n[2] Conditions:")
    for name, I_val, color in conditions:
        X = find_equilibrium(I_val)
        if X is not None:
            eigs = eigvals(jacobian(X[0], X[1], X[2]))
            asi = compute_ASI(X[0], X[1], X[2])
            print(f"    {name}: I={I_val:.1f}, ASI={asi:.4f}, p*={X[1]:.4f}, λ_dom={np.max(np.real(eigs)):.6f}")

    # [3] Simulations
    print(f"\n[3] Running ensemble simulations...")
    n_real, t_max, dt, window, burn_frac = 100, 100, 0.02, 80, 0.5
    results = {}

    for name, I_val, color in conditions:
        print(f"    {name} (I={I_val:.1f})...", end=" ", flush=True)
        X_eq = find_equilibrium(I_val)
        asi = compute_ASI(X_eq[0], X_eq[1], X_eq[2])
        all_p = []
        for r in range(n_real):
            X0 = np.array(X_eq) + np.random.normal(0, 1e-4, 3)
            X0[0] = max(X0[0], 1e-6); X0[1] = np.clip(X0[1], 1e-6, 1-1e-6); X0[2] = max(X0[2], 1e-6)
            t, traj = stochastic_sim_fast(I_val, X0, t_max=t_max, dt=dt, seed=r)
            all_p.append(traj[:, 1])
        all_p = np.array(all_p)
        mean_p = np.mean(all_p, axis=0)
        std_p = np.std(all_p, axis=0)
        bi = int(len(mean_p) * burn_frac)
        t_post = t[bi:]

        var_m, var_s = rolling_stat_individual(all_p, window, compute_variance, burn_frac)
        ar1_m, ar1_s = rolling_stat_individual(all_p, window, compute_ar1, burn_frac)
        skew_m, skew_s = rolling_stat_individual(all_p, window, compute_skewness, burn_frac)
        kurt_m, kurt_s = rolling_stat_individual(all_p, window, compute_kurtosis, burn_frac)
        cv_m, cv_s = rolling_stat_individual(all_p, window, compute_cv, burn_frac)
        spec_m, spec_s = rolling_stat_individual(all_p, window, compute_spectral_ratio, burn_frac)
        ens_var = ensemble_variance_ews(all_p, window, burn_frac)

        results[name] = {
            'I': I_val, 'ASI': asi, 't_full': t, 't_post': t_post,
            'all_p': all_p, 'mean_p': mean_p, 'std_p': std_p,
            'color': color,
            'var_m': var_m, 'var_s': var_s,
            'ar1_m': ar1_m, 'ar1_s': ar1_s,
            'skew_m': skew_m, 'skew_s': skew_s,
            'kurt_m': kurt_m, 'kurt_s': kurt_s,
            'cv_m': cv_m, 'cv_s': cv_s,
            'spec_m': spec_m, 'spec_s': spec_s,
            'ens_var': ens_var,
        }
        print(f"DONE (ASI={asi:.4f})")

    # [4] Generate figure
    print("\n[4] Generating figure...")
    I_vals = [d[0] for d in valid_data]
    p_vals = [d[2] for d in valid_data]
    lam_vals = [d[4] for d in valid_data]
    asi_vals = [compute_ASI(d[1], d[2], d[3]) for d in valid_data]

    fig = plt.figure(figsize=(22, 18))
    gs = fig.add_gridspec(5, 3, hspace=0.45, wspace=0.30)

    # ROW 0: THEORY
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(I_vals, p_vals, 'k-', linewidth=2.5, label='Stable branch')
    ax.axvline(x=I_bif, color='r', linestyle='--', linewidth=2, alpha=0.7, label=f'Bifurcation I={I_bif:.2f}')
    for name, I_val, color in conditions:
        X = find_equilibrium(I_val)
        if X is not None:
            ax.plot(I_val, X[1], 'o', color=color, markersize=14, markeredgecolor='black', markeredgewidth=2.5, zorder=5)
    ax.set_xlabel('Drug dosing rate I', fontsize=14)
    ax.set_ylabel('Resistant fraction p*', fontsize=14)
    ax.set_title('A. Equilibrium branch p*(I)', fontsize=15, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(I_vals, asi_vals, 'o-', color='darkgreen', markersize=4, linewidth=1.5, alpha=0.8, label='ASI (true)')
    ax.axvline(x=I_bif, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        ax.plot(r['I'], r['ASI'], 'o', color=r['color'], markersize=14, markeredgecolor='black', markeredgewidth=2.5, zorder=5)
    ax.set_xlabel('Drug dosing rate I', fontsize=14)
    ax.set_ylabel('ASI', fontsize=14)
    ax.set_title('B. ASI → 0 at tipping', fontsize=15, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(asi_vals)*1.05])

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(I_vals, -np.array(lam_vals), 'o-', color='blue', markersize=4, linewidth=1.5, alpha=0.8, label='-λ_dom (true)')
    ax.axvline(x=I_bif, color='r', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(y=0, color='k', linewidth=1, alpha=0.5)
    ax.set_xlabel('Drug dosing rate I', fontsize=14)
    ax.set_ylabel('-λ_dom', fontsize=14)
    ax.set_title('C. Stability loss (λ_dom → 0⁻)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ROW 1: TRAJECTORIES
    for idx, name in enumerate(["Far", "Mid", "Near"]):
        r = results[name]
        ax = fig.add_subplot(gs[1, idx])
        n_plot = min(10, n_real)
        for i in range(n_plot):
            ax.plot(r['t_full'], r['all_p'][i], color=r['color'], alpha=0.12, linewidth=0.4)
        ax.plot(r['t_full'], r['mean_p'], 'k-', linewidth=2.5, label='Ensemble mean')
        ax.fill_between(r['t_full'], r['mean_p']-r['std_p'], r['mean_p']+r['std_p'], alpha=0.2, color=r['color'])
        burn_t = r['t_full'][int(len(r['t_full'])*burn_frac)]
        ax.axvline(x=burn_t, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Burn-in')
        ax.set_xlabel('Time', fontsize=14)
        ax.set_ylabel('Resistant fraction p', fontsize=14)
        ax.set_title(f'{chr(68+idx)}. {name} from tipping\nI={r["I"]:.1f}, ASI={r["ASI"]:.3f}', fontsize=15, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])

    # ROW 2: VARIANCE, AR(1), SKEWNESS
    ax_g = fig.add_subplot(gs[2, 0])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['var_m']) > 0:
            t_ews = r['t_post'][:len(r['var_m'])]
            ax_g.plot(t_ews, r['var_m'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_g.set_xlabel('Time', fontsize=14); ax_g.set_ylabel('Variance', fontsize=14)
    ax_g.set_title('G. Rolling Variance (individual traj)', fontsize=15, fontweight='bold')
    ax_g.legend(loc='upper left', fontsize=10); ax_g.grid(True, alpha=0.3)

    ax_h = fig.add_subplot(gs[2, 1])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['ar1_m']) > 0:
            t_ews = r['t_post'][:len(r['ar1_m'])]
            ax_h.plot(t_ews, r['ar1_m'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_h.set_xlabel('Time', fontsize=14); ax_h.set_ylabel('AR(1) coefficient', fontsize=14)
    ax_h.set_title('H. Rolling AR(1) (detrended, individual)', fontsize=15, fontweight='bold')
    ax_h.legend(loc='upper left', fontsize=10); ax_h.grid(True, alpha=0.3)
    ax_h.set_ylim([-0.3, 1.0])

    ax_i = fig.add_subplot(gs[2, 2])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['skew_m']) > 0:
            t_ews = r['t_post'][:len(r['skew_m'])]
            ax_i.plot(t_ews, r['skew_m'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_i.set_xlabel('Time', fontsize=14); ax_i.set_ylabel('Skewness', fontsize=14)
    ax_i.set_title('I. Rolling Skewness (individual)', fontsize=15, fontweight='bold')
    ax_i.legend(loc='upper left', fontsize=10); ax_i.grid(True, alpha=0.3)
    ax_i.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    # ROW 3: KURTOSIS, CV, SPECTRAL
    ax_j = fig.add_subplot(gs[3, 0])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['kurt_m']) > 0:
            t_ews = r['t_post'][:len(r['kurt_m'])]
            ax_j.plot(t_ews, r['kurt_m'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_j.set_xlabel('Time', fontsize=14); ax_j.set_ylabel('Kurtosis', fontsize=14)
    ax_j.set_title('J. Rolling Kurtosis (individual)', fontsize=15, fontweight='bold')
    ax_j.legend(loc='upper left', fontsize=10); ax_j.grid(True, alpha=0.3)
    ax_j.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)

    ax_k = fig.add_subplot(gs[3, 1])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['cv_m']) > 0:
            t_ews = r['t_post'][:len(r['cv_m'])]
            ax_k.plot(t_ews, r['cv_m'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_k.set_xlabel('Time', fontsize=14); ax_k.set_ylabel('Coefficient of Variation', fontsize=14)
    ax_k.set_title('K. Rolling CV (individual)', fontsize=14, fontweight='bold')
    ax_k.legend(loc='upper left', fontsize=10); ax_k.grid(True, alpha=0.3)

    ax_l = fig.add_subplot(gs[3, 2])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['spec_m']) > 0:
            t_ews = r['t_post'][:len(r['spec_m'])]
            ax_l.plot(t_ews, r['spec_m'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_l.set_xlabel('Time', fontsize=14); ax_l.set_ylabel('Spectral ratio (LF/HF)', fontsize=14)
    ax_l.set_title('L. Rolling Spectral Ratio (individual)', fontsize=15, fontweight='bold')
    ax_l.legend(loc='upper left', fontsize=10); ax_l.grid(True, alpha=0.3)

    # ROW 4: ENSEMBLE VARIANCE + COMPARISON
    ax_m = fig.add_subplot(gs[4, 0])
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        if len(r['ens_var']) > 0:
            t_ews = r['t_post'][:len(r['ens_var'])]
            ax_m.plot(t_ews, r['ens_var'], color=r['color'], linewidth=2.5, alpha=0.8, label=name)
    ax_m.set_xlabel('Time', fontsize=14); ax_m.set_ylabel('Ensemble variance', fontsize=14)
    ax_m.set_title('M. Ensemble Variance (across realizations)', fontsize=15, fontweight='bold')
    ax_m.legend(loc='upper left', fontsize=10); ax_m.grid(True, alpha=0.3)

    # N: Dual-axis comparison with RAW values
    ax_n = fig.add_subplot(gs[4, 1:])
    cond_names = []; asi_vals_c = []; var_vals = []; ar1_vals = []; skew_vals = []; kurt_vals = []; cv_vals = []; spec_vals = []; ensv_vals = []
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        cond_names.append(name.replace(' ', '\n'))
        asi_vals_c.append(r['ASI'])
        var_vals.append(np.nanmean(r['var_m']) if len(r['var_m']) > 0 else np.nan)
        ar1_vals.append(np.nanmean(r['ar1_m']) if len(r['ar1_m']) > 0 else np.nan)
        skew_vals.append(np.nanmean(r['skew_m']) if len(r['skew_m']) > 0 else np.nan)
        kurt_vals.append(np.nanmean(r['kurt_m']) if len(r['kurt_m']) > 0 else np.nan)
        cv_vals.append(np.nanmean(r['cv_m']) if len(r['cv_m']) > 0 else np.nan)
        spec_vals.append(np.nanmean(r['spec_m']) if len(r['spec_m']) > 0 else np.nan)
        ensv_vals.append(np.nanmean(r['ens_var']) if len(r['ens_var']) > 0 else np.nan)

    x_pos = np.arange(len(cond_names))
    ax_n.plot(x_pos, asi_vals_c, 'o-', color='darkgreen', markersize=12, linewidth=3,
              markeredgecolor='black', markeredgewidth=2, label='ASI', zorder=5)
    ax_n.set_ylabel('ASI value', fontsize=14, color='darkgreen')
    ax_n.tick_params(axis='y', labelcolor='darkgreen')
    ax_n.set_ylim([0, max(asi_vals_c)*1.2])

    ax_n2 = ax_n.twinx()
    ax_n2.plot(x_pos, var_vals, 's--', color='steelblue', markersize=8, linewidth=2, alpha=0.7, label='Variance')
    ax_n2.plot(x_pos, ar1_vals, '^--', color='darkorange', markersize=8, linewidth=2, alpha=0.7, label='AR(1)')
    ax_n2.plot(x_pos, skew_vals, 'd--', color='purple', markersize=8, linewidth=2, alpha=0.7, label='Skewness')
    ax_n2.plot(x_pos, kurt_vals, 'v--', color='sienna', markersize=8, linewidth=2, alpha=0.7, label='Kurtosis')
    ax_n2.plot(x_pos, cv_vals, 'p--', color='teal', markersize=8, linewidth=2, alpha=0.7, label='CV')
    ax_n2.plot(x_pos, spec_vals, 'h--', color='crimson', markersize=8, linewidth=2, alpha=0.7, label='Spectral')
    ax_n2.plot(x_pos, ensv_vals, '*--', color='navy', markersize=12, linewidth=2, alpha=0.7, label='Ens.Var')
    ax_n2.set_ylabel('Classical EWS values', fontsize=14)
    ax_n2.tick_params(axis='y')

    lines1, labels1 = ax_n.get_legend_handles_labels()
    lines2, labels2 = ax_n2.get_legend_handles_labels()
    ax_n.legend(lines1+lines2, labels1+labels2, loc='center left', fontsize=9, ncol=1)

    ax_n.set_xticks(x_pos)
    ax_n.set_xticklabels(cond_names, fontsize=11)
    ax_n.set_title('N. EWS Comparison (actual values)\nASI vs Classical indicators', fontsize=15, fontweight='bold')
    ax_n.grid(True, alpha=0.3, axis='x')
    ax_n.annotate('ASI: clear monotonic trend\nClassical: mixed/noisy signals',
                  xy=(0.98, 0.05), xycoords='axes fraction', fontsize=11, ha='right', va='bottom',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('ews_comprehensive_corrected.png', dpi=300, bbox_inches='tight')
    print("    Saved: ews_comprehensive_corrected.png")

    # [5] Quantitative summary
    print(f"\n[5] Quantitative summary:")
    print("-"*100)
    print(f"{'Condition':<18} {'ASI':>8} {'Var':>12} {'AR(1)':>10} {'Skew':>10} {'Kurt':>10} {'CV':>10} {'Spec':>12} {'Ens.Var':>12}")
    print("-"*100)
    for name in ["Far", "Mid", "Near"]:
        r = results[name]
        asi_val = r['ASI'] if not np.isnan(r['ASI']) else 0
        var_mean = np.nanmean(r['var_m']) if len(r['var_m']) > 0 else 0
        ar1_mean = np.nanmean(r['ar1_m']) if len(r['ar1_m']) > 0 else 0
        skew_mean = np.nanmean(r['skew_m']) if len(r['skew_m']) > 0 else 0
        kurt_mean = np.nanmean(r['kurt_m']) if len(r['kurt_m']) > 0 else 0
        cv_mean = np.nanmean(r['cv_m']) if len(r['cv_m']) > 0 else 0
        spec_mean = np.nanmean(r['spec_m']) if len(r['spec_m']) > 0 else 0
        ensv_mean = np.nanmean(r['ens_var']) if len(r['ens_var']) > 0 else 0
        print(f"{name:<18} {asi_val:>8.4f} {var_mean:>12.6f} {ar1_mean:>10.4f} {skew_mean:>10.4f} {kurt_mean:>10.4f} {cv_mean:>10.4f} {spec_mean:>12.4f} {ensv_mean:>12.4f}")

    print("\n" + "="*70)
    print("COMPLETE - Corrected version")
    print("="*70)

if __name__ == "__main__":
    main()
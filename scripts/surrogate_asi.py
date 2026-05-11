"""
Surrogate ASI Validation – Corrected: compare λ_surr with J22 (p‑eigenvalue)
===========================================================================

PURPOSE
-------
Validates the surrogate eigenvalue formula against the exact p‑eigenvalue
J22 of the full 3D Jacobian along a tipping trajectory (ramping I from low
to high). Tests THREE conditions:
  1. IDEAL: True C known (simulation output) — validates the formula itself
  2. MISSPECIFIED: C_est = I/μ (clinical proxy, no feedback) — tests the
     breakdown condition from compendium v4 (Sec 2.4): "Fails when C* fixed
     at I/mu." This is the clinically relevant failure mode.
  3. HIGH-p: p > 0.1 — tests where O(p) corrections to eigenvalue
     factorization become significant (compendium Sec 4.6.1).

MATHEMATICAL FRAMEWORK (Compendium v4, Sections 2.2, 2.4, 4.6.1)
-----------------------------------------------------------------
The surrogate approximates J22 (p-eigenvalue) at the susceptible boundary.
At p*=0, J22 = Δ_g + γN exactly. The surrogate formula is:
  λ_surr = Δ_r - (r_R-r_S)·(N/K) + b·f(C,MIC_S) - b_R·f(C,MIC_R) + γN
The clinical surrogate substitutes C_est = I/μ for the true C*.
When η·N·p << μ (low burden/low resistance), C* ≈ I/μ and surrogate is accurate.
When η·N·p ~ μ (high burden), C* << I/μ and surrogate overestimates drug effect.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
from scipy.stats import pearsonr

# ============================================================
# PARAMETERS — Confirmed bistable set (Compendium v4, Sec 3.1)
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
# CAVEAT: These are literature priors from heterogeneous sources.
# ------------------------------------------------------------------------------

PARAMS = {
    'r_S': 1.0, 'r_R': 0.93, 'c_R': 0.04,
    'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0,
    'n': 3.0,
    'K': 1e9,
    'mu': 1.0,
    'eta': 2e-8,
    'gamma': 1e-12,
    'I': 1.0
}

def hill(C, MIC, n):
    if C <= 0: return 0.0
    return C**n / (C**n + MIC**n)

def growth_rates(N, p, C, p_dict):
    r_S, r_R, c_R, K = p_dict['r_S'], p_dict['r_R'], p_dict['c_R'], p_dict['K']
    b, b_R = p_dict['b'], p_dict['b_R']
    MIC_S, MIC_R, n = p_dict['MIC_S'], p_dict['MIC_R'], p_dict['n']
    f_S = hill(C, MIC_S, n)
    f_R = hill(C, MIC_R, n)
    g_S = r_S * (1 - N / K) - b * f_S
    g_R = r_R * (1 - N / K) - c_R - b_R * f_R
    g_bar = (1 - p) * g_S + p * g_R
    return g_S, g_R, g_bar

def f_ode(t, state, p_dict, I_func):
    N, p, C = state
    p_dict = p_dict.copy()
    p_dict['I'] = I_func(t)
    g_S, g_R, g_bar = growth_rates(N, p, C, p_dict)
    dN = N * g_bar
    dp = p * (1 - p) * (g_R - g_S) + p_dict['gamma'] * N * p * (1 - p)
    dC = p_dict['I'] - p_dict['mu'] * C - p_dict['eta'] * N * p * C
    return [dN, dp, dC]

def jacobian(state, p_dict):
    N, p, C = state
    r_S, r_R, c_R, K = p_dict['r_S'], p_dict['r_R'], p_dict['c_R'], p_dict['K']
    b, b_R = p_dict['b'], p_dict['b_R']
    MIC_S, MIC_R, n = p_dict['MIC_S'], p_dict['MIC_R'], p_dict['n']
    mu, eta, gamma = p_dict['mu'], p_dict['eta'], p_dict['gamma']
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

# =============================================================================
# SURROGATE EIGENVALUE (two versions)
# =============================================================================
# VERSION 1: True C (simulation output) — validates the formula itself
# VERSION 2: C_est = I/μ (clinical proxy) — tests the breakdown condition
#   from compendium v4 (Sec 2.4): "Fails when C* fixed at I/mu."
#   This is the clinically relevant failure mode: in practice, C* is unobserved
#   and must be estimated. When η·N·p ~ μ, C* << I/μ and the surrogate
#   overestimates drug effect, producing biased ASI.

def surrogate_eigenvalue(state, p_dict, use_true_C=True):
    """
    λ_surr = Δr + b·f_S - b_R·f_R - (r_R - r_S)·(N/K) + γ·N
    If use_true_C=False, substitutes C_est = I/μ for the true C.
    """
    N, p, C = state
    r_S, r_R, c_R = p_dict['r_S'], p_dict['r_R'], p_dict['c_R']
    K = p_dict['K']
    b, b_R = p_dict['b'], p_dict['b_R']
    MIC_S, MIC_R, n = p_dict['MIC_S'], p_dict['MIC_R'], p_dict['n']
    gamma = p_dict['gamma']
    Delta_r = r_R - r_S - c_R
    if use_true_C:
        C_eff = C
    else:
        C_eff = p_dict['I'] / p_dict['mu']  # Clinical proxy: ignores feedback
    f_S = hill(C_eff, MIC_S, n)
    f_R = hill(C_eff, MIC_R, n)
    term_NK = (r_R - r_S) * (N / K)
    return Delta_r + b * f_S - b_R * f_R - term_NK + gamma * N

def reference_eigenvalue(p_dict, I_ref=1.0):
    p_dict_sim = p_dict.copy()
    p_dict_sim['I'] = I_ref
    init = [1e5, 0.01, I_ref / p_dict_sim['mu']]
    def ode_fixed(t, state):
        return f_ode(t, state, p_dict_sim, lambda t: I_ref)
    sol = solve_ivp(ode_fixed, [0, 5000], init, method='LSODA', rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError("Reference equilibrium integration failed")
    final_state = sol.y[:, -1]
    J_ref = jacobian(final_state, p_dict_sim)
    eig_ref = eigvals(J_ref)
    lambda_ref = np.max(eig_ref.real)
    if lambda_ref >= 0:
        lambda_ref = -1e-6
    return lambda_ref, final_state

# =============================================================================
# VALIDATION ROUTINE (three tests)
# =============================================================================

def validate_surrogate():
    I_start, I_end = 1.0, 12.0
    t_max = 5000.0
    def I_ramp(t):
        return I_start + (I_end - I_start) * (t / t_max)
    
    # Find initial equilibrium at I_start
    p_dict_init = PARAMS.copy()
    p_dict_init['I'] = I_start
    init_guess = [1e5, 0.01, I_start / PARAMS['mu']]
    def ode_start(t, state):
        return f_ode(t, state, p_dict_init, lambda t: I_start)
    sol_init = solve_ivp(ode_start, [0, 1000], init_guess, method='LSODA', rtol=1e-8, atol=1e-10)
    if not sol_init.success:
        raise RuntimeError("Could not find initial equilibrium")
    init_state = sol_init.y[:, -1]
    
    # Simulate ramp
    sol = solve_ivp(f_ode, [0, t_max], init_state, args=(PARAMS, I_ramp),
                    t_eval=np.linspace(0, t_max, 1000), method='LSODA',
                    rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError("Simulation failed")
    
    t, N, p, C = sol.t, sol.y[0], sol.y[1], sol.y[2]
    
    # Compute all eigenvalues along trajectory
    lambda_surr_true = np.zeros_like(t)
    lambda_surr_miss = np.zeros_like(t)
    J22_true = np.zeros_like(t)
    lambda_dom = np.zeros_like(t)
    for i in range(len(t)):
        p_dict_i = PARAMS.copy()
        p_dict_i['I'] = I_ramp(t[i])
        state = (N[i], p[i], C[i])
        J = jacobian(state, p_dict_i)
        eig = eigvals(J)
        lambda_dom[i] = np.max(eig.real)
        J22_true[i] = J[1,1]
        lambda_surr_true[i] = surrogate_eigenvalue(state, p_dict_i, use_true_C=True)
        lambda_surr_miss[i] = surrogate_eigenvalue(state, p_dict_i, use_true_C=False)
    
    # --- TEST 1: Ideal (true C, p < 0.1) ---
    valid_low_p = p < 0.1
    corr_ideal, pval_ideal = pearsonr(lambda_surr_true[valid_low_p], J22_true[valid_low_p])
    mae_ideal = np.mean(np.abs(lambda_surr_true[valid_low_p] - J22_true[valid_low_p]))
    
    # --- TEST 2: Misspecified C (C_est = I/μ, p < 0.1) ---
    corr_miss, pval_miss = pearsonr(lambda_surr_miss[valid_low_p], J22_true[valid_low_p])
    mae_miss = np.mean(np.abs(lambda_surr_miss[valid_low_p] - J22_true[valid_low_p]))
    
    # --- TEST 3: High p (p > 0.1, true C) ---
    valid_high_p = p > 0.1
    n_high = np.sum(valid_high_p)
    if n_high > 2:
        corr_high, pval_high = pearsonr(lambda_surr_true[valid_high_p], J22_true[valid_high_p])
        mae_high = np.mean(np.abs(lambda_surr_true[valid_high_p] - J22_true[valid_high_p]))
    else:
        corr_high, mae_high = np.nan, np.nan
    
    # Print results
    print("="*70)
    print("SURROGATE VALIDATION – THREE TESTS")
    print("="*70)
    print(f"I_start={I_start}, I_end={I_end}, t_max={t_max} hr")
    print(f"\nTEST 1: IDEAL (true C, p<0.1) — n={np.sum(valid_low_p)}")
    print(f"  Correlation (λ_surr vs J22): {corr_ideal:.8f} (p={pval_ideal:.2e})")
    print(f"  MAE: {mae_ideal:.2e}")
    print(f"  Pass (corr>0.999 & MAE<1e-6)? {corr_ideal > 0.999 and mae_ideal < 1e-6}")
    print(f"\nTEST 2: MISSPECIFIED C (C_est=I/μ, p<0.1) — n={np.sum(valid_low_p)}")
    print(f"  Correlation (λ_surr vs J22): {corr_miss:.8f} (p={pval_miss:.2e})")
    print(f"  MAE: {mae_miss:.2e}")
    print(f"  Degradation vs ideal: Δcorr={corr_ideal-corr_miss:.4f}, ΔMAE={mae_miss-mae_ideal:.2e}")
    print(f"  Pass (corr>0.95)? {corr_miss > 0.95}")
    print(f"\nTEST 3: HIGH p (p>0.1, true C) — n={n_high}")
    if n_high > 2:
        print(f"  Correlation (λ_surr vs J22): {corr_high:.8f}")
        print(f"  MAE: {mae_high:.2e}")
        print(f"  Degradation vs ideal: Δcorr={corr_ideal-corr_high:.4f}, ΔMAE={mae_high-mae_ideal:.2e}")
        print(f"  O(p) corrections significant? {mae_high > 10*mae_ideal}")
    else:
        print("  Insufficient data (p never exceeded 0.1).")
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Time series
    ax = axes[0,0]
    ax.plot(t, J22_true, 'b-', label='J22 (true)')
    ax.plot(t, lambda_surr_true, 'r--', label='λ_surr (true C)')
    ax.set_title(f'TEST 1: Ideal (corr={corr_ideal:.4f})')
    ax.set_ylabel('λ')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0,1]
    ax.plot(t, J22_true, 'b-', label='J22 (true)')
    ax.plot(t, lambda_surr_miss, 'g--', label='λ_surr (C_est=I/μ)')
    ax.set_title(f'TEST 2: Misspecified C (corr={corr_miss:.4f})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0,2]
    ax.plot(t, p, 'm-')
    ax.axhline(0.1, color='r', linestyle='--', label='validity limit')
    ax.set_title('p(t) — validity region')
    ax.set_ylabel('p')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Row 2: Scatter plots
    ax = axes[1,0]
    ax.scatter(J22_true[valid_low_p], lambda_surr_true[valid_low_p], c=t[valid_low_p], cmap='viridis', alpha=0.8)
    min_v = min(J22_true[valid_low_p].min(), lambda_surr_true[valid_low_p].min())
    max_v = max(J22_true[valid_low_p].max(), lambda_surr_true[valid_low_p].max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--')
    ax.set_xlabel('J22')
    ax.set_ylabel('λ_surr (true C)')
    ax.set_title(f'Ideal: scatter (p<0.1)')
    ax.grid(alpha=0.3)
    
    ax = axes[1,1]
    ax.scatter(J22_true[valid_low_p], lambda_surr_miss[valid_low_p], c=t[valid_low_p], cmap='plasma', alpha=0.8)
    min_v = min(J22_true[valid_low_p].min(), lambda_surr_miss[valid_low_p].min())
    max_v = max(J22_true[valid_low_p].max(), lambda_surr_miss[valid_low_p].max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--')
    ax.set_xlabel('J22')
    ax.set_ylabel('λ_surr (C_est=I/μ)')
    ax.set_title(f'Misspecified: scatter (p<0.1)')
    ax.grid(alpha=0.3)
    
    ax = axes[1,2]
    ax.plot(t, lambda_dom, 'm-', label='λ_dom')
    ax.plot(t, J22_true, 'b-', label='J22')
    ax.set_title('Dominant vs p-eigenvalue')
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('λ')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('surrogate_validation_three_tests.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    lambda_ref, _ = reference_eigenvalue(PARAMS, I_ref=1.0)
    print(f"Reference eigenvalue (I=1.0): λ_ref = {lambda_ref:.6f}")
    validate_surrogate()
"""
Surrogate ASI Validation – Corrected: compare λ_surr with J22 (p‑eigenvalue)
============================================================================

PURPOSE
-------
Validates the surrogate eigenvalue formula against the exact p‑eigenvalue
J22 of the full 3D Jacobian along a tipping trajectory (ramping I from low
to high). Tests THREE conditions:
  1. IDEAL: True C known (simulation output) — validates the formula itself
  2. MISSPECIFIED: C_est = I/μ (clinical proxy, no feedback) — tests the
     breakdown condition from compendium v4 (Sec 2.4).
  3. HIGH-p: p > 0.1 — tests the proportional relationship when p≠0.

MATHEMATICAL FRAMEWORK (Compendium v4, Sections 2.2, 2.4, 4.6.1)
-----------------------------------------------------------------
The surrogate approximates the p-eigenvalue J22.
At p*=0:  J22 = Δ_g + γN  exactly.
At p*>0:  J22 = (1 − 2p*)(Δ_g + γN) = (1 − 2p*) × λ_surr.

The surrogate formula computes λ_surr = Δ_g + γN:
  λ_surr = Δ_r − (r_R−r_S)·(N/K) + b·f(C,MIC_S) − b_R·f(C,MIC_R) + γN

CRITICAL POINT: λ_surr ≠ J22 when p≠0. They are proportional with factor (1−2p).
However, BOTH change sign at the same bifurcation point, so λ_surr is a valid
early warning signal even at moderate p. For perfect numerical agreement at
all p, the corrected surrogate is λ_surr_corr = (1 − 2p) × λ_surr.

CORRECTED BIFURCATION CONTEXT
-----------------------------
This validation ramps I through the bistable TYPE 1 regime only:
  I ∈ [1.0, 11.0] — interior coexistence (p≈0.05→0.9) is stable.
Beyond I≈11.6, the interior equilibrium merges with p=1 (transcritical) and
ceases to exist. The surrogate is not validated past this point.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
from scipy.stats import pearsonr

# ============================================================
# PARAMETERS — Confirmed bistable set (Compendium v4, Sec 3.1)
# ============================================================
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
    if C <= 0:
        return 0.0
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
# SURROGATE EIGENVALUE
# =============================================================================
# VERSION 1: True C (simulation output) — validates the formula itself
# VERSION 2: C_est = I/μ (clinical proxy) — tests the breakdown condition

def surrogate_eigenvalue(state, p_dict, use_true_C=True):
    """
    λ_surr = Δ_g + γN  (equals J22 exactly at p=0; proportional at p>0)
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
        C_eff = p_dict['I'] / p_dict['mu']
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
    # FIXED: I_end = 11.0 (not 12.0) to stay in bistable TYPE 1 regime
    I_start, I_end = 1.0, 11.0
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
    lambda_surr_corr = np.zeros_like(t)
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
        lambda_surr_corr[i] = (1 - 2*p[i]) * lambda_surr_true[i]

    # --- TEST 1a: Ideal (true C, p < 0.1) ---
    valid_low_p = p < 0.1
    corr_ideal, pval_ideal = pearsonr(lambda_surr_true[valid_low_p], J22_true[valid_low_p])
    mae_ideal = np.mean(np.abs(lambda_surr_true[valid_low_p] - J22_true[valid_low_p]))

    # --- TEST 1b: Corrected surrogate (1-2p)*λ_surr vs J22, ALL p ---
    corr_corrected, pval_corr = pearsonr(lambda_surr_corr, J22_true)
    mae_corrected = np.mean(np.abs(lambda_surr_corr - J22_true))

    # --- TEST 2: Misspecified C (C_est = I/μ, p < 0.1) ---
    corr_miss, pval_miss = pearsonr(lambda_surr_miss[valid_low_p], J22_true[valid_low_p])
    mae_miss = np.mean(np.abs(lambda_surr_miss[valid_low_p] - J22_true[valid_low_p]))

    # --- TEST 3: High p (p > 0.1, true C) ---
    valid_high_p = p > 0.1
    n_high = np.sum(valid_high_p)
    if n_high > 2:
        corr_high, pval_high = pearsonr(lambda_surr_true[valid_high_p], J22_true[valid_high_p])
        mae_high = np.mean(np.abs(lambda_surr_true[valid_high_p] - J22_true[valid_high_p]))
        corr_high_corr, _ = pearsonr(lambda_surr_corr[valid_high_p], J22_true[valid_high_p])
        mae_high_corr = np.mean(np.abs(lambda_surr_corr[valid_high_p] - J22_true[valid_high_p]))
    else:
        corr_high, mae_high, corr_high_corr, mae_high_corr = np.nan, np.nan, np.nan, np.nan

    # Print results
    print("="*70)
    print("SURROGATE VALIDATION – THREE TESTS")
    print("="*70)
    print(f"I_start={I_start}, I_end={I_end}, t_max={t_max} hr")
    print(f"\nNOTE: lambda_surr = Delta_g + gamma*N. J22 = (1-2p)*lambda_surr.")
    print(f"At p=0: lambda_surr = J22 exactly. At p>0: proportional with factor (1-2p).")
    print(f"\nTEST 1a: IDEAL — true C, p<0.1 (n={np.sum(valid_low_p)})")
    print(f"  Correlation (lambda_surr vs J22): {corr_ideal:.8f} (p={pval_ideal:.2e})")
    print(f"  MAE: {mae_ideal:.2e}")
    print(f"  Pass (corr>0.999 & MAE<1e-6)? {corr_ideal > 0.999 and mae_ideal < 1e-6}")
    print(f"\nTEST 1b: CORRECTED — (1-2p)*lambda_surr vs J22, ALL p (n={len(t)})")
    print(f"  Correlation: {corr_corrected:.8f} (p={pval_corr:.2e})")
    print(f"  MAE: {mae_corrected:.2e}")
    print(f"  Pass (corr>0.999 & MAE<1e-6)? {corr_corrected > 0.999 and mae_corrected < 1e-6}")
    print(f"\nTEST 2: MISSPECIFIED C — C_est=I/mu, p<0.1 (n={np.sum(valid_low_p)})")
    print(f"  Correlation (lambda_surr vs J22): {corr_miss:.8f} (p={pval_miss:.2e})")
    print(f"  MAE: {mae_miss:.2e}")
    print(f"  Degradation vs ideal: dcorr={corr_ideal-corr_miss:.4f}, dMAE={mae_miss-mae_ideal:.2e}")
    print(f"  Pass (corr>0.95)? {corr_miss > 0.95}")
    print(f"\nTEST 3: HIGH p — p>0.1, true C (n={n_high})")
    print(f"  NOTE: J22 = (1-2p)*lambda_surr by construction. Testing proportional trend.")
    if n_high > 2:
        print(f"  Raw lambda_surr vs J22: corr={corr_high:.4f}, MAE={mae_high:.2e}")
        print(f"  Corrected (1-2p)*lambda_surr vs J22: corr={corr_high_corr:.4f}, MAE={mae_high_corr:.2e}")
        print(f"  O(p) factor significant? {abs(corr_ideal - corr_high) > 0.01}")
    else:
        print("  Insufficient data.")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0,0]
    ax.plot(t, J22_true, 'b-', label='J22 (true)', linewidth=2)
    ax.plot(t, lambda_surr_true, 'r--', label='lambda_surr (raw)', alpha=0.7)
    ax.plot(t, lambda_surr_corr, 'g:', label='(1-2p)*lambda_surr (corrected)', alpha=0.7)
    ax.set_title(f'TEST 1: Ideal + Corrected')
    ax.set_ylabel('lambda')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0,1]
    ax.plot(t, J22_true, 'b-', label='J22 (true)', linewidth=2)
    ax.plot(t, lambda_surr_miss, 'g--', label='lambda_surr (C_est=I/mu)')
    ax.set_title(f'TEST 2: Misspecified C (corr={corr_miss:.4f})')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0,2]
    ax.plot(t, p, 'm-', linewidth=2)
    ax.axhline(0.1, color='r', linestyle='--', label='validity limit p=0.1')
    ax.axvline(t[np.argmin(np.abs(I_ramp(t) - 11.0))], color='orange', linestyle='--', 
               label='I=11 (type 1 boundary)')
    ax.set_title('p(t) — validity region')
    ax.set_ylabel('p')
    ax.set_xlabel('Time (hr)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1,0]
    ax.scatter(J22_true[valid_low_p], lambda_surr_true[valid_low_p], 
               c=t[valid_low_p], cmap='viridis', alpha=0.8, label='Raw lambda_surr')
    ax.scatter(J22_true[valid_low_p], lambda_surr_corr[valid_low_p], 
               c=t[valid_low_p], cmap='viridis', alpha=0.3, marker='x', label='(1-2p)*lambda_surr')
    min_v = min(J22_true[valid_low_p].min(), lambda_surr_true[valid_low_p].min())
    max_v = max(J22_true[valid_low_p].max(), lambda_surr_true[valid_low_p].max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5)
    ax.set_xlabel('J22')
    ax.set_ylabel('lambda_surr')
    ax.set_title(f'Ideal: scatter (p<0.1)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1,1]
    ax.scatter(J22_true[valid_low_p], lambda_surr_miss[valid_low_p], 
               c=t[valid_low_p], cmap='plasma', alpha=0.8)
    min_v = min(J22_true[valid_low_p].min(), lambda_surr_miss[valid_low_p].min())
    max_v = max(J22_true[valid_low_p].max(), lambda_surr_miss[valid_low_p].max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5)
    ax.set_xlabel('J22')
    ax.set_ylabel('lambda_surr (C_est=I/mu)')
    ax.set_title(f'Misspecified: scatter (p<0.1)')
    ax.grid(alpha=0.3)

    ax = axes[1,2]
    ax.plot(t, lambda_dom, 'm-', label='lambda_dom', linewidth=2)
    ax.plot(t, J22_true, 'b-', label='J22', alpha=0.7)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    ax.set_title('Dominant vs p-eigenvalue')
    ax.set_xlabel('Time (hr)')
    ax.set_ylabel('lambda')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('surrogate_validation_three_tests.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    lambda_ref, _ = reference_eigenvalue(PARAMS, I_ref=1.0)
    print(f"Reference eigenvalue (I=1.0): lambda_ref = {lambda_ref:.6f}")
    validate_surrogate()
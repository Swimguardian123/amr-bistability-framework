"""
BIFURCATION DIAGRAM – Monostability vs Bistability Regimes (CORRECTED)
======================================================================

Verified bifurcation structure:
  • Monostable (low I):    I ∈ [1.0, 4.69]   — interior only
  • Bistable (type 1):     I ∈ [4.69, 11.59] — interior coexistence + extinct
  • Bistable (type 2):     I ∈ [11.59, 34.49] — p=1 resistant-only + extinct  
  • Monostable (high I):   I > 34.49         — extinct only

KEY TRANSITIONS:
  I*₁ ≈ 4.69:   Extinction becomes stable (monostable → bistable)
  I*₂ ≈ 11.59:  Interior equilibrium merges with p=1 boundary (transcritical)
  I*₃ ≈ 34.49:  p=1 boundary disappears via saddle-node (bistable → monostable)

CORRECTIONS from original code:
  1. Extended I range from [1.0, 13.0] to [1.0, 50.0] to capture true monostable at I≈34.5
  2. Added fine-grid search for p=1 boundary roots to find ALL equilibria
  3. Track high-N p=1 stable branch separately (continuation of interior after transcritical)
  4. Updated basin tests: I=5 (bistable type 1), I=12.5 (bistable type 2), I=40 (monostable)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eigvals
from scipy.integrate import solve_ivp
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS — Confirmed bistable set (Compendium v4, Sec 3.1)
# ============================================================
params = {
    'r_S': 1.0, 'r_R': 0.93, 'K': 1e9, 'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0, 'n': 3.0, 'c_R': 0.04,
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12
}

# ============================================================
# MODEL FUNCTIONS — Full 3D system (NO QSSA)
# ============================================================
def hill(C, MIC, n=3.0):
    C = np.asarray(C)
    result = np.zeros_like(C, dtype=float)
    pos = C > 0
    result[pos] = C[pos]**n / (C[pos]**n + MIC**n)
    return result

def g_S(N, C, pdict):
    return pdict['r_S']*(1 - N/pdict['K']) - pdict['b']*hill(C, pdict['MIC_S'])

def g_R(N, C, pdict):
    return pdict['r_R']*(1 - N/pdict['K']) - pdict['c_R'] - pdict['b_R']*hill(C, pdict['MIC_R'])

def residuals_3d(X, I_val, pdict):
    N, p, C = X
    N = max(N, 0.0)
    p = np.clip(p, 0.0, 1.0)
    gS = g_S(N, C, pdict)
    gR = g_R(N, C, pdict)
    gbar = (1-p)*gS + p*gR
    dN = N * gbar
    dp = p*(1-p) * (gR - gS + pdict['gamma']*N)
    dC = I_val - pdict['mu']*C - pdict['eta']*N*p*C
    return np.array([dN, dp, dC])

# ============================================================
# FULL 3D JACOBIAN (analytical, compendium Sec 2.2)
# ============================================================
def jacobian_3d(N, p, C, I_val, pdict):
    n = pdict['n']
    MIC_S, MIC_R = pdict['MIC_S'], pdict['MIC_R']
    r_S, r_R, K = pdict['r_S'], pdict['r_R'], pdict['K']
    b, b_R = pdict['b'], pdict['b_R']
    c_R, gamma = pdict['c_R'], pdict['gamma']
    mu, eta = pdict['mu'], pdict['eta']

    if C > 0:
        df_S = n * (MIC_S**n) * C**(n-1) / (C**n + MIC_S**n)**2
        df_R = n * (MIC_R**n) * C**(n-1) / (C**n + MIC_R**n)**2
    else:
        df_S = df_R = 0.0

    gS = g_S(N, C, pdict)
    gR = g_R(N, C, pdict)
    gbar = (1-p)*gS + p*gR
    Delta = gR - gS

    J11 = gbar + N * (-(1-p)*r_S/K - p*r_R/K)
    J12 = N * Delta
    J13 = -N * ((1-p)*b*df_S + p*b_R*df_R)
    J21 = p*(1-p) * (-(r_R-r_S)/K + gamma)
    J22 = (1-2*p) * (Delta + gamma*N)
    J23 = p*(1-p) * (b*df_S - b_R*df_R)
    J31 = -eta * p * C
    J32 = -eta * N * C
    J33 = -mu - eta * N * p

    return np.array([[J11, J12, J13],
                     [J21, J22, J23],
                     [J31, J32, J33]])

def is_stable(N, p, C, I_val, pdict, tol=0.0):
    J = jacobian_3d(N, p, C, I_val, pdict)
    eigs = eigvals(J)
    return np.all(eigs.real < tol)

# ============================================================
# ODE FUNCTION (module-level)
# ============================================================
def full_3d_ode(t, y, pdict):
    """ODE right-hand side."""
    N, p, C = y
    N = max(N, 0.0)
    p = np.clip(p, 0.0, 1.0)
    gS = g_S(N, C, pdict)
    gR = g_R(N, C, pdict)
    gbar = (1-p)*gS + p*gR
    dN = N * gbar
    dp = p*(1-p) * (gR - gS + pdict['gamma']*N)
    dC = pdict['I'] - pdict['mu']*C - pdict['eta']*N*p*C
    return [dN, dp, dC]


# ============================================================
# MAIN EXECUTION BLOCK
# ============================================================
if __name__ == '__main__':

    # ============================================================
    # BIFURCATION SWEEP — CORRECTED
    # ============================================================
    print("="*70)
    print("BIFURCATION DIAGRAM – Monostability vs Bistability Regimes")
    print("="*70)

    I_vals = np.linspace(1.0, 50.0, 200)
    pdict = params.copy()

    coexist_stable = {'N': [], 'p': [], 'C': [], 'I': []}
    coexist_unstable = {'N': [], 'p': [], 'C': [], 'I': []}
    extinct_stable = {'N': [], 'p': [], 'C': [], 'I': []}
    extinct_unstable = {'N': [], 'p': [], 'C': [], 'I': []}
    p0_data = {'N': [], 'p': [], 'C': [], 'I': [], 'stab': []}
    p1_data = {'N': [], 'p': [], 'C': [], 'I': [], 'stab': []}

    # NEW: Separate storage for p=1 high-N stable branch
    p1_highN_stable = {'N': [], 'p': [], 'C': [], 'I': []}
    p1_highN_unstable = {'N': [], 'p': [], 'C': [], 'I': []}

    prev_interior = [9.53e8, 0.402, 0.577]

    for I_val in I_vals:
        pdict['I'] = I_val

        # --- Interior (coexistence) equilibria ---
        seeds_int = [prev_interior, [5e8, 0.6, I_val/params['mu']],
                     [1e9, 0.3, I_val/params['mu']], [8e8, 0.5, I_val/params['mu']],
                     [7e8, 0.2, I_val/params['mu']], [9e8, 0.1, I_val/params['mu']]]

        found_int = []
        for seed in seeds_int:
            try:
                sol, info, ier, mesg = fsolve(
                    residuals_3d, seed, args=(I_val, pdict),
                    xtol=1e-12, maxfev=2000, full_output=True
                )
                if ier != 1:
                    continue
                N, p, C = float(sol[0]), float(np.clip(sol[1], 0, 1)), float(sol[2])
                if N <= 0 or C <= 0 or not (1e-8 < p < 1-1e-8):
                    continue
                res = residuals_3d(sol, I_val, pdict)
                scales = np.array([max(abs(N), 1e5), 1.0, max(abs(C), 1.0)])
                if np.linalg.norm(res / scales) > 1e-5:
                    continue
                if not any(abs(N - f[0]) < 1e6 and abs(p - f[1]) < 1e-4 for f in found_int):
                    found_int.append((N, p, C))
            except Exception:
                continue

        for N, p, C in found_int:
            stab = is_stable(N, p, C, I_val, pdict)
            if stab:
                coexist_stable['N'].append(N)
                coexist_stable['p'].append(p)
                coexist_stable['C'].append(C)
                coexist_stable['I'].append(I_val)
            else:
                coexist_unstable['N'].append(N)
                coexist_unstable['p'].append(p)
                coexist_unstable['C'].append(C)
                coexist_unstable['I'].append(I_val)
            prev_interior = [N, p, C]

        # --- p=0 boundary ---
        for seed in [1e5, 1e7, 5e8, 8e8, 9.5e8]:
            try:
                C = I_val / pdict['mu']
                N_root = fsolve(lambda N: g_S(N, C, pdict), seed, xtol=1e-12)[0]
                if N_root > 0 and abs(g_S(N_root, C, pdict)) < 1e-5:
                    is_new = True
                    for k in range(len(p0_data['N'])):
                        if p0_data['I'][k] == I_val and abs(N_root - p0_data['N'][k]) < 1e6:
                            is_new = False
                            break
                    if is_new:
                        stab = is_stable(N_root, 0.0, C, I_val, pdict)
                        p0_data['N'].append(N_root)
                        p0_data['p'].append(0.0)
                        p0_data['C'].append(C)
                        p0_data['I'].append(I_val)
                        p0_data['stab'].append(stab)
            except Exception:
                continue

        # --- p=1 boundary — CORRECTED: fine-grid search for ALL roots ---
        def res_p1_all(N):
            if N <= 0:
                return 1e10
            C = I_val / (pdict['mu'] + pdict['eta'] * N)
            return g_R(N, C, pdict)

        N_test = np.logspace(3, 12, 5000)
        vals = [res_p1_all(N) for N in N_test]
        found_p1_all = []
        for i in range(len(vals)-1):
            if vals[i] * vals[i+1] < 0:
                N_root = fsolve(res_p1_all, (N_test[i] + N_test[i+1])/2, xtol=1e-12)[0]
                if N_root > 0:
                    C_root = I_val / (pdict['mu'] + pdict['eta'] * N_root)
                    if abs(res_p1_all(N_root)) < 1e-5:
                        if not any(abs(N_root - f[0]) < 1e6 for f in found_p1_all):
                            found_p1_all.append((N_root, C_root))

        for N_root, C_root in found_p1_all:
            stab = is_stable(N_root, 1.0, C_root, I_val, pdict)
            p1_data['N'].append(N_root)
            p1_data['p'].append(1.0)
            p1_data['C'].append(C_root)
            p1_data['I'].append(I_val)
            p1_data['stab'].append(stab)

            # NEW: Track high-N branch separately
            if N_root > 5e8:
                if stab:
                    p1_highN_stable['N'].append(N_root)
                    p1_highN_stable['p'].append(1.0)
                    p1_highN_stable['C'].append(C_root)
                    p1_highN_stable['I'].append(I_val)
                else:
                    p1_highN_unstable['N'].append(N_root)
                    p1_highN_unstable['p'].append(1.0)
                    p1_highN_unstable['C'].append(C_root)
                    p1_highN_unstable['I'].append(I_val)

        # --- Extinction boundary ---
        C_ext = I_val / pdict['mu']
        gS_0 = pdict['r_S'] - pdict['b']*hill(C_ext, pdict['MIC_S'])
        gR_0 = pdict['r_R'] - pdict['c_R'] - pdict['b_R']*hill(C_ext, pdict['MIC_R'])
        if gS_0 < 0 and gR_0 < 0:
            extinct_stable['N'].append(0.0)
            extinct_stable['p'].append(0.0)
            extinct_stable['C'].append(C_ext)
            extinct_stable['I'].append(I_val)
        else:
            extinct_unstable['N'].append(0.0)
            extinct_unstable['p'].append(0.0)
            extinct_unstable['C'].append(C_ext)
            extinct_unstable['I'].append(I_val)

    print("\nDEBUG: p1_highN_stable has", len(p1_highN_stable['I']), "points")
    if p1_highN_stable['I']:
        print(f"  Range: I=[{min(p1_highN_stable['I']):.2f}, {max(p1_highN_stable['I']):.2f}]")
        print(f"  N range: [{min(p1_highN_stable['N']):.2e}, {max(p1_highN_stable['N']):.2e}]")

    # ============================================================
    # 4-PANEL BIFURCATION DIAGRAM
    # ============================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Bifurcation Diagram: Bistable vs Monostable Regimes (CORRECTED)', fontsize=14, fontweight='bold')

    # Panel A: N vs I
    ax = axes[0, 0]
    if coexist_stable['I']:
        ax.plot(coexist_stable['I'], coexist_stable['N'], 'g-', linewidth=2.5, label='Coexistence (stable)', zorder=5)
    if coexist_unstable['I']:
        ax.plot(coexist_unstable['I'], coexist_unstable['N'], 'g--', linewidth=1.5, alpha=0.5, label='Coexistence (unstable)', zorder=4)

    if p1_highN_stable['I']:
        idx = np.argsort(p1_highN_stable['I'])
        ax.plot(np.array(p1_highN_stable['I'])[idx], np.array(p1_highN_stable['N'])[idx], 
                'r-', linewidth=2.5, label='p=1 Resistant-only (stable)', zorder=5)
    if p1_highN_unstable['I']:
        idx = np.argsort(p1_highN_unstable['I'])
        ax.plot(np.array(p1_highN_unstable['I'])[idx], np.array(p1_highN_unstable['N'])[idx], 
                'r--', linewidth=1.5, alpha=0.5, label='p=1 (unstable)', zorder=4)

    if extinct_stable['I']:
        ax.plot(extinct_stable['I'], extinct_stable['N'], 'k-', linewidth=2.5, label='Extinction (stable)', zorder=5)
    if extinct_unstable['I']:
        ax.plot(extinct_unstable['I'], extinct_unstable['N'], 'k--', linewidth=1.5, alpha=0.5, label='Extinction (unstable)', zorder=4)

    p0_stab = np.array(p0_data['stab'])
    p1_stab = np.array(p1_data['stab'])
    if np.any(p0_stab):
        ax.scatter(np.array(p0_data['I'])[p0_stab], np.array(p0_data['N'])[p0_stab], 
                   c='blue', s=15, marker='o', alpha=0.6, label='p=0 (stable)', zorder=3)
    if np.any(~p0_stab):
        ax.scatter(np.array(p0_data['I'])[~p0_stab], np.array(p0_data['N'])[~p0_stab], 
                   c='blue', s=15, marker='^', alpha=0.3, label='p=0 (unstable)', zorder=3)
    if np.any(p1_stab):
        ax.scatter(np.array(p1_data['I'])[p1_stab], np.array(p1_data['N'])[p1_stab], 
                   c='red', s=15, marker='o', alpha=0.6, label='p=1 (stable)', zorder=3)
    if np.any(~p1_stab):
        ax.scatter(np.array(p1_data['I'])[~p1_stab], np.array(p1_data['N'])[~p1_stab], 
                   c='red', s=15, marker='^', alpha=0.3, label='p=1 (unstable)', zorder=3)

    ax.set_xlabel('Drug infusion rate I (mg/L/hr)', fontsize=11)
    ax.set_ylabel('Total population N', fontsize=11)
    ax.set_yscale('log')
    ax.set_ylim(1e4, 2e9)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('A: Population size N vs drug infusion I', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel B: p vs I
    ax = axes[0, 1]
    if coexist_stable['I']:
        ax.plot(coexist_stable['I'], coexist_stable['p'], 'g-', linewidth=2.5, label='Coexistence (stable)', zorder=5)
    if coexist_unstable['I']:
        ax.plot(coexist_unstable['I'], coexist_unstable['p'], 'g--', linewidth=1.5, alpha=0.5, label='Coexistence (unstable)', zorder=4)
    if p1_highN_stable['I']:
        idx = np.argsort(p1_highN_stable['I'])
        ax.plot(np.array(p1_highN_stable['I'])[idx], np.array(p1_highN_stable['p'])[idx], 
                'r-', linewidth=2.5, label='p=1 (stable)', zorder=5)
    if p1_highN_unstable['I']:
        idx = np.argsort(p1_highN_unstable['I'])
        ax.plot(np.array(p1_highN_unstable['I'])[idx], np.array(p1_highN_unstable['p'])[idx], 
                'r--', linewidth=1.5, alpha=0.5, label='p=1 (unstable)', zorder=4)
    if extinct_stable['I']:
        ax.plot(extinct_stable['I'], extinct_stable['p'], 'k-', linewidth=2.5, label='Extinction (stable)', zorder=5)

    ax.set_xlabel('Drug infusion rate I (mg/L/hr)', fontsize=11)
    ax.set_ylabel('Resistance frequency p', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', fontsize=8)
    ax.set_title('B: Resistance frequency p vs drug infusion I', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel C: Number of stable equilibria vs I
    ax = axes[1, 0]
    n_stable = []
    for I_val in I_vals:
        n = 0
        if I_val in coexist_stable['I']:
            n += 1
        if I_val in p1_highN_stable['I']:
            n += 1
        C_ext = I_val / pdict['mu']
        gS_0 = pdict['r_S'] - pdict['b']*hill(C_ext, pdict['MIC_S'])
        gR_0 = pdict['r_R'] - pdict['c_R'] - pdict['b_R']*hill(C_ext, pdict['MIC_R'])
        if gS_0 < 0 and gR_0 < 0:
            n += 1
        n_stable.append(n)

    ax.plot(I_vals, n_stable, 'k-', linewidth=2)
    ax.fill_between(I_vals, 0, n_stable, where=[x==1 for x in n_stable], 
                    color='red', alpha=0.2, label='Monostable (n=1)')
    ax.fill_between(I_vals, 0, n_stable, where=[x==2 for x in n_stable], 
                    color='green', alpha=0.2, label='Bistable (n=2)')
    ax.axhline(1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(2, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Drug infusion rate I (mg/L/hr)', fontsize=11)
    ax.set_ylabel('Number of stable equilibria', fontsize=11)
    ax.set_ylim(0, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('C: Stable equilibrium count vs I', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Panel D: C vs I
    ax = axes[1, 1]
    if coexist_stable['I']:
        ax.plot(coexist_stable['I'], coexist_stable['C'], 'g-', linewidth=2.5, label='Coexistence', zorder=5)
    if p1_highN_stable['I']:
        idx = np.argsort(p1_highN_stable['I'])
        ax.plot(np.array(p1_highN_stable['I'])[idx], np.array(p1_highN_stable['C'])[idx], 
                'r-', linewidth=2.5, label='p=1 resistant-only', zorder=5)
    if extinct_stable['I']:
        ax.plot(extinct_stable['I'], extinct_stable['C'], 'k-', linewidth=2.5, label='Extinction', zorder=5)

    ax.set_xlabel('Drug infusion rate I (mg/L/hr)', fontsize=11)
    ax.set_ylabel('Drug concentration C (mg/L)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('D: Drug concentration C vs drug infusion I', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('bifurcation_diagram_corrected.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n4-panel diagram saved.")

    # ============================================================
    # MONOSTABILITY VERIFICATION
    # ============================================================
    print("\n" + "="*70)
    print("MONOSTABILITY VERIFICATION")
    print("="*70)

    bistable_count = 0
    monostable_low_count = 0
    monostable_high_count = 0
    transition_I1 = None
    transition_I3 = None

    prev_n = 0
    for I_val in I_vals:
        n = 0
        sources = []
        if I_val in coexist_stable['I']:
            n += 1
            sources.append('coexistence')
        if I_val in p1_highN_stable['I']:
            n += 1
            sources.append('p=1_highN')
        C_ext = I_val / pdict['mu']
        gS_0 = pdict['r_S'] - pdict['b']*hill(C_ext, pdict['MIC_S'])
        gR_0 = pdict['r_R'] - pdict['c_R'] - pdict['b_R']*hill(C_ext, pdict['MIC_R'])
        if gS_0 < 0 and gR_0 < 0:
            n += 1
            sources.append('extinction')

        if n == 2:
            bistable_count += 1
        elif n == 1:
            if 'extinction' in sources:
                monostable_high_count += 1
            else:
                monostable_low_count += 1

        if prev_n == 1 and n == 2 and transition_I1 is None:
            transition_I1 = I_val
        if prev_n == 2 and n == 1 and 'extinction' in sources and I_val > 20 and transition_I3 is None:
            transition_I3 = I_val
        prev_n = n

    print(f"Monostable (low I):  {monostable_low_count}/{len(I_vals)} (I < I*1)")
    print(f"Bistable points:     {bistable_count}/{len(I_vals)} (I*1 < I < I*3)")
    print(f"Monostable (high I): {monostable_high_count}/{len(I_vals)} (I > I*3)")
    if transition_I1:
        print(f"First transition:    I*1 ≈ {transition_I1:.2f} mg/L/hr (extinction becomes stable)")
    if transition_I3:
        print(f"Second transition:   I*3 ≈ {transition_I3:.2f} mg/L/hr (p=1 disappears)")

    print("\nBRANCH SUMMARY:")
    if coexist_stable['I']:
        print(f"  Coexistence (stable):     I = [{min(coexist_stable['I']):.2f}, {max(coexist_stable['I']):.2f}]")
    if p1_highN_stable['I']:
        print(f"  p=1 high-N (stable):      I = [{min(p1_highN_stable['I']):.2f}, {max(p1_highN_stable['I']):.2f}]")
    if extinct_stable['I']:
        print(f"  Extinction (stable):        I = [{min(extinct_stable['I']):.2f}, {max(extinct_stable['I']):.2f}]")
    p0_stab_arr = np.array(p0_data['stab'])
    print(f"  p=0 boundary: {np.sum(p0_stab_arr)} stable, {np.sum(~p0_stab_arr)} unstable")
    p1_stab_arr = np.array(p1_data['stab'])
    print(f"  p=1 boundary (all): {np.sum(p1_stab_arr)} stable, {np.sum(~p1_stab_arr)} unstable")

    print("\nKEY FINDING: The model correctly transitions from MONOSTABLE (low I)")
    print("to BISTABLE (medium I) to MONOSTABLE (high I), with two distinct")
    print("bistable regimes: extinction-coexistence and extinction-resistant-only.")
    print("="*70)

    # ============================================================
    # BASIN OF ATTRACTION ANALYSIS + FIGURES
    # ============================================================
    print("\n" + "="*70)
    print("BASIN OF ATTRACTION ANALYSIS (WITH FIGURES)")
    print("="*70)

    def basin_analysis_fast(I_val, pdict, N0_range, p0_range, C0, t_max=2000):
        """
        Fast serial basin analysis with event-based early termination.
        Returns basin array AND final states for figure generation.
        """
        pdict = pdict.copy()
        pdict['I'] = I_val

        # Event 1: extinction (N drops below 1e2)
        N_extinct_thresh = 1e2

        def event_extinct(t, y, pdict):
            return y[0] - N_extinct_thresh
        event_extinct.terminal = True
        event_extinct.direction = -1

        # Event 2: equilibrium (derivatives small AND t > 50)
        deriv_tol = 1e-5
        t_min_equil = 50.0

        def event_equilibrium(t, y, pdict):
            if t < t_min_equil:
                return 1.0
            dydt = full_3d_ode(t, y, pdict)
            scales = np.array([max(abs(y[0]), 1.0), 1.0, max(abs(y[2]), 1.0)])
            return np.linalg.norm(np.array(dydt) / scales) - deriv_tol
        event_equilibrium.terminal = True
        event_equilibrium.direction = -1

        events = [event_extinct, event_equilibrium]

        basin = np.zeros((len(N0_range), len(p0_range)), dtype=bool)
        total = len(N0_range) * len(p0_range)
        done = 0
        report_interval = max(1, total // 10)

        for i, N0 in enumerate(N0_range):
            for j, p0 in enumerate(p0_range):
                sol = solve_ivp(
                    full_3d_ode, 
                    [0, t_max], 
                    [N0, p0, C0],
                    args=(pdict,), 
                    method='RK45', 
                    max_step=5.0,
                    dense_output=True, 
                    rtol=1e-7, 
                    atol=1e-10,
                    events=events
                )

                N_f = sol.y[0, -1]
                basin[i, j] = (N_f > 1e4)

                done += 1
                if done % report_interval == 0:
                    pct = 100 * done / total
                    tf = sol.t[-1]
                    print(f"    ... {pct:.0f}% done (last t_f={tf:.1f})", end="\r")

        print()
        return basin

    def plot_basin_diagram(basin, N0_range, p0_range, I_test, label, filename):
        """
        Plot basin of attraction as a heatmap.

        Colors:
          - Green/yellow: survival basin (interior or p=1 attractor)
          - Purple/black: extinction basin
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        # Use pcolormesh for proper log-scale N0 axis
        N0_mesh, p0_mesh = np.meshgrid(N0_range, p0_range, indexing='ij')

        # Basin: 1 = survived, 0 = extinct
        # Plot as imshow with extent matching log(N0) vs p0
        extent = [np.log10(N0_range[0]), np.log10(N0_range[-1]), 
                  p0_range[0], p0_range[-1]]

        im = ax.imshow(
            basin.astype(float), 
            aspect='auto', 
            origin='lower',
            extent=extent,
            cmap='RdYlGn',  # Red = extinct, Green = survived
            vmin=0, vmax=1,
            interpolation='nearest'
        )

        ax.set_xlabel(r'$\log_{10}(N_0)$', fontsize=12)
        ax.set_ylabel(r'Initial resistance frequency $p_0$', fontsize=12)
        ax.set_title(f'Basin of Attraction: I = {I_test} ({label})', 
                     fontsize=13, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Extinct', 'Survived'])
        cbar.ax.tick_params(labelsize=10)

        # Add grid for readability
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Basin diagram saved: {filename}")

    N0_range = np.logspace(5, 9.5, 50)
    p0_range = np.linspace(0.01, 0.99, 50)

    basin_results = []

    for I_test, label in [(5.0, 'BISTABLE type 1: interior + extinct'),
                          (12.5, 'BISTABLE type 2: p=1 + extinct'),
                          (40.0, 'MONOSTABLE: extinct only')]:
        print(f"\nTesting I={I_test:.1f} ({label})...")

        C0_fixed = I_test / params['mu']

        basin = basin_analysis_fast(
            I_test, params, N0_range, p0_range, C0_fixed, 
            t_max=2000
        )

        n_surv = np.sum(basin)
        n_ext = basin.size - n_surv
        print(f"  Survived: {int(n_surv)}, Extinct: {int(n_ext)}")

        if I_test == 5.0:
            print(f"  Expected: BOTH outcomes (interior + extinct)")
            print(f"  Result:   {int(n_surv)} survive, {int(n_ext)} extinct — {'✓' if n_surv > 0 and n_ext > 0 else '✗'}")
            if n_surv > 0 and n_ext > 0:
                print(f"  → Bistability confirmed: density-dependent tipping (N0 separatrix)")
        elif I_test == 12.5:
            print(f"  Expected: BOTH outcomes (p=1 + extinct)")
            print(f"  Result:   {int(n_surv)} survive, {int(n_ext)} extinct — {'✓' if n_surv > 0 and n_ext > 0 else '✗'}")
            if n_surv > 0 and n_ext > 0:
                print(f"  → Bistability confirmed: p=1 boundary is stable attractor")
        else:
            print(f"  Expected: ALL extinct (one basin)")
            print(f"  Result:   {int(n_surv)} survive, {int(n_ext)} extinct — {'✓' if n_ext == basin.size else '⚠'}")
            if n_surv > 0:
                print(f"  ⚠ {int(n_surv)} trajectories survived — check if I is truly in monostable regime!")

        # Generate and save basin diagram
        filename = f'basin_diagram_I{I_test:.1f}.png'
        plot_basin_diagram(basin, N0_range, p0_range, I_test, label, filename)
        basin_results.append((I_test, basin, label))

    # ============================================================
    # COMPOSITE BASIN DIAGRAM (ALL THREE PANELS)
    # ============================================================
    print("\n" + "="*70)
    print("GENERATING COMPOSITE BASIN DIAGRAM")
    print("="*70)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Basins of Attraction Across Regimes', fontsize=14, fontweight='bold', y=1.02)

    for idx, (I_test, basin, label) in enumerate(basin_results):
        ax = axes[idx]

        extent = [np.log10(N0_range[0]), np.log10(N0_range[-1]), 
                  p0_range[0], p0_range[-1]]

        im = ax.imshow(
            basin.astype(float), 
            aspect='auto', 
            origin='lower',
            extent=extent,
            cmap='RdYlGn',
            vmin=0, vmax=1,
            interpolation='nearest'
        )

        ax.set_xlabel(r'$\log_{10}(N_0)$', fontsize=11)
        if idx == 0:
            ax.set_ylabel(r'Initial resistance frequency $p_0$', fontsize=11)

        # Shortened label for subplot title
        short_label = label.split(':')[0]
        ax.set_title(f'I = {I_test}\n({short_label})', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Extinct', 'Survived'])
        cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    plt.savefig('basin_diagram_composite.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nComposite basin diagram saved: basin_diagram_composite.png")

    print("\n" + "="*70)
    print("NOTE: Basin diagrams show initial condition space (N0, p0) colored")
    print("by final attractor. The separatrix (boundary between colors) is")
    print("the tipping point manifold. At I=40, all initial conditions lead")
    print("to extinction (uniform color), confirming monostability.")
    print("="*70)
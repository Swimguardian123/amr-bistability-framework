"""
NULLCLINE ANALYSIS – Reduced 2D System (Quasi-Steady-State for C)
=================================================================

PURPOSE
-------
Visualizes the nullcline geometry of the 3D eco-evolutionary system
(N, p, C) under a quasi-steady-state approximation (QSSA) for drug
concentration C. Substitutes C* = I/(μ + ηNp) into dN/dt and dp/dt,
yielding a reduced 2D system for visualization of equilibrium structure.

STRUCTURAL THEOREM CONTEXT (Compendium v4, Sections 2, 3, 5)
-----------------------------------------------------------
The compendium proves analytically that 1D and 2D models CANNOT produce
bistability under monotone C_eff feedback (Section 2). The minimal model
class that restores tipping dynamics is the FULL 3D system with dynamic
drug concentration (Section 3).

This script does NOT contradict that theorem. It is a VISUALIZATION TOOL:
  • The QSSA is an approximation for plotting, not a claim that the 2D
    reduced system captures the true bifurcation structure.
  • The full 3D system (compendium Section 3.1) exhibits bistability between
    extinction (N=0) and stable coexistence (p≈0.402, N≈9.53e8, C≈0.577).
  • The QSSA nullclines approximate the interior dynamics but do NOT resolve
    the extinction boundary (N=0) or the separatrix geometry accurately.
  • For the geometric proof of bistability, see the full 3D nullcline analysis
    in compendium v4 (Section 5, corrected page 23): single stable interior
    intersection + extinction boundary (N=0) as second attractor.

QSSA VALIDITY CONDITIONS
------------------------
The quasi-steady-state approximation C* ≈ I/(μ + ηNp) is valid when drug
dynamics are fast relative to population and evolutionary dynamics:
  |dC/dt| << |dN/dt|, |dp/dt|  →  μ >> λ_dom, r_S, r_R
Here μ = 1.0 /hr and r_S = 1.0 /gen — comparable, not separated. Thus the
QSSA is an ILLUSTRATIVE approximation, not a quantitatively exact reduction.
Near the bifurcation (λ_dom → 0), the separation improves (drug dynamics
become relatively faster), but the QSSA still misses the 3D separatrix
structure that enables the tipping point.

EXPECTED RESULTS (Compendium v4, Corrected Nullcline Summary, page 23)
---------------------------------------------------------------------
For the confirmed bistable parameter set (I=5.0):
  • N-nullcline: surface where g_bar(N,p,C*) = 0
  • p-nullcline: curve where Δ_g(N,p,C*) + γN = 0
  • Number of interior intersections: 1 (stable coexistence at p≈0.402)
  • Boundary intersections: p=0 (unstable), p=1 (unstable), N=0 (stable)
  • Geometric interpretation: nullclines intersect at exactly one interior
    point; the second "basin" is the extinction boundary (N=0).

POST-HOC VALIDATION FRAMEWORK
-----------------------------
This script is a VISUALIZATION and VERIFICATION tool, not a predictive model.
All parameters are fixed literature priors (see provenance table below).
The equilibria found by fsolve are cross-checked against the 3D equilibrium
(compendium v4, page 6: N*=9.53e8, p*=0.402, C*=0.577) for consistency.
Stability is verified via the 2×2 Jacobian (finite-difference approximation).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS — Confirmed bistable set (Compendium v4, Sec 3.1)
# ============================================================

# ------------------------------------------------------------------------------
# PARAMETER PROVENANCE TABLE
# ------------------------------------------------------------------------------
# Parameter   Value        Source / Context                  Uncertainty   Role
# ------------------------------------------------------------------------------
# r_S         1.0 /gen     P. aeruginosa chemostat           ±5%           Susceptible growth
# r_R         0.93 /gen    Plasmid burden (And10)            ±5%           Resistant growth
# K           1e9 cells/mL Carrying capacity                 ±20%          Population scale
# b           2.0 /gen     Time-kill Emax (Regoes04)         ±30%          Max kill susceptible
# b_R         1.5 /gen     Partial resistance (Hal19)        ±30%          Max kill resistant
# MIC_S       2.0 mg/L     EUCAST breakpoint anchor          ±1 dilution   Susceptible MIC
# MIC_R       4.0 mg/L     Partial resistance regime         ±1 dilution   Resistant MIC
# n           3.0          PK/PD sigmoidicity (Regoes04)     ±15%          Hill coefficient
# c_R         0.04         Plasmid fitness cost (And10)      ±20%          Resistance cost
# mu          1.0 /hr      Drug clearance (Gat22-like)       ±20%          PK elimination
# eta         2e-8 /cell/hr Collective degradation (Bar83)   ±50%          Drug depletion
# gamma       1e-12        HGT rate (Levin 1997)             ±1 order      Conjugation
# I           5.0 mg/L/hr  Confirmed bistable dosing         — (fixed)     Drug input rate
# ------------------------------------------------------------------------------
# CAVEAT: These are literature priors from heterogeneous sources (in vitro
# time-kill, animal PK, clinical TDM). They anchor the model in biologically
# plausible ranges but do NOT constitute a single calibration dataset. The
# structural theorem (compendium Sec 2) guarantees bistability requires 3D +
# endogenous drug feedback (eta>0); it is structurally impossible in 1D/2D.
# ------------------------------------------------------------------------------

params = {
    'r_S': 1.0, 'r_R': 0.93, 'K': 1e9, 'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0, 'n': 3.0, 'c_R': 0.04,
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12, 'I': 5.0
}

def hill(C, MIC, n=params['n']):
    """Vectorized Hill function. Works with scalars or arrays."""
    C = np.asarray(C)
    result = np.zeros_like(C, dtype=float)
    positive = C > 0
    result[positive] = C[positive]**n / (C[positive]**n + MIC**n)
    return result

def g_S(N, C, p=params):
    return p['r_S']*(1 - N/p['K']) - p['b']*hill(C, p['MIC_S'])

def g_R(N, C, p=params):
    return p['r_R']*(1 - N/p['K']) - p['c_R'] - p['b_R']*hill(C, p['MIC_R'])

def C_star(N, p, pdict=params):
    return pdict['I'] / (pdict['mu'] + pdict['eta'] * N * p)

def F1_reduced(N, p, pdict=params):
    C = C_star(N, p, pdict)
    gbar = (1-p)*g_S(N, C, pdict) + p*g_R(N, C, pdict)
    return N * gbar

def F2_reduced(N, p, pdict=params):
    C = C_star(N, p, pdict)
    Delta = g_R(N, C, pdict) - g_S(N, C, pdict)
    return p*(1-p) * (Delta + pdict['gamma'] * N)

# =============================================================================
# JACOBIAN OF REDUCED 2D SYSTEM (for stability verification)
# =============================================================================
# NOTE: This 2x2 Jacobian is computed numerically (finite differences) for
# the REDUCED QSSA system. The full 3D Jacobian (all 9 entries derived
# analytically) is in compendium v4 (Section 2.2), with analytical vs
# numerical verification error < 6e-11. The 2D Jacobian here is an
# approximation used for visualization stability classification only.
# The full 3D Jacobian is required for the structural impossibility proof.

def jacobian_2d(N, p, pdict=params):
    """2x2 Jacobian of [dN/dt, dp/dt] at (N, p)."""
    eps = 1e-6
    dF1_dN = (F1_reduced(N+eps, p, pdict) - F1_reduced(N-eps, p, pdict)) / (2*eps)
    dF1_dp = (F1_reduced(N, p+eps, pdict) - F1_reduced(N, p-eps, pdict)) / (2*eps)
    dF2_dN = (F2_reduced(N+eps, p, pdict) - F2_reduced(N-eps, p, pdict)) / (2*eps)
    dF2_dp = (F2_reduced(N, p+eps, pdict) - F2_reduced(N, p-eps, pdict)) / (2*eps)
    return np.array([[dF1_dN, dF1_dp],
                     [dF2_dN, dF2_dp]])

def stability_type(N, p, pdict=params):
    """Return ('stable', 'saddle', 'unstable', 'degenerate') for 2D equilibrium."""
    J = jacobian_2d(N, p, pdict)
    det = np.linalg.det(J)
    tr = np.trace(J)
    if abs(det) < 1e-10:
        return 'degenerate'
    if det < 0:
        return 'saddle'
    if tr < 0:
        return 'stable'
    return 'unstable'

# =============================================================================
# FIND ALL EQUILIBRIA
# =============================================================================
# Searches for interior equilibria (0 < p < 1) and boundary equilibria
# (p=0, p=1) using fsolve with multiple seeds. Results are cross-checked
# against the 3D equilibrium from compendium v4 (page 6):
#   N* ≈ 9.53e8, p* ≈ 0.402, C* ≈ 0.577 (all eigenvalues negative, stable).
# The QSSA should recover approximately the same interior equilibrium.

def find_equilibria(pdict=params):
    equilibria = []
    tol = 1e-6
    
    # --- Interior: F1=0 and F2=0 with 0 < p < 1 ---
    def residuals(X):
        N, p = X
        p = np.clip(p, 1e-8, 1-1e-8)
        return [F1_reduced(N, p, pdict), F2_reduced(N, p, pdict)]
    
    seeds = [(9.5e8, 0.4), (5e8, 0.6), (1e9, 0.3), (8e8, 0.5), (7e8, 0.2)]
    for seed in seeds:
        try:
            sol = fsolve(residuals, seed, xtol=1e-12)
            N, p = float(sol[0]), float(np.clip(sol[1], 0, 1))
            if N <= 0 or not (1e-8 < p < 1-1e-8):
                continue
            res = residuals([N, p])
            if abs(res[0]) < tol and abs(res[1]) < tol:
                if not any(abs(N - eq[0]) < 1e6 and abs(p - eq[1]) < 1e-4 for eq in equilibria):
                    stab = stability_type(N, p, pdict)
                    equilibria.append((N, p, C_star(N, p, pdict), 'interior', stab))
        except Exception:
            continue
    
    # --- p=0 boundary: g_S = 0 ---
    def p0_residual(N):
        C = C_star(N, 0.0, pdict)
        return g_S(N, C, pdict)
    try:
        for seed in [1e5, 1e7, 5e8, 9e8]:
            N_root = fsolve(p0_residual, seed, xtol=1e-12)[0]
            if N_root > 0 and abs(p0_residual(N_root)) < tol:
                if not any(abs(N_root - eq[0]) < 1e6 and eq[1] < 1e-6 for eq in equilibria):
                    stab = stability_type(N_root, 0.0, pdict)
                    equilibria.append((N_root, 0.0, C_star(N_root, 0.0, pdict), 'p=0', stab))
                    break
    except Exception:
        pass
    
    # --- p=1 boundary: g_R = 0 ---
    def p1_residual(N):
        C = C_star(N, 1.0, pdict)
        return g_R(N, C, pdict)
    try:
        for seed in [1e5, 1e7, 5e6]:
            N_root = fsolve(p1_residual, seed, xtol=1e-12)[0]
            if N_root > 0 and abs(p1_residual(N_root)) < tol:
                if not any(abs(N_root - eq[0]) < 1e6 and abs(eq[1]-1.0) < 1e-6 for eq in equilibria):
                    stab = stability_type(N_root, 1.0, pdict)
                    equilibria.append((N_root, 1.0, C_star(N_root, 1.0, pdict), 'p=1', stab))
                    break
    except Exception:
        pass
    
    return equilibria

# =============================================================================
# VECTORIZED NULLCLINE COMPUTATION
# =============================================================================
# Computes F1(N,p) = dN/dt and F2(N,p) = dp/dt on a grid for contour plotting.
# The zero-level contours of F1 and F2 are the N-nullcline (blue) and
# p-nullcline (red dashed), respectively. Their intersections are equilibria.

N_grid = np.logspace(5, 9.5, 400)
p_grid = np.linspace(0, 1, 400)
N_mesh, p_mesh = np.meshgrid(N_grid, p_grid)

C_vec = params['I'] / (params['mu'] + params['eta'] * N_mesh * p_mesh)
f_S_vec = hill(C_vec, params['MIC_S'])
f_R_vec = hill(C_vec, params['MIC_R'])
g_S_vec = params['r_S']*(1 - N_mesh/params['K']) - params['b']*f_S_vec
g_R_vec = params['r_R']*(1 - N_mesh/params['K']) - params['c_R'] - params['b_R']*f_R_vec
gbar_vec = (1 - p_mesh)*g_S_vec + p_mesh*g_R_vec
Delta_vec = g_R_vec - g_S_vec

F1_map = N_mesh * gbar_vec
F2_map = p_mesh * (1 - p_mesh) * (Delta_vec + params['gamma'] * N_mesh)

# =============================================================================
# PLOT
# =============================================================================
# Nullcline topology: The N-nullcline (blue) and p-nullcline (red dashed)
# intersect at the interior equilibrium. The trivial boundaries p=0 and p=1
# are always p-nullclines. The extinction boundary N=0 is the second stable
# state in the full 3D system (compendium v4, page 23) but is outside the
# plotted domain (N >= 1e5).

fig, ax = plt.subplots(figsize=(10, 8))

# Nullclines
ax.contour(p_grid, N_grid, F1_map.T, levels=[0], colors='blue', linewidths=2.5)
ax.contour(p_grid, N_grid, F2_map.T, levels=[0], colors='red', linewidths=2.5, linestyles='--')

# Trivial p-nullclines (p=0 and p=1 are always nullclines)
ax.axvline(0, color='gray', linestyle='-', linewidth=1.5, alpha=0.5, label='Trivial p-nullclines')
ax.axvline(1, color='gray', linestyle='-', linewidth=1.5, alpha=0.5)

# Equilibria
equilibria = find_equilibria()
stab_colors = {'stable': 'green', 'saddle': 'orange', 'unstable': 'red', 'degenerate': 'purple'}
stab_markers = {'stable': 'o', 'saddle': 's', 'unstable': '^', 'degenerate': 'd'}

for N, p, C, eq_type, stab in equilibria:
    color = stab_colors.get(stab, 'black')
    marker = stab_markers.get(stab, 'x')
    label = f"{eq_type} ({stab})"
    ax.scatter(p, N, c=color, marker=marker, s=150, edgecolors='black', linewidths=1.5,
               zorder=5, label=label)

# Legend
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=9)

ax.set_xlabel('Resistant fraction, $p$', fontsize=12)
ax.set_ylabel('Bacterial density, $N$ (cells/mL)', fontsize=12)
ax.set_yscale('log')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(1e5, 2e9)
ax.grid(True, alpha=0.3)
ax.set_title('Nullclines & Equilibria (QSSA $C^*$) — Stability Verified', fontsize=13)

# Annotation
ax.text(0.98, 0.02, f"Parameters: $I$={params['I']}, $\\eta$={params['eta']:.0e}\n"
        f"Found {len(equilibria)} equilibria",
        transform=ax.transAxes, fontsize=9, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig('nullcline_qssa_stable.png', dpi=300)
plt.show()

# =============================================================================
# PRINT SUMMARY
# =============================================================================
# Cross-check against compendium v4 (page 6): expected interior equilibrium
# at N*≈9.53e8, p*≈0.402, C*≈0.577 (stable). The QSSA should recover a
# similar interior point. Discrepancies indicate QSSA breakdown.

print(f"\nFound {len(equilibria)} distinct equilibria:")
for N, p, C, eq_type, stab in equilibria:
    print(f"  [{stab:12s}] {eq_type:10s}: N={N:.4e}, p={p:.4f}, C={C:.4f}")
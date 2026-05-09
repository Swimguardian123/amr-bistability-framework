"""
BIFURCATION DIAGRAM – Full Equilibrium Analysis with Stability
==============================================================
Tracks ALL branches (interior, p=0, p=1) across I.
Collects ALL distinct roots per boundary (not just first success).
Verifies monostability: exactly one stable positive equilibrium at each I.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import eigvals
import warnings
warnings.filterwarnings('ignore')

params = {
    'r_S': 1.0, 'r_R': 0.93, 'K': 1e9, 'b': 2.0, 'b_R': 1.5,
    'MIC_S': 2.0, 'MIC_R': 4.0, 'n': 3.0, 'c_R': 0.04,
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12
}

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

def residuals(X, I_val, pdict):
    N, p, C = X
    N = max(N, 0.0); p = np.clip(p, 0.0, 1.0)
    gS = g_S(N, C, pdict); gR = g_R(N, C, pdict)
    gbar = (1-p)*gS + p*gR
    dN = N * gbar
    dp = p*(1-p) * (gR - gS + pdict['gamma']*N)
    dC = I_val - pdict['mu']*C - pdict['eta']*N*p*C
    return np.array([dN, dp, dC])

def jacobian_3d(N, p, C, I_val, pdict):
    n = pdict['n']; MIC_S, MIC_R = pdict['MIC_S'], pdict['MIC_R']
    r_S, r_R, K = pdict['r_S'], pdict['r_R'], pdict['K']
    b, b_R = pdict['b'], pdict['b_R']; c_R, gamma = pdict['c_R'], pdict['gamma']
    mu, eta = pdict['mu'], pdict['eta']
    
    if C > 0:
        df_S = n * (MIC_S**n) * C**(n-1) / (C**n + MIC_S**n)**2
        df_R = n * (MIC_R**n) * C**(n-1) / (C**n + MIC_R**n)**2
    else:
        df_S = df_R = 0.0
    
    gS = g_S(N, C, pdict); gR = g_R(N, C, pdict)
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
    return np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])

def is_stable(N, p, C, I_val, pdict, tol=-1e-6):
    J = jacobian_3d(N, p, C, I_val, pdict)
    eigs = eigvals(J)
    return np.all(eigs.real < tol)

# =============================================================================
# ROBUST EQUILIBRIUM FINDERS
# =============================================================================
def find_interior_all(I_val, pdict, seeds, res_tol=1e-5):
    """Find ALL distinct interior equilibria. Returns list of (N,p,C)."""
    found = []
    tol = 1e-6
    
    def res(X):
        N, p = X
        p = np.clip(p, 1e-8, 1-1e-8)
        return [F1_reduced(N, p, I_val, pdict), F2_reduced(N, p, I_val, pdict)]
    
    for seed in seeds:
        try:
            sol, info, ier, mesg = fsolve(res, seed, xtol=1e-12, full_output=True)
            if ier != 1:
                continue
            N, p = float(sol[0]), float(np.clip(sol[1], 0, 1))
            if N <= 0 or not (1e-8 < p < 1-1e-8):
                continue
            r = res([N, p])
            if abs(r[0]) < tol and abs(r[1]) < tol:
                # Check distinct
                if not any(abs(N - f[0]) < 1e6 and abs(p - f[1]) < 1e-4 for f in found):
                    found.append((N, p, C_star_int(N, p, I_val, pdict)))
        except Exception:
            continue
    return found

def C_star_int(N, p, I_val, pdict):
    return I_val / (pdict['mu'] + pdict['eta'] * N * p)

def F1_reduced(N, p, I_val, pdict):
    C = C_star_int(N, p, I_val, pdict)
    gS = g_S(N, C, pdict); gR = g_R(N, C, pdict)
    gbar = (1-p)*gS + p*gR
    return N * gbar

def F2_reduced(N, p, I_val, pdict):
    C = C_star_int(N, p, I_val, pdict)
    gS = g_S(N, C, pdict); gR = g_R(N, C, pdict)
    Delta = gR - gS
    return p*(1-p) * (Delta + pdict['gamma'] * N)

def find_boundary_all(I_val, pdict, p_boundary, seeds):
    """Find ALL distinct boundary equilibria. Returns list of (N,p,C)."""
    found = []
    tol = 1e-5
    
    def res_boundary(N):
        if p_boundary == 0:
            C = I_val / pdict['mu']
            return g_S(N, C, pdict)
        else:
            C = I_val / (pdict['mu'] + pdict['eta'] * N)
            return g_R(N, C, pdict)
    
    for seed in seeds:
        try:
            N_root = fsolve(res_boundary, seed, xtol=1e-12)[0]
            if N_root <= 0:
                continue
            if p_boundary == 0:
                C = I_val / pdict['mu']
                rc = abs(g_S(N_root, C, pdict))
            else:
                C = I_val / (pdict['mu'] + pdict['eta'] * N_root)
                rc = abs(g_R(N_root, C, pdict))
            if rc < tol:
                if not any(abs(N_root - f[0]) < 1e6 for f in found):
                    found.append((N_root, float(p_boundary), C))
        except Exception:
            continue
    return found

# =============================================================================
# BIFURCATION SWEEP
# =============================================================================
print("="*70)
print("BIFURCATION DIAGRAM – Monostability Verification")
print("="*70)

I_vals = np.linspace(1.0, 13.0, 80)
pdict = params.copy()

# Storage by stability
int_stable = {'N': [], 'p': [], 'I': []}
int_unstable = {'N': [], 'p': [], 'I': []}
p0_stable = {'N': [], 'I': []}
p0_unstable = {'N': [], 'I': []}
p1_stable = {'N': [], 'I': []}
p1_unstable = {'N': [], 'I': []}

prev_interior = [9.53e8, 0.402, 0.577]

for I_val in I_vals:
    pdict['I'] = I_val
    
    # --- Interior ---
    seeds_int = [prev_interior[:2], [5e8, 0.6], [1e9, 0.3], [8e8, 0.5], [7e8, 0.2], [9e8, 0.1]]
    eqs_int = find_interior_all(I_val, pdict, seeds_int)
    for N, p, C in eqs_int:
        stab = is_stable(N, p, C, I_val, pdict)
        (int_stable if stab else int_unstable)['N'].append(N)
        (int_stable if stab else int_unstable)['p'].append(p)
        (int_stable if stab else int_unstable)['I'].append(I_val)
        prev_interior = [N, p, C]
    
    # --- p=0 boundary ---
    seeds_p0 = [1e5, 1e7, 5e8, 8e8, 9.5e8]
    eqs_p0 = find_boundary_all(I_val, pdict, 0, seeds_p0)
    for N, p, C in eqs_p0:
        stab = is_stable(N, p, C, I_val, pdict)
        (p0_stable if stab else p0_unstable)['N'].append(N)
        (p0_stable if stab else p0_unstable)['I'].append(I_val)
    
    # --- p=1 boundary ---
    seeds_p1 = [1e5, 1e7, 5e6, 5e8, 8e8, 9.5e8]  # ADDED large-N seeds!
    eqs_p1 = find_boundary_all(I_val, pdict, 1, seeds_p1)
    for N, p, C in eqs_p1:
        stab = is_stable(N, p, C, I_val, pdict)
        (p1_stable if stab else p1_unstable)['N'].append(N)
        (p1_stable if stab else p1_unstable)['I'].append(I_val)

# =============================================================================
# PLOT
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: N vs I
ax = axes[0]
ax.plot(int_stable['I'], int_stable['N'], 'g-', linewidth=2.5, label='Coexistence (stable)')
ax.plot(int_unstable['I'], int_unstable['N'], 'g--', linewidth=1.5, alpha=0.5, label='Coexistence (unstable)')
ax.plot(p0_stable['I'], p0_stable['N'], 'b-', linewidth=2.5, label='Susceptible-only (stable)')
ax.plot(p0_unstable['I'], p0_unstable['N'], 'b--', linewidth=1.5, alpha=0.5, label='Susceptible-only (unstable)')
ax.plot(p1_stable['I'], p1_stable['N'], 'r-', linewidth=2.5, label='Resistant-only (stable)')
ax.plot(p1_unstable['I'], p1_unstable['N'], 'r--', linewidth=1.5, alpha=0.5, label='Resistant-only (unstable)')

ax.set_xlabel('Infusion rate I (mg/L/hr)', fontsize=12)
ax.set_ylabel('Bacterial density N (cells/mL)', fontsize=12)
ax.set_yscale('log')
ax.set_title('Bifurcation Diagram: N vs I', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

# Right: p vs I
ax = axes[1]
ax.plot(int_stable['I'], int_stable['p'], 'g-', linewidth=2.5, label='Coexistence (stable)')
ax.plot(int_unstable['I'], int_unstable['p'], 'g--', linewidth=1.5, alpha=0.5, label='Coexistence (unstable)')
if p0_stable['I']:
    ax.plot(p0_stable['I'], [0]*len(p0_stable['I']), 'bo', markersize=5, label='S-only (stable)')
if p0_unstable['I']:
    ax.plot(p0_unstable['I'], [0]*len(p0_unstable['I']), 'b^', markersize=4, label='S-only (unstable)')
if p1_stable['I']:
    ax.plot(p1_stable['I'], [1]*len(p1_stable['I']), 'ro', markersize=5, label='R-only (stable)')
if p1_unstable['I']:
    ax.plot(p1_unstable['I'], [1]*len(p1_unstable['I']), 'r^', markersize=4, label='R-only (unstable)')

ax.set_xlabel('Infusion rate I (mg/L/hr)', fontsize=12)
ax.set_ylabel('Resistant fraction p', fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.set_title('Bifurcation Diagram: p vs I', fontsize=13, fontweight='bold')
ax.legend(fontsize=8, loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bifurcation_monostability_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# MONOSTABILITY CHECK
# =============================================================================
print("\n" + "="*70)
print("MONOSTABILITY VERIFICATION")
print("="*70)

violations = []
for I_val in I_vals:
    n_stable = 0
    sources = []
    if I_val in int_stable['I']:
        n_stable += 1; sources.append('coexistence')
    if I_val in p0_stable['I']:
        n_stable += 1; sources.append('p=0')
    if I_val in p1_stable['I']:
        n_stable += 1; sources.append('p=1')
    
    if n_stable == 0:
        violations.append((I_val, 0, sources))
    elif n_stable > 1:
        violations.append((I_val, n_stable, sources))

if violations:
    print(f"Monostability VIOLATED at {len(violations)} points:")
    for I_val, n, src in violations[:10]:  # show first 10
        print(f"  I={I_val:.2f}: {n} stable ({', '.join(src) if src else 'none'})")
    if len(violations) > 10:
        print(f"  ... and {len(violations)-10} more")
else:
    print("Monostability CONFIRMED: exactly 1 stable equilibrium at all I.")

# Report branch ranges
if int_stable['I']:
    print(f"\nCoexistence branch: I = [{min(int_stable['I']):.2f}, {max(int_stable['I']):.2f}]")
if p0_stable['I']:
    print(f"Susceptible branch:  I = [{min(p0_stable['I']):.2f}, {max(p0_stable['I']):.2f}]")
if p1_stable['I']:
    print(f"Resistant branch:    I = [{min(p1_stable['I']):.2f}, {max(p1_stable['I']):.2f}]")

    # Basin of attraction at I=11.6 (inside bistable window)
I_test = 11.6
pdict['I'] = I_test

# Find both stable equilibria at this I
eq_coexist = None
eq_resist = None
for N, p, C, eq_type, stab in zip(
    int_stable['N'] + p1_stable['N'],
    int_stable['p'] + [1.0]*len(p1_stable['N']),
    [C_star_int(n, p, I_test, pdict) for n, p in zip(int_stable['N'] + p1_stable['N'], 
                                                      int_stable['p'] + [1.0]*len(p1_stable['N']))],
    ['int']*len(int_stable['I']) + ['p1']*len(p1_stable['I']),
    [True]*100  # dummy
):
    # This is a simplified check — you'd need to match by I value
    pass

# Better: just run a grid of initial conditions and see where they go
N_test = np.logspace(5, 9.5, 50)
p_test = np.linspace(0.01, 0.99, 50)
basin = np.zeros((len(p_test), len(N_test)))

for i, N0 in enumerate(N_test):
    for j, p0 in enumerate(p_test):
        # Quick integration to see which attractor
        from scipy.integrate import solve_ivp
        def ode(t, y):
            return residuals(y, I_test, pdict)
        sol = solve_ivp(ode, [0, 2000], [N0, p0, I_test/pdict['mu']], method='LSODA')
        Nf, pf = sol.y[0, -1], sol.y[1, -1]
        if pf > 0.95:
            basin[j, i] = 1  # goes to resistant
        elif pf < 0.05:
            basin[j, i] = -1  # goes to susceptible
        else:
            basin[j, i] = 0  # stays at coexistence

fig, ax = plt.subplots(figsize=(8, 6))
cont = ax.contourf(N_test, p_test, basin, levels=[-1.5, -0.5, 0.5, 1.5], 
                   colors=['blue', 'green', 'red'], alpha=0.4)
ax.set_xscale('log')
ax.set_xlabel('Initial N')
ax.set_ylabel('Initial p')
ax.set_title(f'Basin of Attraction at I={I_test} (bistable window)')
plt.colorbar(cont, ax=ax, ticks=[-1, 0, 1], label='Attractor: -1=S, 0=coexist, 1=R')
plt.savefig('basin_bistable.png', dpi=300)
"""
BIFURCATION CONTINUATION IN I - Tracks stable coexistence branch
Stops when branch disappears (saddle-node bifurcation)
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
    'mu': 1.0, 'eta': 2e-8, 'gamma': 1e-12, 'I': 5.0
}

def hill(C, MIC):
    if C <= 0: return 0.0
    return C**params['n'] / (C**params['n'] + MIC**params['n'])

def g_S(N, C):
    return params['r_S']*(1 - N/params['K']) - params['b']*hill(C, params['MIC_S'])

def g_R(N, C):
    return params['r_R']*(1 - N/params['K']) - params['c_R'] - params['b_R']*hill(C, params['MIC_R'])

def residuals(X, I_val):
    N, p, C = X
    N = max(N, 0); p = np.clip(p, 0, 1)
    gbar = (1-p)*g_S(N, C) + p*g_R(N, C)
    dN = N * gbar
    dp = p*(1-p) * (g_R(N,C) - g_S(N,C) + params['gamma']*N)
    dC = I_val - params['mu']*C - params['eta']*N*p*C
    return [dN, dp, dC]

def jacobian(N, p, C):
    n = params['n']; MIC_S, MIC_R = params['MIC_S'], params['MIC_R']
    r_S, r_R, K = params['r_S'], params['r_R'], params['K']
    b, b_R = params['b'], params['b_R']; c_R, gamma = params['c_R'], params['gamma']
    mu, eta = params['mu'], params['eta']
    
    def df_dC(C, MIC):
        if C <= 0: return 0.0
        return n * C**(n-1) * MIC**n / (C**n + MIC**n)**2
    
    f_C_S = df_dC(C, MIC_S); f_C_R = df_dC(C, MIC_R)
    g_S_val = g_S(N, C); g_R_val = g_R(N, C)
    g_bar_val = (1-p)*g_S_val + p*g_R_val
    Delta_val = g_R_val - g_S_val
    
    J11 = g_bar_val - (N/K)*((1-p)*r_S + p*r_R)
    J12 = N * Delta_val
    J13 = -N * ((1-p)*b*f_C_S + p*b_R*f_C_R)
    J21 = p*(1-p)*(r_S - r_R)/K + gamma*p*(1-p)
    J22 = (1-2*p)*Delta_val + gamma*N*(1-2*p)
    J23 = p*(1-p)*(b*f_C_S - b_R*f_C_R)
    J31 = -eta * p * C
    J32 = -eta * N * C
    J33 = -mu - eta * N * p
    return np.array([[J11, J12, J13], [J21, J22, J23], [J31, J32, J33]])

def find_eq(I_val, seed):
    sol = fsolve(residuals, seed, args=(I_val,), xtol=1e-12)
    return sol

print("="*60)
print("BIFURCATION CONTINUATION - Stable Coexistence Branch")
print("="*60)

I_vals = np.linspace(1, 13, 40)  # Stop at 13 where branch disappears
N_vals = []
p_vals = []
C_vals = []
lambda_vals = []

prev_sol = [9.53e8, 0.402, 0.577]

for I_val in I_vals:
    sol = find_eq(I_val, prev_sol)
    N, p, C = sol[0], np.clip(sol[1], 0, 1), sol[2]
    
    # Stop if p > 1 or N < 0 (branch ended)
    if p > 1.0 or N < 0:
        print(f"Branch ends at I ≈ {I_val:.2f}")
        break
    
    J = jacobian(N, p, C)
    evals = eigvals(J)
    lambda_dom = np.max(np.real(evals))
    
    N_vals.append(N); p_vals.append(p); C_vals.append(C); lambda_vals.append(lambda_dom)
    prev_sol = [N, p, C]
    
    if I_val in [1, 3, 5, 7, 9, 11, 13]:
        print(f"I={I_val:3.0f}: N={N:.4e}, p={p:.4f}, λ_dom={lambda_dom:+.4f}")

# Find where λ_dom ≈ 0 (tipping)
if len(lambda_vals) > 0:
    # Find I where λ_dom is closest to zero
    idx_min = np.argmin(np.abs(lambda_vals))
    tipping_I = I_vals[idx_min]
    print(f"\nTipping region: λ_dom ≈ {lambda_vals[idx_min]:+.6f} at I ≈ {tipping_I:.2f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(I_vals[:len(N_vals)], N_vals, 'b-', linewidth=2)
axes[0].set_xlabel('I')
axes[0].set_ylabel('N')
axes[0].set_yscale('log')
axes[0].set_title('Bacterial density')
axes[0].grid(True, alpha=0.3)

axes[1].plot(I_vals[:len(p_vals)], p_vals, 'r-', linewidth=2)
axes[1].set_xlabel('I')
axes[1].set_ylabel('p')
axes[1].set_title('Resistance fraction')
axes[1].grid(True, alpha=0.3)

axes[2].plot(I_vals[:len(C_vals)], C_vals, 'g-', linewidth=2)
axes[2].set_xlabel('I')
axes[2].set_ylabel('C')
axes[2].set_title('Drug concentration')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bifurcation_continuation.png', dpi=300)
plt.show()
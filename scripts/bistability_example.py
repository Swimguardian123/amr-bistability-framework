"""
AMR BISTABILITY MODEL: E. faecalis / Ampicillin
Complete analysis: nullclines, continuation, ASI, eigenvalues
Parameters from compendium Section 3.1
"""

import os

# Must be set before importing matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from amr.asi import asi  # noqa: E402
from amr.equilibria import prefer_interior_equilibrium, solve_equilibrium  # noqa: E402
from amr.model import jacobian, rhs  # noqa: E402
from amr.params import params_ef_ampicillin_bistable  # noqa: E402
from amr.stability import dominant_real_eigenvalue  # noqa: E402

# ============================================================
# PARAMETERS (from compendium Section 3.1)
# ============================================================

par = params_ef_ampicillin_bistable()

# ============================================================
# MODEL FUNCTIONS
# ============================================================

def dXdt(X, t, I_val):
    # kept for compatibility with the rest of this file
    return rhs(np.array(X, dtype=float), float(I_val), par).tolist()

# ============================================================
# FIND EQUILIBRIUM FOR GIVEN I
# ============================================================

def find_equilibrium(I_val, seed=(9.53e8, 0.402, 0.577)):
    res = solve_equilibrium(float(I_val), par, seed, branch="interior")
    if res is None:
        raise RuntimeError(f"Failed to find bounded equilibrium for I={I_val}")
    return res.x

# ============================================================
# CONTINUATION IN I
# ============================================================

I_vals = np.linspace(1, 15, 30)
lambda_vals = []
p_vals = []
N_vals = []
C_vals = []

print("="*60)
print("E. FAECALIS BISTABLE MODEL: CONTINUATION IN I")
print("="*60)

prev = (9.53e8, 0.402, 0.577)
for I_val in I_vals:
    eqr = prefer_interior_equilibrium(float(I_val), par, prev)
    if eqr is None:
        N, p, C = np.nan, np.nan, np.nan
        lambda_dom = np.nan
    else:
        N, p, C = float(eqr.x[0]), float(eqr.x[1]), float(eqr.x[2])
        J = jacobian(N, p, C, par)
        lambda_dom = dominant_real_eigenvalue(J)
        prev = (N, p, C)
    
    lambda_vals.append(lambda_dom)
    p_vals.append(p)
    N_vals.append(N)
    C_vals.append(C)
    
    if I_val in [1, 5, 10, 15]:
        print(f"I={I_val:3.0f}: N={N:.4e}, p={p:.4f}, C={C:.4f}, λ_dom={lambda_dom:+.6f}")

# Find tipping point
tipping_I = None
for i in range(len(lambda_vals)-1):
    if np.isfinite(lambda_vals[i]) and np.isfinite(lambda_vals[i + 1]) and lambda_vals[i] * lambda_vals[i+1] < 0:
        tipping_I = I_vals[i] + (I_vals[i+1]-I_vals[i]) * (-lambda_vals[i])/(lambda_vals[i+1]-lambda_vals[i])
        break

if tipping_I is None:
    print("\nNo λ_dom sign-crossing found on the bounded interior equilibrium branch.")
else:
    print(f"\nApprox λ_dom=0 crossing at I ≈ {tipping_I:.2f}")

# ============================================================
# ASI COMPUTATION
# ============================================================

def compute_ASI(N, p, C, I_val):
    return asi(float(N), float(p), float(C), float(I_val), par, I_ref=0.5, p_ref=0.01, N_ref=0.5)

# ============================================================
# SIMULATION: APPROACH TIPPING
# ============================================================

print("\n" + "="*60)
print("SIMULATION: Increasing I from 1 to tipping-0.5")
print("="*60)

t_max = 300
n_steps = 3000
I_max_sim = (tipping_I - 0.5) if tipping_I is not None else 11.0
I_schedule = np.linspace(1.0, I_max_sim, n_steps)

eq0 = prefer_interior_equilibrium(1.0, par, (9.53e8, 0.04, 0.576))
X0 = eq0.x if eq0 is not None else find_equilibrium(1.0)
print(f"Initial: N={X0[0]:.4e}, p={X0[1]:.4f}, C={X0[2]:.4f}")

X = X0.copy()
traj = []
lambda_traj = []
ASI_traj = []
dt = t_max / n_steps

for I_val in I_schedule:
    traj.append(X.copy())
    
    J = jacobian(float(X[0]), float(X[1]), float(X[2]), par)
    lambda_dom = dominant_real_eigenvalue(J)
    lambda_traj.append(lambda_dom)
    ASI_traj.append(compute_ASI(X[0], X[1], X[2], I_val))
    
    X = X + rhs(X, float(I_val), par) * dt
    X[0] = max(float(X[0]), 0.0)
    X[1] = float(np.clip(X[1], 0.0, 1.0))
    X[2] = max(float(X[2]), 0.0)

traj = np.array(traj)
t_span = np.linspace(0, t_max, n_steps)

# ============================================================
# PLOT RESULTS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# State variables
axes[0,0].plot(t_span, traj[:,0], 'b-', linewidth=2)
axes[0,0].set_ylabel('N (cells/mL)')
axes[0,0].set_yscale('log')
axes[0,0].set_title('Bacterial density')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(t_span, traj[:,1], 'r-', linewidth=2)
axes[0,1].set_ylabel('p (resistant fraction)')
axes[0,1].set_title('Resistance fraction')
axes[0,1].grid(True, alpha=0.3)

axes[0,2].plot(t_span, I_schedule, 'g-', linewidth=2)
axes[0,2].set_ylabel('I (dosing rate)')
axes[0,2].set_title('Forcing')
axes[0,2].grid(True, alpha=0.3)

# Stability indices
axes[1,0].plot(t_span, lambda_traj, 'm-', linewidth=2)
axes[1,0].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[1,0].set_ylabel('λ_dom')
axes[1,0].set_xlabel('Time')
axes[1,0].set_title('Dominant eigenvalue')
axes[1,0].grid(True, alpha=0.3)

axes[1,1].plot(t_span, ASI_traj, 'c-', linewidth=2)
axes[1,1].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[1,1].set_ylabel('ASI')
axes[1,1].set_xlabel('Time')
axes[1,1].set_title('AMR Stability Index')
axes[1,1].grid(True, alpha=0.3)

# Final ASI bar
axes[1,2].bar(['E. faecalis'], [ASI_traj[-1]], color='blue')
axes[1,2].axhline(y=0, color='k', linestyle='--', linewidth=1)
axes[1,2].set_ylabel('Final ASI')
axes[1,2].set_title('ASI at simulation end')
axes[1,2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ef_bistable_final.png', dpi=300)
plt.show()

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if tipping_I is None:
    print("Tipping point (λ_dom=0): not detected on bounded equilibrium branch")
else:
    print(f"Tipping point (λ_dom=0): I = {tipping_I:.2f}")
print(f"Final λ_dom: {lambda_traj[-1]:.6f}")
print(f"Final ASI: {ASI_traj[-1]:.6f}")
print(f"Final p: {traj[-1,1]:.4f}")
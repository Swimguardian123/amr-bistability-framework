"""
whe21_analysis_final.py
Empirical validation of surrogate ASI using Whe21 dense time‑series data.
Computes ASI for each isolate over time, plots mean ± SEM, overlays MIC and treatment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns

# ---------- 1. Model parameters (same as before) ----------
Delta_r = -0.11
b = 2.0
b_R = 2.0
n = 3.0
MIC_S_mero = 0.25
mu = 1.0
C_ref = 0.5 / mu

def hill(C, MIC):
    if C <= 0:
        return 0.0
    return C**n / (C**n + MIC**n)

lambda_ref = Delta_r + b * hill(C_ref, MIC_S_mero) - b_R * hill(C_ref, 1e6)
ASI_denom = abs(lambda_ref)
print(f"Reference λ_ref = {lambda_ref:.6f}, ASI denominator = {ASI_denom:.6f}")

def surrogate_asi(mic, c_drug):
    lam = Delta_r + b * hill(c_drug, MIC_S_mero) - b_R * hill(c_drug, mic)
    if lam >= 0:
        return 0.0
    return -lam / ASI_denom

# ---------- 2. Load Excel data ----------
file_path = "whe21_data.xlsx"   # adjust if needed

# Sheet with isolate data
sheet_name = 'Figure 1c-1e, 3b, 4b, 4d'
df_isolates = pd.read_excel(file_path, sheet_name=sheet_name, header=0)

# Column names as printed
df_isolates['day'] = df_isolates['Days in study ']   # note trailing space
df_isolates['meropenem_mic'] = df_isolates['Meropenem MIC (ug/mL)']
df_isolates['source'] = df_isolates['Isolate source']
df_isolates['mechanism'] = df_isolates['Lung lineage']

# Drop rows with missing MIC
df = df_isolates.dropna(subset=['meropenem_mic']).copy()

# Convert to numeric (just in case)
df['meropenem_mic'] = pd.to_numeric(df['meropenem_mic'], errors='coerce')
df = df.dropna(subset=['meropenem_mic'])

# ---------- 3. Drug concentration (fixed for simplicity, as in Chu22) ----------
C_fixed = 10.0   # mg/L (typical meropenem steady‑state)
df['ASI'] = df['meropenem_mic'].apply(lambda mic: surrogate_asi(mic, C_fixed))

# ---------- 4. Daily aggregation ----------
daily_stats = df.groupby('day').agg(
    mean_ASI=('ASI', 'mean'),
    sem_ASI=('ASI', 'sem'),
    mean_MIC=('meropenem_mic', 'mean'),
    sem_MIC=('meropenem_mic', 'sem'),
    n_isolates=('ASI', 'count')
).reset_index()

# Keep days with at least 2 isolates for error bars
daily_stats = daily_stats[daily_stats['n_isolates'] >= 2]

# ---------- 5. Treatment timeline from sheet 'Figure 1a, 5a' ----------
df_treat = pd.read_excel(file_path, sheet_name='Figure 1a, 5a', header=0)
# The first column is 'Days in study', then antibiotics columns
df_treat['day'] = df_treat['Days in study']
# We'll manually define treatment phases based on the paper and the table
# Day 2-3: Piperacillin/tazobactam + Colistin; Day 3-4: + Meropenem; Day 4-13: Colistin only
treatments = [
    (2, 3, 'Piperacillin/tazobactam\n+ Colistin', 'orange'),
    (3, 4, '+ Meropenem', 'red'),
    (4, 13, 'Colistin only', 'green')
]

# ---------- 6. Spearman correlation ----------
corr, p_val = spearmanr(df['ASI'], np.log2(df['meropenem_mic']))
print(f"Spearman correlation ASI vs log2(MIC): r = {corr:.3f}, p = {p_val:.2e}")

# ---------- 7. Plot ----------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Treatment shading
for start, end, label, color in treatments:
    ax1.axvspan(start, end, alpha=0.2, color=color, label=label if start==2 else "")
    mid = (start + end) / 2
    ax1.text(mid, 0.95, label, ha='center', va='top', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# ASI
ax1.errorbar(daily_stats['day'], daily_stats['mean_ASI'], yerr=daily_stats['sem_ASI'],
             fmt='o-', color='blue', capsize=3, label='ASI (surrogate)', linewidth=2)

# MIC on secondary y‑axis
ax2 = ax1.twinx()
ax2.errorbar(daily_stats['day'], daily_stats['mean_MIC'], yerr=daily_stats['sem_MIC'],
             fmt='s-', color='red', capsize=3, label='Meropenem MIC (μg/mL)', linewidth=2, alpha=0.7)

ax1.set_xlabel('Day of infection')
ax1.set_ylabel('Surrogate ASI')
ax2.set_ylabel('Meropenem MIC (μg/mL)')
ax1.set_title('Whe21: ASI and resistance over time (C = 10 mg/L fixed)')
ax1.grid(alpha=0.3)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('whe21_asi_trend.png', dpi=300)
plt.show()

# ---------- 8. Boxplot by resistance mechanism ----------
df_mech = df[df['mechanism'] != 'Unknown'].copy()
if len(df_mech) > 0:
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df_mech, x='mechanism', y='ASI')
    plt.xticks(rotation=45)
    plt.ylabel('Surrogate ASI')
    plt.xlabel('Resistance mechanism')
    plt.title('ASI by resistance mechanism (Whe21)')
    plt.tight_layout()
    plt.savefig('whe21_asi_by_mechanism.png', dpi=300)
    plt.show()

# ---------- 9. Sensitivity analysis for C ----------
print("\nSensitivity of mean ASI on days 3–4 to C:")
C_test = [5, 10, 15, 20]
for C in C_test:
    df['ASI_temp'] = df['meropenem_mic'].apply(lambda mic: surrogate_asi(mic, C))
    mean_asi = df[df['day'].isin([3,4])]['ASI_temp'].mean()
    print(f"C = {C:2d} mg/L -> mean ASI on days 3-4 = {mean_asi:.3f}")

print("\nAnalysis complete. Figures saved as 'whe21_asi_trend.png' and 'whe21_asi_by_mechanism.png'")
"""
Empirical Validation of Surrogate ASI using Chu22 P. aeruginosa Serial Isolate Data
=====================================================================================
Final version with:
- Organism-specific parameter adjustment (b_R = 2.0 for P. aeruginosa efflux/porin resistance)
- N/K term explicitly included
- Reference lambda computed with actual MIC_R
- Full statistical framework
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from scipy.stats import mannwhitneyu, spearmanr, bootstrap
from sklearn.metrics import roc_curve, auc

# ============================================================
# 1. MODEL PARAMETERS
# ============================================================
# THEORETICAL base parameters (from main pharmacodynamic model)
r_S = 1.0
r_R = 0.93
K = 1e9
Delta_r = r_R - r_S - 0.04   # = -0.11
b = 2.0
n = 3.0
MIC_S_mero = 0.25
MIC_S_cefe = 1.0
MIC_R_ref = 4.0
gamma = 1e-12
mu = 1.0
C_ref = 0.5 / mu

# EMPIRICAL adjustment for P. aeruginosa meropenem resistance:
# Resistance in this dataset is primarily OprD porin loss / efflux pump
# overexpression, which increases MIC but does NOT reduce the intrinsic
# kill rate (b_R ≈ b). This differs from target-modification resistance
# (e.g., β-lactamase) where b_R < b.
b_R_empirical = 2.0

# High-density infection assumption
N_over_K = 0.95
N_fixed = N_over_K * K

def hill(C, MIC):
    if C <= 0:
        return 0.0
    return C**n / (C**n + MIC**n)

# Reference lambda at low drug (C_ref = 0.5)
# Full form: lambda_ref = Delta_r + b*f_S - b_R*f_R - (r_R-r_S)*(N/K) + gamma*N
lambda_ref = (Delta_r
              + b * hill(C_ref, MIC_S_mero)
              - b_R_empirical * hill(C_ref, MIC_R_ref)
              - (r_R - r_S) * N_over_K
              + gamma * N_fixed)

ASI_denom = abs(lambda_ref)
print(f"Reference λ_ref = {lambda_ref:.6f}, ASI denominator = {ASI_denom:.6f}")
print(f"  (b_R = {b_R_empirical} for P. aeruginosa efflux/porin mechanism)")
print(f"  (N/K = {N_over_K}, N = {N_fixed:.2e}, gamma*N = {gamma*N_fixed:.6f})")

def surrogate_asi(mic, c_drug, drug='meropenem'):
    """
    Surrogate ASI for patient isolate data.
    
    Uses organism-specific b_R = 2.0 to match P. aeruginosa meropenem
    resistance mechanism (OprD loss/efflux), where resistance increases
    MIC without reducing intrinsic antibiotic susceptibility (b_R ≈ b).
    """
    if drug == 'meropenem':
        MIC_S = MIC_S_mero
    else:
        MIC_S = MIC_S_cefe
    
    lam = (Delta_r
           + b * hill(c_drug, MIC_S)
           - b_R_empirical * hill(c_drug, mic)
           - (r_R - r_S) * N_over_K
           + gamma * N_fixed)
    
    if lam >= 0:
        return 0.0
    return -lam / ASI_denom

# ============================================================
# 2. LOAD DATA (unchanged)
# ============================================================
folder = "chu22_data"
csv_files = glob.glob(os.path.join(folder, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in folder '{folder}'.")

all_data = []
for file in csv_files:
    df = pd.read_csv(file)
    base = os.path.basename(file).replace(".csv", "")
    df['sample_id'] = base
    all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

def parse_mic(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str):
        val = val.strip()
        if val.startswith('<='):
            return float(val[2:])
        elif val.startswith('>='):
            return float(val[2:])
        elif val.startswith('<'):
            return float(val[1:])
        elif val.startswith('>'):
            return float(val[1:])
    return float(val)

for col in ['Meropenem', 'Cefepime']:
    if col in data.columns:
        data[col] = data[col].apply(parse_mic)

data = data.dropna(subset=['Meropenem'])
data['is_resistant'] = data['Meropenem'] >= 16

# ============================================================
# 3. SENSITIVITY ANALYSIS OF C
# ============================================================
print("\n=== Sensitivity analysis of assumed drug concentration (C) ===")
C_range = [5, 8, 10, 12, 15, 20]
results = []
for C in C_range:
    data['ASI'] = data['Meropenem'].apply(lambda x: surrogate_asi(x, C, 'meropenem'))
    non_res = data[~data['is_resistant']]['ASI'].dropna()
    res = data[data['is_resistant']]['ASI'].dropna()
    if len(non_res) > 1 and len(res) > 1:
        stat, p = mannwhitneyu(non_res, res, alternative='two-sided')
        mean_diff = res.mean() - non_res.mean()
        pooled_std = np.sqrt(((len(non_res)-1)*non_res.var() + (len(res)-1)*res.var()) / (len(non_res)+len(res)-2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan
        results.append({'C (mg/L)': C, 'p-value': p, 'mean_diff': mean_diff, 'Cohens_d': cohens_d})

df_sens = pd.DataFrame(results)
print(df_sens.round(4))

C_fixed = 10.0
data['ASI'] = data['Meropenem'].apply(lambda x: surrogate_asi(x, C_fixed, 'meropenem'))

# ============================================================
# 4. DESCRIPTIVE STATISTICS & EFFECT SIZE
# ============================================================
non_res_asi = data[~data['is_resistant']]['ASI'].values
res_asi = data[data['is_resistant']]['ASI'].values

mean_diff = np.mean(res_asi) - np.mean(non_res_asi)
pooled_std = np.sqrt(((len(non_res_asi)-1)*np.var(non_res_asi, ddof=1) + 
                      (len(res_asi)-1)*np.var(res_asi, ddof=1)) / 
                     (len(non_res_asi) + len(res_asi) - 2))
cohens_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

print(f"\nDescriptive statistics (C = {C_fixed} mg/L, b_R = {b_R_empirical}):")
print(f"  Non-resistant (n={len(non_res_asi)}): mean ASI = {np.mean(non_res_asi):.4f} ± {np.std(non_res_asi, ddof=1):.4f}")
print(f"  Resistant (n={len(res_asi)}): mean ASI = {np.mean(res_asi):.4f} ± {np.std(res_asi, ddof=1):.4f}")
print(f"  Mean difference: {mean_diff:.4f}")
print(f"  Cohen's d: {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'} effect)")

# ============================================================
# 5. FIGURE 1: BOXPLOT AND HISTOGRAM
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.boxplot(data=data, x='is_resistant', y='ASI', ax=axes[0], hue='is_resistant', palette='Set2', legend=False)
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['Non‑resistant (MIC≤8)', 'Resistant (MIC≥16)'])
axes[0].set_ylabel('Surrogate ASI')
axes[0].set_title('ASI by resistance category')

colors = ['blue', 'red']
labels = ['Non‑resistant', 'Resistant']
for i, grp in enumerate([False, True]):
    subset = data[data['is_resistant'] == grp]['ASI'].dropna()
    axes[1].hist(subset, bins=20, alpha=0.5, label=labels[i], color=colors[i])
axes[1].set_xlabel('ASI')
axes[1].set_ylabel('Frequency')
axes[1].set_title('ASI distribution')
axes[1].legend()

stat, p_val = mannwhitneyu(non_res_asi, res_asi, alternative='two-sided')
axes[2].text(0.5, 0.6, f'Mann-Whitney U test\np = {p_val:.2e}\nCohen\'s d = {cohens_d:.3f}', 
             ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
axes[2].axis('off')

plt.tight_layout()
plt.savefig('asi_boxplot_hist.png', dpi=300)
plt.show()

# ============================================================
# 6. BOOTSTRAP CI FOR MEAN ASI DIFFERENCE
# ============================================================
def diff_means(x, y):
    return np.mean(y) - np.mean(x)

bootstrap_result = bootstrap((non_res_asi, res_asi), diff_means, 
                              n_resamples=1000, method='BCa', random_state=42)
print(f"\nBootstrap 95% CI for mean ASI difference (Resistant - Non-resistant):")
print(f"  [{bootstrap_result.confidence_interval.low:.4f}, {bootstrap_result.confidence_interval.high:.4f}]")

# ============================================================
# 7. PERMUTATION TEST (TWO-TAILED)
# ============================================================
def perm_test_two_tailed(x, y, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)
    combined = np.concatenate([x, y])
    n1 = len(x)
    obs_diff = abs(np.mean(y) - np.mean(x))
    count = 0
    for _ in range(n_perm):
        permuted = rng.permutation(combined)
        new_x = permuted[:n1]
        new_y = permuted[n1:]
        new_diff = abs(np.mean(new_y) - np.mean(new_x))
        if new_diff >= obs_diff:
            count += 1
    return count / n_perm

perm_p = perm_test_two_tailed(non_res_asi, res_asi)
print(f"Two-tailed permutation test p-value: {perm_p:.6f}")

# ============================================================
# 8. ROC CURVE AND AUC
# ============================================================
y_true = data['is_resistant'].astype(int).values
y_score = -data['ASI'].values

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC (using -ASI): {roc_auc:.3f}")

# Bootstrap 95% CI for AUC (stratified)
n_bootstrap = 2000
rng = np.random.default_rng(42)
boot_aucs = []

for _ in range(n_bootstrap):
    idx_0 = rng.choice(np.where(y_true == 0)[0], size=np.sum(y_true == 0), replace=True)
    idx_1 = rng.choice(np.where(y_true == 1)[0], size=np.sum(y_true == 1), replace=True)
    idx = np.concatenate([idx_0, idx_1])
    
    y_true_boot = y_true[idx]
    y_score_boot = y_score[idx]
    
    if len(np.unique(y_true_boot)) < 2:
        continue
    fpr_b, tpr_b, _ = roc_curve(y_true_boot, y_score_boot)
    boot_aucs.append(auc(fpr_b, tpr_b))

boot_aucs = np.array(boot_aucs)
ci_low = np.percentile(boot_aucs, 2.5)
ci_high = np.percentile(boot_aucs, 97.5)
print(f"Bootstrap 95% CI for AUC: [{ci_low:.3f}, {ci_high:.3f}]")

# Benchmark comparison
benchmark_auc = 0.73
print(f"\nComparison to clinical benchmark:")
print(f"  Surrogate ASI AUC: {roc_auc:.3f} (95% CI: {ci_low:.3f}-{ci_high:.3f})")
print(f"  fAUC/MIC benchmark (Mar23): {benchmark_auc:.3f}")
if roc_auc > benchmark_auc:
    print(f"  → Surrogate ASI exceeds benchmark by +{roc_auc - benchmark_auc:.3f}")
else:
    print(f"  → Within {abs(roc_auc - benchmark_auc):.3f} of benchmark")

# Plot ROC curve
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkgreen', linewidth=2.5,
         label=f'ASI (AUC = {roc_auc:.3f}, 95% CI {ci_low:.3f}-{ci_high:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Random classifier')
plt.fill_between(fpr, tpr, alpha=0.15, color='darkgreen')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve: Surrogate ASI Predicting Meropenem Resistance', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('asi_roc.png', dpi=300)
plt.show()

# ============================================================
# 8b. LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION (LOPOCV)
# ============================================================
print("\n" + "="*60)
print("LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION (LOPOCV)")
print("="*60)

# Extract patient IDs for ALL isolates (not just longitudinal)
data['patient'] = data['sample_id'].str.extract(r'(\d+)SP', expand=False)
patients = data['patient'].dropna().unique()

if len(patients) < 2:
    print("Insufficient patients for LOPOCV.")
else:
    lopo_results = []
    
    for held_out_patient in patients:
        # Split
        train = data[data['patient'] != held_out_patient].copy()
        test = data[data['patient'] == held_out_patient].copy()
        
        if len(test) == 0 or train['is_resistant'].nunique() < 2:
            print(f"  Patient {held_out_patient}: skipped (insufficient data or no class variation in train)")
            continue
        
        # Train: find optimal threshold on training data using Youden's J
        y_train = train['is_resistant'].astype(int).values
        score_train = -train['ASI'].values  # higher = more likely resistant
        
        fpr_tr, tpr_tr, thresh_tr = roc_curve(y_train, score_train)
        
        # Youden's J = sensitivity + specificity - 1
        j_scores = tpr_tr - fpr_tr
        best_idx = np.argmax(j_scores)
        best_thresh = thresh_tr[best_idx]
        
        # Test: apply threshold to held-out patient
        y_test = test['is_resistant'].astype(int).values
        score_test = -test['ASI'].values
        y_pred = (score_test >= best_thresh).astype(int)
        
        # Confusion matrix (raw counts)
        tp = int(np.sum((y_test == 1) & (y_pred == 1)))
        tn = int(np.sum((y_test == 0) & (y_pred == 0)))
        fp = int(np.sum((y_test == 0) & (y_pred == 1)))
        fn = int(np.sum((y_test == 1) & (y_pred == 0)))
        
        n_total = len(y_test)
        acc = (tp + tn) / n_total if n_total > 0 else np.nan
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        
        n_res_test = int(np.sum(y_test == 1))
        n_sus_test = int(np.sum(y_test == 0))
        
        lopo_results.append({
            'patient': held_out_patient,
            'n_test': n_total,
            'n_resistant': n_res_test,
            'n_susceptible': n_sus_test,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'threshold': best_thresh,
            'accuracy': acc,
            'sensitivity': sens,
            'specificity': spec,
            'ppv': ppv,
            'npv': npv
        })
        
        print(f"  Patient {held_out_patient}: n={n_total} "
              f"(R={n_res_test}, S={n_sus_test}), "
              f"thresh={best_thresh:.4f}, "
              f"Acc={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}")
    
    if lopo_results:
        df_lopo = pd.DataFrame(lopo_results)
        print(f"\n--- LOPOCV Summary (n={len(df_lopo)} patients) ---")
        print(f"  Accuracy:   {df_lopo['accuracy'].median():.3f} "
              f"(IQR: {df_lopo['accuracy'].quantile(0.25):.3f}-{df_lopo['accuracy'].quantile(0.75):.3f})")
        print(f"  Sensitivity: {df_lopo['sensitivity'].median():.3f} "
              f"(IQR: {df_lopo['sensitivity'].quantile(0.25):.3f}-{df_lopo['sensitivity'].quantile(0.75):.3f})")
        print(f"  Specificity: {df_lopo['specificity'].median():.3f} "
              f"(IQR: {df_lopo['specificity'].quantile(0.25):.3f}-{df_lopo['specificity'].quantile(0.75):.3f})")
        print(f"  PPV: {df_lopo['ppv'].median():.3f} "
              f"(IQR: {df_lopo['ppv'].quantile(0.25):.3f}-{df_lopo['ppv'].quantile(0.75):.3f})")
        print(f"  NPV: {df_lopo['npv'].median():.3f} "
              f"(IQR: {df_lopo['npv'].quantile(0.25):.3f}-{df_lopo['npv'].quantile(0.75):.3f})")
        
                # Threshold stability
        thresholds = df_lopo['threshold'].values
        thresh_median = np.median(thresholds)
        thresh_min = np.min(thresholds)
        thresh_max = np.max(thresholds)
        thresh_std = np.std(thresholds, ddof=1)
        thresh_mean = np.mean(thresholds)
        
        if thresh_std > 1e-10 and abs(thresh_mean) > 1e-10:
            cv = thresh_std / abs(thresh_mean)
            cv_str = f"{cv:.2%}"
        else:
            cv_str = "<0.01% (essentially constant)"
        
        print(f"\n  Threshold stability: median = {thresh_median:.6f}, "
              f"range = [{thresh_min:.6f}, {thresh_max:.6f}], "
              f"std = {thresh_std:.2e}, CV = {cv_str}")
        
        # Pooled (micro-averaged) confusion matrix using RAW counts
        total_tp = int(df_lopo['tp'].sum())
        total_tn = int(df_lopo['tn'].sum())
        total_fp = int(df_lopo['fp'].sum())
        total_fn = int(df_lopo['fn'].sum())
        total_all = total_tp + total_tn + total_fp + total_fn
        
        pooled_acc = (total_tp + total_tn) / total_all if total_all > 0 else np.nan
        pooled_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else np.nan
        pooled_spec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else np.nan
        pooled_ppv = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else np.nan
        pooled_npv = total_tn / (total_tn + total_fn) if (total_tn + total_fn) > 0 else np.nan
        
        print(f"\n  Pooled (micro-averaged) performance:")
        print(f"    Accuracy:    {pooled_acc:.3f}")
        print(f"    Sensitivity: {pooled_sens:.3f}")
        print(f"    Specificity: {pooled_spec:.3f}")
        print(f"    PPV:         {pooled_ppv:.3f}")
        print(f"    NPV:         {pooled_npv:.3f}")
        print(f"    Total isolates: {total_all} (TP={total_tp}, TN={total_tn}, FP={total_fp}, FN={total_fn})")

# ============================================================
# 9. SPEARMAN CORRELATION
# ============================================================
data['log2MIC'] = np.log2(data['Meropenem'])
corr, p_corr = spearmanr(data['ASI'], data['log2MIC'])
print(f"\nSpearman correlation ASI vs log2(MIC): r = {corr:.3f}, p = {p_corr:.2e}")

# ============================================================
# 10. CEFEPIME ANALYSIS
# ============================================================
if 'Cefepime' in data.columns:
    data_cefe = data.dropna(subset=['Cefepime'])
    C_cefe = 20.0
    data_cefe['ASI_cefe'] = data_cefe['Cefepime'].apply(
        lambda x: surrogate_asi(x, C_cefe, 'cefepime'))
    data_cefe['is_resistant_cefe'] = data_cefe['Cefepime'] >= 32
    
    non_res_c = data_cefe[~data_cefe['is_resistant_cefe']]['ASI_cefe'].dropna()
    res_c = data_cefe[data_cefe['is_resistant_cefe']]['ASI_cefe'].dropna()
    
    if len(non_res_c) > 1 and len(res_c) > 1:
        stat_c, p_c = mannwhitneyu(non_res_c, res_c, alternative='two-sided')
        mean_diff_c = res_c.mean() - non_res_c.mean()
        pooled_std_c = np.sqrt(((len(non_res_c)-1)*non_res_c.var() + (len(res_c)-1)*res_c.var()) / 
                               (len(non_res_c)+len(res_c)-2))
        cohens_d_c = mean_diff_c / pooled_std_c if pooled_std_c > 0 else np.nan
        print(f"\nCefepime analysis (C = {C_cefe} mg/L):")
        print(f"  Non-resistant (n={len(non_res_c)}): mean ASI = {non_res_c.mean():.4f}")
        print(f"  Resistant (n={len(res_c)}): mean ASI = {res_c.mean():.4f}")
        print(f"  Mann-Whitney p = {p_c:.4e}, Cohen's d = {cohens_d_c:.3f}")
    else:
        print("\nNot enough resistant isolates for cefepime analysis.")
else:
    print("\nCefepime column not found. Skipping.")

# ============================================================
# 11. LONGITUDINAL PLOT
# ============================================================
day_map = {
    '01SP9': 1, '01SP10': 12, '05SP1A': 1, '09SP1': 1, '09SP4': 12,
    '09ST1': 6, '10SP1': 1, '10SP4': 12, '13SP1': 1, '13SP3': 5,
    '14SP1': 1, '14SP3': 11, '42SP1': 1, '42SP': 6, '42ST1': 7,
    '45SP1': 1, '46SP1B': 1, '46SP3': 11,
}

if day_map:
    data['day'] = data['sample_id'].map(day_map)
    data_long = data.dropna(subset=['day']).copy()
    
    if not data_long.empty:
        data_long['patient'] = data_long['sample_id'].str.extract(r'(\d+)SP', expand=False)
        patient_day_counts = data_long.groupby('patient')['day'].nunique()
        serial_patients = patient_day_counts[patient_day_counts >= 2].index
        
        if len(serial_patients) > 0:
            plt.figure(figsize=(10, 6))
            for pat in serial_patients:
                sub = data_long[data_long['patient'] == pat].sort_values('day')
                mean_asi = sub.groupby('day')['ASI'].mean()
                sem_asi = sub.groupby('day')['ASI'].sem()
                plt.errorbar(mean_asi.index, mean_asi, yerr=sem_asi, 
                           marker='o', label=f'Patient {pat}', capsize=3, linewidth=1.5)
            plt.xlabel('Day of infection', fontsize=12)
            plt.ylabel('Mean ASI', fontsize=12)
            plt.title('Longitudinal ASI Trends (Serial Isolates)', fontsize=13, fontweight='bold')
            plt.legend(fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('asi_longitudinal.png', dpi=300)
            plt.show()
        else:
            print("\nNo patients with at least two sampling days.")
    else:
        print("\nNo samples matched the day mapping.")
else:
    print("\nSkipping longitudinal plot: day_map not provided.")

print("\n=== Analysis complete. ===")
"""
AUC Approximation from Published OR (Dichotomized Predictor)
Assumes: underlying continuous predictor is normally distributed 
         with equal variance in cases/controls; threshold at optimal point
"""

import numpy as np
from scipy.stats import norm

# Inputs from Mar23
OR = 0.25                    # Protective OR (high fAUC/MIC → low resistance)
CI_lower = 0.11
CI_upper = 0.61
prevalence = 0.19
n_total = 256

def or_to_auc_normal(or_val):
    """
    Back-calculate AUC from OR at optimal threshold.
    Assumption: equal-variance normal distributions, threshold at midpoint.
    
    For OR < 1 (protective), we compute AUC for the flipped predictor
    and report 1 - AUC to maintain AUC > 0.5 convention.
    """
    # Use absolute log-OR since direction is arbitrary for AUC
    abs_or = max(or_val, 1.0 / or_val)  # fold-change > 1
    sqrt_or = np.sqrt(abs_or)
    
    # At optimal threshold for equal-variance normals: Se = Sp
    # DOR = Se²/(1-Se)² => Se = sqrt(DOR)/(1+sqrt(DOR))
    se_opt = sqrt_or / (1 + sqrt_or)
    
    # Se = Phi(d/2) where d is standardized mean difference
    # d = 2 * Phi^{-1}(Se)
    d = 2 * norm.ppf(se_opt)
    
    # AUC = Phi(d/sqrt(2)) for equal-variance binormal ROC
    auc = norm.cdf(d / np.sqrt(2))
    
    return auc, se_opt, d

def auc_ci_delta_method(or_val, or_lower, or_upper, n):
    """
    Compute AUC CI using delta method.
    Var(ln(OR)) ≈ (ln(OR_upper) - ln(OR_lower)) / (2*1.96)
    """
    # SE of log-OR
    se_log_or = (np.log(or_upper) - np.log(or_lower)) / (2 * 1.96)
    
    # AUC as function of log(OR) - use numerical derivative
    eps = 1e-4
    auc_center, _, _ = or_to_auc_normal(or_val)
    auc_plus, _, _ = or_to_auc_normal(np.exp(np.log(or_val) + eps))
    auc_minus, _, _ = or_to_auc_normal(np.exp(np.log(or_val) - eps))
    
    dauc_dlogor = (auc_plus - auc_minus) / (2 * eps)
    se_auc = abs(dauc_dlogor) * se_log_or
    
    z = 1.96
    auc_low = max(0.5, auc_center - z * se_auc)
    auc_high = min(1.0, auc_center + z * se_auc)
    
    return auc_low, auc_high

# Compute
auc_est, se_est, d_est = or_to_auc_normal(OR)
auc_low, auc_high = auc_ci_delta_method(OR, CI_lower, CI_upper, n_total)

print("="*60)
print("AUC APPROXIMATION FROM PUBLISHED OR")
print("="*60)
print(f"Source: Mar23 (n={n_total}, prevalence={prevalence:.1%})")
print(f"Published OR: {OR} (95% CI: {CI_lower}-{CI_upper})")
print(f"\nAssumptions:")
print(f"  1. fAUC/MIC is continuously distributed")
print(f"  2. Equal variance in resistant vs susceptible populations")
print(f"  3. Threshold ≥494 was at/near the optimal (Youden) point")
print(f"\nResults:")
print(f"  Standardized effect size: d = {d_est:.3f}")
print(f"  Implied Se/Sp at threshold: {se_est:.1%}")
print(f"  Estimated AUC: {auc_est:.3f} (95% CI: {auc_low:.3f}-{auc_high:.3f})")

if auc_est > 0.8:
    print(f"\nInterpretation: Excellent discrimination (AUC > 0.8)")
elif auc_est > 0.7:
    print(f"\nInterpretation: Good discrimination (AUC 0.7-0.8)")
else:
    print(f"\nInterpretation: Moderate discrimination (AUC < 0.7)")

print(f"\nLimitation: This is an approximation. True AUC requires")
print(f"individual-level data or the published ROC curve.")

# Add this after the main computation:

print("\n" + "="*60)
print("SENSITIVITY: Unequal variance assumption")
print("="*60)

# If case variance = 2× control variance (common in biomarkers):
# d_unequal = d_equal * sqrt(2/(1+ratio)) — rough correction
ratio = 2.0  # case variance / control variance
d_unequal = d_est * np.sqrt(2 / (1 + ratio))
auc_unequal = norm.cdf(d_unequal / np.sqrt(2))
print(f"  If case variance = {ratio}× control variance:")
print(f"    d = {d_unequal:.3f}, AUC = {auc_unequal:.3f}")
print(f"  Range across plausible variance ratios (0.5–2.0):")
for r in [0.5, 1.0, 2.0]:
    d_r = d_est * np.sqrt(2 / (1 + r))
    auc_r = norm.cdf(d_r / np.sqrt(2))
    print(f"    ratio={r}: AUC = {auc_r:.3f}")
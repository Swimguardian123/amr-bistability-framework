# Acute Stability Index (ASI) for Early Warning of Resistance Tipping in Antimicrobial Resistance

**Research code for:** Acute Stability Index + EWS for Antimicrobial Resistance
**Authors:** Sanath Seetharam
**Contact:** sanath.seetharam@gmail.com
---

## Overview

This repository contains the complete computational framework for deriving, validating, and empirically testing a **surrogate Acute Stability Index (ASI)** — a model-based early warning signal (EWS) for predicting antibiotic resistance tipping points in bacterial populations. The surrogate ASI approximates the dominant eigenvalue of the full pharmacodynamic Jacobian using only clinically observable quantities (MIC, drug concentration), bypassing the need for unobservable state variables (bacterial density, resistant fraction).

### Key Contributions

1. **Theoretical surrogate**: An analytically tractable approximation to the p-eigenvalue (J₂₂) derived from a 3D pharmacodynamic competition model
2. **Validation**: Correlation r = 0.996 between surrogate and true eigenvalue across tipping trajectories
3. **Sensitivity analysis**: Stability-checked bifurcation analysis across 11 biologically relevant parameters
4. **Empirical validation**: LOPOCV-validated prediction of meropenem resistance (AUC = 0.981) on Chu22 *P. aeruginosa* serial isolates
5. **Benchmark comparison**: Surrogate exceeds published fAUC/MIC benchmark (AUC ≈ 0.73) by +0.25

---

## Repository Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── chu22_data/                         # Patient isolate data (user-provided)
│   └── *.csv                          # MIC panels per patient
│
├── theory/                             # Theoretical validation scripts
│   ├── surrogate_validation.py        # Surrogate vs J22 eigenvalue comparison
│   ├── nullcline_analysis.py          # 2D QSSA nullclines with stability checks
│   └── bifurcation_monostability.py   # Full branch tracking + bistability analysis
│
├── sensitivity/                        # Parameter robustness analyses
│   ├── comprehensive_sensitivity.py   # 11-parameter stability-checked sweep
│   └── eta_sensitivity.py             # Tipping threshold vs antibiotic uptake
│
├── ews_comparison/                     # Early warning signal benchmarking
│   └── ews_comprehensive.py           # ASI vs variance/AR1/skew/kurt/CV/spectral
│
├── empirical/                          # Clinical validation
│   ├── empirical_validation.py        # Chu22 analysis: ROC, LOPOCV, longitudinal
│   └── auc_approximation_mar23.py     # Benchmark extraction from published OR
│
└── figures/                            # Generated figures (gitignored)
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/Swimguardian123/amr-bistability-framework.git
cd amr-bistability-framework
pip install -r requirements.txt
```

**Dependencies:** Python ≥3.8, NumPy, SciPy, Matplotlib, Pandas, Seaborn, scikit-learn

### 2. Run Theoretical Validation

```bash
python theory/surrogate_validation.py
```

Validates the surrogate eigenvalue against the true Jacobian p-eigenvalue (J₂₂) along a tipping trajectory. Expected output: correlation r ≈ 0.996, MAE ≈ 4×10⁻⁴.

### 3. Run Sensitivity Analysis

```bash
python sensitivity/comprehensive_sensitivity.py
```

Computes critical infusion rate I* across 11 parameters with full stability verification and bisection refinement. Runtime: ~10–15 minutes.

### 4. Run EWS Comparison

```bash
python ews_comparison/ews_comprehensive.py
```

Generates the 5×3 comprehensive figure comparing surrogate ASI against six classical EWS indicators (variance, AR1, skewness, kurtosis, CV, spectral ratio) across far/mid/near-tipping conditions.

### 5. Run Empirical Validation

**Prerequisite:** Place Chu22 MIC panel CSV files in `chu22_data/`.

```bash
python empirical/empirical_validation.py
```

Performs leave-one-patient-out cross-validation (LOPOCV) on *P. aeruginosa* serial isolates. Expected output: median accuracy = 1.000, pooled AUC ≈ 0.98.

---

## Key Parameters & Assumptions

| Parameter | Value | Biological Meaning |
|-----------|-------|-------------------|
| r_S | 1.0 hr⁻¹ | Sensitive strain max growth rate |
| r_R | 0.93 hr⁻¹ | Resistant strain max growth rate |
| c_R | 0.04 hr⁻¹ | Resistance metabolic cost |
| b | 2.0 hr⁻¹ | Max antibiotic kill rate (sensitive) |
| b_R | 1.5 hr⁻¹ (theory) / 2.0 hr⁻¹ (empirical) | Max kill rate (resistant) |
| MIC_S | 2.0 mg/L (theory) / 0.25 mg/L (mero) | Sensitive MIC |
| MIC_R | 4.0 mg/L | Resistant MIC |
| n | 3.0 | Hill coefficient (steepness) |
| K | 1×10⁹ cells/mL | Carrying capacity |
| μ | 1.0 hr⁻¹ | Antibiotic clearance rate |
| η | 2×10⁻⁸ L/(cell·hr) | Bacterial antibiotic uptake |
| γ | 1×10⁻¹² | Frequency-dependent competition |

### Important Assumptions

1. **N/K = 0.95** for patient isolates: clinical infections are high-density (established infection)
2. **b_R = 2.0** for *P. aeruginosa* empirical validation: efflux/porin-loss resistance does not reduce intrinsic antibiotic susceptibility
3. **Equal-variance binormal** for Mar23 benchmark approximation (sensitivity to violation: AUC shifts < ±0.03)
4. **Wright-Fisher diffusion** for demographic noise: justified for large well-mixed populations (N ~ 10⁹)

---

## Data Requirements

### Chu22 Dataset
The empirical validation requires MIC panel data from Chu et al. (2022) or equivalent serial isolate dataset:
- One CSV file per patient
- Columns: `Meropenem`, `Cefepime` (MIC values as strings, e.g., `"<=0.5"`, `">=32"`)
- Row: `sample_id` column with patient-timepoint IDs (e.g., `"01SP9"`, `"01SP10"`)
- Place all CSVs in `chu22_data/`

**Note:** The `day_map` dictionary in `empirical_validation.py` must be populated with your patient-specific sampling day mapping.

---

## Main Results Summary

| Analysis | Key Result | Script |
|----------|-----------|--------|
| Surrogate vs true eigenvalue | r = 0.996, MAE = 4×10⁻⁴ | `surrogate_validation.py` |
| Baseline critical infusion | I* ≈ 7.84 mg/L/hr | `comprehensive_sensitivity.py` |
| Most sensitive parameter | MIC_S (SensIdx = +1.94) | `comprehensive_sensitivity.py` |
| Monostability | Confirmed; narrow bistable window (ΔI ≈ 0.1) at saddle-node | `bifurcation_monostability.py` |
| ASI vs classical EWS | ASI shows monotonic trend; classical indicators mixed/noisy | `ews_comprehensive.py` |
| Empirical ROC (Chu22) | AUC = 0.981 (95% CI: 0.972–0.991) | `empirical_validation.py` |
| LOPOCV | Median accuracy = 1.000, pooled sensitivity = 1.000 | `empirical_validation.py` |
| Benchmark comparison | Exceeds fAUC/MIC (AUC ≈ 0.73) by +0.25 | `empirical_validation.py` |

---

## Reproducibility Notes

- **Random seeds:** All stochastic simulations use `np.random.default_rng(42)` or `random_state=42`
- **Numerical tolerances:** Equilibrium finding uses scale-invariant residual checks (`‖res‖/‖state‖ < 10⁻⁵`)
- **Stability verification:** All equilibria verified via 3×3 Jacobian eigenvalues (`Re(λ) < 0` for all)
- **Bisection precision:** Critical I* determined to ±0.05 mg/L/hr

## License

MIT License — see `LICENSE` for details.

---

## Contact

For questions about the code or model, please open a GitHub issue or contact sanath.seetharam@gmail.com.

# Paper Outline: Executable Hybrid Tumor Dynamics in EGFR-Mutant NSCLC

## Title (working)
Hybrid executable dynamical models improve intervention-aware response prediction in EGFR-mutant NSCLC.

## Abstract
- Problem: static ML fails under treatment perturbations.
- Method: RNA -> pathways -> constrained parameter net -> ODE simulation -> resistance time.
- Results: in-distribution parity, perturbation robustness superiority, interpretable parameters.
- Impact: executable biological models as translational infrastructure.

## 1. Introduction
- Drug response prediction limitations in oncology.
- Limits of black-box models for intervention planning.
- Need for executable hybrid models.

## 2. Methods
### 2.1 Cohort and data
- TCGA-LUAD EGFR-mutant primary tumors.
- RNA-seq processing and clinical endpoint construction.

### 2.2 Pathway feature construction
- Curated pathway sets (EGFR, PI3K/AKT, MAPK, apoptosis, cell cycle, EMT/stress).

### 2.3 Hybrid model
- Parameter network mapping pathways -> r, alpha, K.
- Tumor ODE with drug term.
- Resistance-time extraction.

### 2.4 Baselines
- Cox PH.
- MLP regression.

### 2.5 Evaluation protocol
- 5-fold CV, RMSE, C-index, Brier, calibration.
- Paired tests and bootstrap CIs.

### 2.6 Perturbation studies
- Reduced dose, delayed start, interruption.

### 2.7 Ablations
- No pathway compression.
- No ODE.
- Unconstrained params.
- Frozen params.
- No drug term.

## 3. Results
- Main benchmark table.
- Calibration and confidence intervals.
- Perturbation robustness figure.
- Ablation degradation figure.
- Parameter distribution interpretation.

## 4. Discussion
- Why executability matters for treatment shifts.
- Limits and external validation path.
- Implications for computational operating systems in biology.

## 5. Reproducibility Statement
- Code, seeds, configs, environment.

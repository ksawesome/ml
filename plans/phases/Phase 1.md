---
title: Phase 1
updated: 2025-11-19 18:21:11Z
created: 2025-11-19 18:20:57Z
latitude: 23.21563540
longitude: 72.63694150
altitude: 0.0000
---

# PHASE 1 — CORE MACHINE LEARNING (RESEARCH-GRADE, EXHAUSTIVE PLAN)

**Duration:** 8–12 weeks
**Purpose:** turn mathematical & coding fluency (Phase 0) into research-grade mastery of classical ML — algorithms, theory, evaluation, and robust engineering for tabular, structured, and time-series problems. After this phase we can design, critique, and extend classical methods and confidently reproduce/benchmarks papers and baselines.

---

# Phase Overview

This phase closes the gap between “knowing algorithms” and “owning them as tools for research.” We’ll master theory (duals, kernels, impurity measures), statistical guarantees, numerical behaviour, feature & pipeline engineering, and time-series modelling — then demonstrate that mastery by reproducing a non-trivial classical ML paper and building a production-quality tabular pipeline. The phase unlocks robust baselines you’ll need to compare against in later deep learning and RAG work.

---

# Modules (6 total)

## Module A — Linear & Generalized Linear Models (GLMs)

**Purpose:** rigorous foundations for linear models used as interpretable baselines and building blocks.

**Fundamentals**

* OLS, weighted least squares, ridge, Lasso (primal/dual views).
* Logistic regression: likelihood, convexity, regularization.
* Generalized linear models and canonical links.

**Deep theoretical coverage**

* Derive closed form OLS; prove optimality in Gaussian noise.
* Bias–variance decomposition proofs for OLS and ridge.
* MLE derivation for logistic regression; convergence properties of Newton vs. GD.
* L1 regularization: subgradient KKT conditions, coordinate descent convergence.

**Mathematical requirements**

* Multivariable calculus, convex analysis, KKT conditions, matrix calculus.

**Hands-on**

* Implement OLS, ridge, Lasso (coordinate descent), logistic regression (Newton, SGD, LBFGS) from scratch.
* Numerical stability: demo ill-conditioned XTX and Tikhonov regularization.

**Expert details often missed**

* Practical differences between L2 penalty and weight decay in implementations.
* Effects of feature scaling on conditioning and penalization.
* Correct intercept handling with regularization.

**Mini-projects / proofs**

* Prove ridge solves Tikhonov-regularized least squares and reduces condition number.
* Implement Newton’s method and show quadratic convergence on logistic loss.

**Deliverables**

* Repo: GLMs folder with implementations, unit tests, and stability experiments (conditioning vs. regularization).

---

## Module B — Support Vector Machines & Kernel Methods

**Purpose:** deep, proof-level understanding of SVMs and kernel learning — essential for reasoning about function classes and representer theorem.

**Fundamentals**

* Hard/soft margin SVM, hinge loss, slack variables.
* Dual formulation, Lagrangian, KKT.
* Kernel trick, RKHS basics, Mercer’s theorem.

**Deep theoretical coverage**

* Full derivation of primal→dual SVM; complementary slackness implications (support vectors).
* Proof sketch of representer theorem and kernel ridge regression equivalence.
* Kernel PCA derivation.

**Mathematical requirements**

* Convex duality, Lagrange multipliers, integral operator eigenanalysis basics.

**Hands-on**

* Implement primal and dual SVM (QP solver use allowed but implement dual objective and KKT checks).
* Implement RBF and polynomial kernels; demonstrate kernel PCA.

**Expert details often missed**

* Numerical stability of kernel matrices (centering, regularization).
* Memory/compute scaling: when to use low-rank approximations.
* Precise conditions for Mercer kernels.

**Mini-projects**

* Reproduce a dual SVM training on a nontrivial dataset; visualize support vectors and KKT residuals.

**Deliverables**

* Notebook: SVM derivations + dual implementation + kernel experiments.

---

## Module C — Trees, Ensembles & Boosting Theory

**Purpose:** exhaustive understanding of decision trees, impurity measures, ensemble mechanics and modern gradient-boosted machines.

**Fundamentals**

* CART algorithm, impurity measures (Gini, entropy), pruning strategies.
* Bagging, Random Forest theory.
* Boosting: AdaBoost derivation as coordinate descent; gradient boosting framework.
* XGBoost algorithmic details; CatBoost categorical handling; LightGBM leaf-wise growth mechanics.

**Deep theoretical coverage**

* Derive Gini and entropy impurity formulas and link to classification error bounds.
* Show boosting as stagewise additive modelling; connect to gradient descent in function space.
* Analyze variance reduction from bagging; bias/variance tradeoff with depth and ensembles.

**Mathematical requirements**

* Probability concentration, VC dimension intuition, functional gradient calculus.

**Hands-on**

* Implement CART splitting and pruning from scratch (binary trees).
* Implement a simplified gradient boosting machine (GBM) that fits least-squares residuals.
* Run and profile XGBoost, LightGBM, CatBoost; reproduce leaf-wise vs. level-wise behaviour.

**Expert details often missed**

* Why LightGBM’s leaf-wise growth can overfit and how to regularize it.
* Practical effect of categorical encoding strategies (target encoding leakage!).
* Importance sampling & histogram binning internals for speed.

**Mini-projects**

* Build a mini-GBM and show equivalence to functional gradient boosting on a synthetic regression problem.
* Demonstrate leakage introduced by naive target encoding and fix it.

**Deliverables**

* Repo: CART + mini-GBM + comparative study vs. XGBoost/LightGBM/CatBoost with ablations.

---

## Module D — Unsupervised Learning & Representation (Spectral / Manifold)

**Purpose:** understand classic unsupervised algorithms, manifold learning and spectral techniques used for structure discovery and pre-processing.

**Fundamentals**

* K-means, EM for GMMs, PCA/ICA, spectral clustering.
* Manifold methods: Laplacian eigenmaps, Isomap.
* Metric learning (pre-deep era): Mahalanobis/LDA, LMNN.

**Deep theoretical coverage**

* EM derivation & convergence properties (monotonicity of likelihood).
* Spectral clustering: justify Laplacian normalization and cut objectives (NCut).
* ICA identifiability conditions.

**Mathematical requirements**

* Linear algebra spectral theory, probabilistic latent variable models.

**Hands-on**

* Implement EM for GMM with covariance choices and EM convergence diagnostics.
* Implement spectral clustering via graph Laplacian eigenvectors.
* Implement Isomap embedding and compare with PCA on nonlinear manifolds.

**Expert details often missed**

* How initialization affects EM convergence to local optima; model selection (BIC/AIC) for GMM component choice.
* Graph construction choices (k-NN vs. ε-ball) and normalization effects in spectral methods.

**Mini-projects**

* Reconstruct a nonlinear manifold (Swiss roll) with Isomap vs. PCA vs. t-SNE (t-SNE optional).

**Deliverables**

* Notebooks: EM diagnostics, spectral clustering experiments, manifold reconstruction.

---

## Module E — Anomaly Detection & Metric Learning

**Purpose:** mastery of anomaly detection algorithms and metric learning that remain valuable baselines and pre-deep era perspectives.

**Fundamentals**

* One-Class SVM theory, kernel density estimation, Isolation Forest mechanics.
* Metric learning: LMNN, triplet loss (pre-deep era formulations).

**Deep theoretical coverage**

* One-Class SVM dual view and ν-parameter interpretation.
* Isolation Forest splitting heuristics and expected path length derivation.

**Hands-on**

* Implement One-Class SVM with kernel and analyze influence of ν.
* Implement Isolation Forest logic and benchmark on imbalanced datasets.

**Mini-projects**

* Construct synthetic anomaly datasets and compare detection ROC across methods.

**Deliverables**

* Repo: anomaly detection suite with evaluation harness.

---

## Module F — Time-Series & Structured Data Models

**Purpose:** rigorous understanding of classical time-series models used for forecasting and as baselines for sequence models.

**Fundamentals**

* AR, MA, ARIMA derivations; seasonal models (SARIMA).
* Holt–Winters exponential smoothing.
* State-space models basics and Kalman filter intuition.

**Deep theoretical coverage**

* Stationarity tests, ACF/PACF interpretation, Box-Jenkins model identification.
* Likelihood estimation for ARIMA; invertibility and causality conditions.

**Hands-on**

* Implement ARIMA estimation (Yule-Walker, MLE), seasonal decomposition, forecast intervals.
* Implement Holt–Winters and compare to ARIMA on seasonality datasets.

**Expert details often missed**

* How differencing affects variance and forecasting intervals.
* Importance of residual checks (white noise) and Ljung-Box test.

**Mini-projects**

* Build forecasting pipeline with automatic model selection (AIC/BIC) and backtesting.

**Deliverables**

* Time-series forecasting toolbox + backtest scripts.

---

## Module G — Model Evaluation, Feature Engineering & Pipelines

**Purpose:** research-grade evaluation, proper CV, calibration and production-quality pipelines for tabular work.

**Fundamentals**

* Cross-validation strategies (k-fold, stratified, time-series CV).
* Calibration (Platt scaling, isotonic regression).
* Feature engineering best practices; leakage, target encoding countermeasures.
* Bias–variance tradeoffs, class imbalance handling.

**Deep coverage**

* Variance estimates for CV scores and nested CV for model selection.
* Theoretical effect of calibration on predicted probabilities.

**Hands-on**

* Build robust pipeline scaffolding: preprocessing, feature unions, CV folds, leakage checks, model selection and ensemble stacking.
* Implement nested CV and proper hyperparameter search.

**Mini-projects**

* Reproduce a full Kaggle-style pipeline with careful leakage prevention and stacked ensemble.

**Deliverables**

* Production-quality pipeline repo (configurable, tested) + experiment logs.

---

# Weekly Breakdown (8–12 weeks schedule)

Assume 20–25 hours/week. If 12 weeks preferred, slow deeper per module.

**Week 1 — GLMs (Module A)**

* Day 1–2: OLS derivations, coding OLS & ridge.
* Day 3–4: Lasso theory + coordinate descent implementation.
* Day 5: Logistic regression MLE; implement Newton & SGD.
* Deliverable: GLM repo + numerical stability report.

**Week 2 — SVMs & Kernels (Module B)**

* Primal→dual derivation, KKT checks.
* Implement dual SVM + kernel PCA.
* Deliverable: SVM notebook with kernel experiments.

**Week 3 — Trees & CART (Module C start)**

* CART algorithm, impurity math (Gini/entropy proofs).
* Implement tree splitting & pruning.
* Deliverable: CART implementation with pruning experiments.

**Week 4 — Ensembles & Boosting (Module C cont.)**

* Bagging theory; implement Random Forest skeleton.
* Implement mini-GBM; compare to XGBoost basics.
* LightGBM vs. CatBoost behavior study.
* Deliverable: mini-GBM + performance/overfitting analysis.

**Week 5 — Unsupervised & Spectral (Module D)**

* Implement EM for GMM; study convergence.
* Spectral clustering + Laplacian derivations.
* Manifold learning (Isomap).
* Deliverable: manifold reconstruction notebook.

**Week 6 — Anomaly & Metric Learning (Module E)**

* Implement One-Class SVM, Isolation Forest.
* Metric learning experiments (LMNN toy).
* Deliverable: anomaly detection benchmark.

**Week 7 — Time-Series (Module F)**

* ARIMA, SARIMA derivations; implement estimation & forecast intervals.
* Holt-Winters and backtesting.
* Deliverable: forecasting pipeline with backtest.

**Week 8 — Pipelines, CV, Calibration (Module G)**

* Nested CV, model selection, calibration methods.
* Build full Kaggle-style pipeline skeleton.
* Deliverable: pipeline + evaluation harness.

**Week 9–10 — Integration & Reproducibility**

* Integrate individual modules into a cohesive repo.
* Hardening, unit tests, deterministic seeds, experiment logging (W&B).
* Deliverable: polished repo.

**Week 11–12 — Capstone Reproduction + Paper**

* Select a classical ML paper (e.g., Friedman 2001 “Greedy Function Approximation: A Gradient Boosting Machine” or Breiman Random Forests original) and reproduce core experiments + one novel ablation or extension (e.g., different loss, regularizer, or categorical strategy).
* Write reproducibility report and short paper draft (~6–8 pages) with code and logs.

---

# Mastery Checks (must pass **all**)

**Theory/proof checks**

* Derive dual of soft-margin SVM and prove complementary slackness implies support vector set.
* Prove boosting reduces exponential loss (AdaBoost derivation).
* Show EM increases likelihood each iteration (monotonicity proof sketch).

**Coding checks**

* End-to-end GLM/SVM/mini-GBM implementations with unit tests.
* Kernel matrix conditioning experiment demonstrating need for regularization.
* Time-series backtest showing correct forecast intervals and model selection.

**Experimental checks**

* Reproduce a baseline result from chosen classical paper within ±5% of reported metric (where possible) or reproduce qualitative behavior (rankings, gain curves).
* Nested CV used for model selection with correct leakage prevention.

**Communication check**

* Produce a reproducibility report that contains: precise data preprocessing, seeds, hyperparameters, compute used, and ablation results.

---

# Capstone — Reproduce a Classical ML Paper (publication-grade)

**Target:** pick one reproducible, classical ML paper. Two recommended options:

1. **Friedman (2001)** — Gradient Boosting Machine (GBM)
2. **Breiman (2001)** — Random Forests

**Scope**

* Reproduce the experiments and reported phenomena (rankings, error decay, variable importance).
* Implement your own version (mini-GBM or RF) and compare to modern implementations (XGBoost/LightGBM/CatBoost).
* Run at least two meaningful ablations (e.g., loss function variant, regularization scheme, categorical encoding method).
* Produce a 6–8 page reproducibility/report paper with figures, tables, code links, and well-documented scripts to rerun experiments.

**Deliverable**

* Public GitHub repo with scripts to reproduce experiments end-to-end, Jupyter notebooks, hyperparameter configs, and the reproducibility PDF.

---

# What NOT to skip (strict)

* Do **not** use scikit-learn/XGBoost as a black box until you’ve implemented simplified versions. Implementation reveals algorithmic tradeoffs.
* Do **not** do CV without nested CV for hyperparameter selection in final evaluation.
* Do **not** use naive target encoding without leakage controls (time folds, out-of-fold estimates).
* Do **not** skip unit tests and deterministic seeds for experiments.

---

# Prerequisites from Phase 0 (explicit)

You must already have mastered:

* Linear algebra (SVD, pseudoinverse) and matrix calculus.
* Multivariable calculus and gradients.
* Probability basics (expectation, KL, MLE).
* Numerical stability and conditioning intuition.
* Comfortable implementing numerical algorithms in Python/NumPy and basic CUDA/C++ knowledge for profiling.

If any of these are weak, fix them before continuing.

---

# Failure Modes & Fixes

**Failure: Overfitting to dataset quirks instead of learning method behavior.**
Fix: use multiple datasets, synthetic datasets, and robust backtests; report confidence intervals.

**Failure: Leakage during preprocessing (target encoding, scaling).**
Fix: move preprocessing inside CV folds; enforce out-of-fold transforms.

**Failure: Accepting black-box performance without understanding internals.**
Fix: implement mini versions and profile behaviour (metadata: node counts, leaf depths, residual distribution).

**Failure: Poor experimental hygiene (no seeds, no logs).**
Fix: adopt experiment template with seeds, config files, deterministic behavior, and W&B.

---

# Resources (depth only — textbooks, original papers, and canonical docs)

**Textbooks & Monographs**

* *The Elements of Statistical Learning* — Hastie, Tibshirani, Friedman (especially chapters on trees, boosting, SVMs).
* *Pattern Recognition and Machine Learning* — Christopher Bishop (EM, mixture models, kernel methods).
* *Time Series Analysis: Forecasting and Control* — Box, Jenkins, Reinsel (ARIMA theory).
* *Understanding Machine Learning: From Theory to Algorithms* — Shai Shalev-Shwartz & Shai Ben-David (convexity, SVM theory).

**Original Papers (must read and annotate)**

* Breiman, L. (2001). *Random Forests*. Machine Learning.
* Friedman, J.H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics.
* Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks*. (original SVM)
* Dempster, A.P., Laird, N.M. & Rubin, D.B. (1977). *Maximum Likelihood from Incomplete Data via the EM Algorithm*.

**Practical / Implementation References**

* XGBoost paper and documentation (Chen & Guestrin).
* LightGBM and CatBoost technical docs (internal algorithmic details).
* scikit-learn source for CART & EM implementations (study code).

**Advanced Tutorials / Notes**

* “A Short Introduction to Boosting” — Freund & Schapire (foundational intuition).
* “Kernel Methods” lecture notes (various universities) — for Mercer theorem proofs.

---

We’ll execute this plan exactly: implement core algorithms from first principles, produce sound experiments, and end with a reproducible paper-quality capstone. Tell me which classical paper (Friedman GBM or Breiman Random Forests — or name another) you want to reproduce, and I’ll produce the exact experiment list, datasets, hyperparameters, and shell scripts to run everything.

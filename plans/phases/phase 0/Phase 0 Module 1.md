---
title: Phase 0 Module 1
updated: 2025-11-23 19:25:37Z
created: 2025-11-23 19:25:20Z
latitude: 23.21563540
longitude: 72.63694150
altitude: 0.0000
---

# 1\. Module Summary & Mastery Goals

**Module:** Linear Algebra & Functional Spaces (Module 1 — Phase 0)  
**Endgame (what “done” looks like):** we can read any ML paper that uses SVD, PCA, spectral methods, matrix calculus, or linear-operator proofs and reproduce the math, implement the algorithms from scratch, and judge numerical stability and complexity in experiments.

**Mastery Goals (4–6 bullets)**

- **Conceptual:** Own geometric & operator intuitions — subspace projection, low-rank structure, singular values as “energy”, conditioning as “how much the output moves when input moves”, and how spectra control optimization and generalization.

- **Mathematical:** Reproduce from first principles: SVD derivation + Eckart–Young (best rank-k Frobenius approx), spectral decomposition for symmetric matrices, conditions for Moore–Penrose pseudoinverse, matrix calculus rules for Jacobian/Hessian of common ML losses (logistic, MSE), and basic functional-space definitions (norms, inner product, Hilbert space idea).

- **Implementation:** From-scratch implementations: power method, randomized SVD sketch, manual pseudoinverse via SVD, PCA via covariance eigen-decomp and via gradient descent, and matrix calculus to code analytic gradients (no autograd) for small models.

- **Research/Engineering Taste:** Know when numeric issues matter (when to use SVD vs. normal equations; when to regularize; diagnosing rank-deficiency and near-singularity), and prefer numerically stable algorithms by default.

- **Transfer:** Be explicitly able to apply to: PCA/representation learning, initialization & spectral analysis of neural nets, kernel eigendecompositions, spectral clustering, low-rank matrix factorization and how these feed into optimization and probabilistic analyses.

* * *

# 2\. Topic Map

We split into **6** ordered subtopics.

1. **Basic matrix algebra & geometry — Core (must master)**  
    *What & why:* Vectors, linear maps, subspaces, projections, orthogonality: foundation for everything.  
    *Prereqs:* Linear systems, basic calculus.  
    *Depth:* Exact identities, projections, orthonormal bases, Gram–Schmidt (numerical caveats).

2. **Eigenvalues, eigenvectors, and symmetric spectral theory — Core**  
    *What & why:* Diagonalization of symmetric matrices, Courant–Fischer, Rayleigh quotient — used in PCA, graph Laplacian, and stability analyses.  
    *Prereqs:* Basic matrix algebra.  
    *Depth:* Proof of spectral theorem for symmetric matrices; Rayleigh principle; interlacing.

3. **Singular Value Decomposition & low-rank approximation — Core**  
    *What & why:* SVD is the universal factorization for rectangular matrices; leads to PCA, pseudoinverse, truncated SVD for compression and denoising.  
    *Prereqs:* Eigen/spectral basics.  
    *Depth:* Full derivation, Eckart–Young proof, numerical methods (power method, bidiagonalization).

4. **Pseudoinverse, least squares, and conditioning — Important**  
    *What & why:* Moore–Penrose pseudoinverse, normal equations vs SVD solutions; condition number, perturbation bounds (Weyl’s theorem intuition).  
    *Prereqs:* SVD, optimization basics.  
    *Depth:* Derive pseudoinverse properties and show why SVD-based solve is numerically stable.

5. **Matrix calculus & derivatives (Jacobian, Hessian) — Important**  
    *What & why:* Be able to derive gradients for vector/matrix-valued functions used in ML (losses, linear layers).  
    *Prereqs:* Multivariable calc.  
    *Depth:* Index notation derivations, vector-Jacobian products, traces trick, matrix cookbook identities.

6. **Spectral graph theory & functional spaces (Hilbert view) — Important / Optional stretch**  
    *What & why:* Graph Laplacian, Fiedler vector, Laplacian eigenmaps — connects geometry, clustering, and kernels. Also norms, inner products, RKHS intuition.  
    *Prereqs:* Eigen/spectral theory.  
    *Depth:* Basic Laplacian properties, relation to cuts; quick RKHS conceptual mapping (no full kernel theory).

**Out-of-scope (for later):** full measure-theoretic functional analysis, deep operator theory, full RKHS theory, and advanced random matrix theorems — we include only the parts directly useful for ML.

* * *

# 3\. Resources & How to Use Them

**Primary backbone (main spine):**

- **Gilbert Strang — *Introduction to Linear Algebra* (selected chapters) + his MIT OCW video lectures.**  
    *How to use:* Read Strang chapter on SVD, eigenvalues, orthogonality first for intuition; watch MIT lectures paired with each reading. Use as daily backbone while doing derivations.

**Secondary deepening resources (pick selectively):**

- **Horn & Johnson — *Matrix Analysis* (targeted sections: SVD, perturbation, singular values)**  
    *How to use:* After we finish a derivation, consult Horn & Johnson for rigorous statements and perturbation bounds. Use for proofs we must reproduce exactly.

- **Golub & Van Loan — *Matrix Computations* (selected: SVD algorithms, stability)**  
    *How to use:* When implementing power method / bidiagonalization and selecting numerically stable variants, read the relevant algorithm descriptions.

- **The Matrix Cookbook (Sylvester/others) — quick identities**  
    *How to use:* Keep as a one-page cheat sheet for matrix derivatives (but derive first, then confirm here).

**Implementation / hands-on references:**

- **NumPy + SciPy docs (linalg.svd, eigh, pinv)** — use as correctness oracle and to compare speed/stability.  
    *How to use:* After coding a manual version, compare outputs and timings to NumPy implementations and explain differences.

- **Small curated repos/notebooks:** a) tiny-SVD/power-method notebook (we'll provide skeleton) b) a spectral clustering demo notebook.  
    *How to use:* Clone as reference only if stuck — our aim is manual implementation first.

**Total resources:** 6 (Strang, Horn & Johnson, Golub & Van Loan, Matrix Cookbook, NumPy/SciPy docs, curated demo notebooks). Use Strang daily, Horn/G&VL for deepening, cookbook for derivatives, NumPy for testing.

* * *

# 4\. Timeline & Block Plan (block-based; each block ≈ 8–12 hours)

We assume **no fixed deadline**; blocks let you expand. Each block lists exact tasks.

* * *

### Block 0 — Setup (2–3 hours)

**Focus:** repo + baseline tests.  
**Concept Work:** none.  
**Practice:** n/a  
**Coding:** create repo skeleton, set up virtualenv/poetry, add `requirements.txt` with numpy/scipy/jupyter, create `notebooks/` and `modules/01-...` folders. Add README with goals.  
**Deliverable:** repo skeleton + a short `run_tests.sh` that runs a minimal NumPy test (matrix multiply).

* * *

### Block 1 — Core matrix geometry & orthogonality (8–12 hours)

**Focus:** vector spaces, projections, orthonormal bases, Gram–Schmidt, numerical caveats.  
**Concept Work:**

- Read Strang: chapters on vector spaces and orthogonality. Watch corresponding MIT lecture(s).

- Write short notes: prove projection matrix formula (P = U U^\\top) for orthonormal U; derive normal equations for least squares.  
    **Practice / Exercises:**

- Do 8 derivation problems: prove uniqueness of projection, show Gram–Schmidt produces orthonormal basis (include numerical stability note).  
    **Coding:**

- Implement classical Gram–Schmidt and modified Gram–Schmidt for random full-rank matrix (size 50×20). Compare orthogonality error (||QᵀQ - I||) and runtime.  
    **Mini-deliverable:** Jupyter notebook `01-geometry.ipynb` with proofs and Gram–Schmidt experiments and short analysis.

* * *

### Block 2 — Eigen/symmetric spectral theory (8–12 hours)

**Focus:** spectral theorem, Rayleigh quotient, power method basics.  
**Concept Work:**

- Read Strang sections on eigenvalues; consult Horn & Johnson for Rayleigh principle. Derive Courant–Fischer in notes.  
    **Practice / Exercises:**

- Prove that symmetric matrix eigenvectors are orthogonal; show Rayleigh quotient minimization property.  
    **Coding:**

- Implement power method (dominant eigenpair) and inverse power method for shifted eigenvalues. Test on symmetric matrices and compare to `numpy.linalg.eigh`. Show convergence rates and dependence on spectral gap.  
    **Mini-deliverable:** `02-spectral.ipynb` with derivations and power-method experiments (plots of error vs iteration for different gaps).

* * *

### Block 3 — SVD & low-rank approximation (12 hours)

**Focus:** full SVD derivation, Eckart–Young theorem, truncated SVD algorithms.  
**Concept Work:**

- Read Strang SVD chapter; derive SVD existence for any matrix by relating to symmetric eigen-decomp of AᵀA and AAᵀ. Prove Eckart–Young (Frobenius form).  
    **Practice / Exercises:**

- 6 derivations: from AᵀA eigenpairs to singular values; prove best rank-k approximation statement.  
    **Coding:**

- Implement truncated SVD via: (a) direct `np.linalg.svd` call, (b) power-method-based algorithm for top-k singular vectors (use randomized initialization), (c) randomized SVD sketch (basic). Compare reconstruction error vs k and timing for matrices size 2000×1000 with low-rank + noise (synthesize).  
    **Mini-deliverable:** `03-svd.ipynb` with proofs, implementations, error vs runtime plots, and conclusion recommending algorithm by problem size and noise.

* * *

### Block 4 — Pseudoinverse & least-squares conditioning (8–10 hours)

**Focus:** Moore–Penrose pseudoinverse, normal equations vs SVD solution, condition number and perturbation intuition.  
**Concept Work:**

- Derive Moore–Penrose definition from SVD. Prove pseudoinverse gives minimum-norm solution. Read relevant Horn & Johnson sections.  
    **Practice / Exercises:**

- Show algebraic equivalence of SVD-solve and QR / normal-equation solutions; derive when normal equations magnify noise (condition number squared).  
    **Coding:**

- Implement `pinv_via_svd(A)` and solve least-squares with `pinv`, QR, and normal equations; create experiments where A is ill-conditioned and compare solution error and residual. Plot error amplification vs condition number.  
    **Mini-deliverable:** `04-pinv.ipynb` with experiments and a decision flowchart: “use SVD when… use QR when…”.

* * *

### Block 5 — Matrix calculus & trace tricks (8–12 hours)

**Focus:** practical matrix derivatives used in ML; index-notation practice.  
**Concept Work:**

- Read Matrix Cookbook sections on derivatives; derive gradients for: linear regression MSE w.r.t. weights matrix, logistic loss w.r.t. weight vector, and gradients for bilinear forms. Write by index notation and then show matrix-trace equivalents.  
    **Practice / Exercises:**

- Derive Jacobian for (f(X)=AXB + C) w.r.t. X, derivative of (|AX-b|^2). Derive Hessian of quadratic form.  
    **Coding:**

- Implement analytic gradients and compare to numeric finite differences for a small neural-layer operation (dense + nonlinearity) — compute relative error <1e-6.  
    **Mini-deliverable:** `05-matcalc.ipynb` with derivations and gradient-check code.

* * *

### Block 6 — Spectral graph basics & functional space view (6–8 hours)

**Focus:** Laplacian, Fiedler vector, eigenmaps; Hilbert-space intuition.  
**Concept Work:**

- Read short notes on graph Laplacian properties (degree, normalized Laplacian), and Strang/notes on functional spaces (norms, inner products).  
    **Practice / Exercises:**

- Prove quadratic form relation (x^\\top L x = \\sum_{ij} w_{ij}(x_i-x_j)^2). Explain Fiedler vector and its relation to cuts.  
    **Coding:**

- Implement spectral clustering on a synthetic two-moons dataset: build similarity matrix, compute Laplacian eigenvectors, run k-means in eigenspace, compare to sklearn spectral_clustering.  
    **Mini-deliverable:** `06-graph.ipynb` spectral clustering demo + short discussion on limitations/scale.

* * *

### Block 7 — Integration mini-project prep (6–10 hours)

**Focus:** combine SVD/pinv/matrix-calculus into a coherent capstone project (see next section).  
**Concept Work:** finalize math notes to include in capstone.  
**Coding:** implement final project skeleton and reproducibility steps (requirements, seed fixed, small CI test).  
**Deliverable:** repo ready for capstone work.

* * *

# 5\. Module Project / Capstone (5–15 hours)

**Project idea:** *“SVD & PCA: From Theory to Practice”* — reproduce PCA using three routes, evaluate on synthetic and real data, and analyze stability and approximation tradeoffs.

**Dataset suggestions:**

- Synthetic low-rank matrix with Gaussian noise (controlled SNR).

- MNIST subset (first 5000 images) or Olivetti faces (small, easy to run) for visual PCA experiments.

**Required steps (concrete):**

1. **Data handling:** synthesize low-rank data (n=2000, d=1000, rank=20) and prepare MNIST/Olivetti subset.

2. **Modeling / algorithms:** implement PCA via (a) covariance eigendecomposition, (b) truncated SVD, (c) gradient-descent PCA (optimize reconstruction loss). Implement randomized SVD for faster top-k.

3. **Evaluation:** compare reconstruction error vs k, runtime, memory footprint, and sensitivity to noise and missing entries (simulate missingness). Report singular value decay plots.

4. **Ablations:** show where normal equations or naive SVD fails numerically (e.g., near-rank-deficient), and test fixes (regularization, centering, scaling).

5. **Interpretation:** write short analysis: how singular spectrum reflects data structure, when to choose each algorithm in practice.

6. **Reproducibility:** Dockerfile or `requirements.txt`, seed fixed, small CI run that executes a quick experiment.

**Stretch goals:** implement randomized subspace iteration with power iterations; show effect of number of iterations on approximation error.

**Deliverable:** a single reproducible notebook `capstone-pca.ipynb`, a 4–6 page PDF write-up with math derivations (Eckart–Young), and code in `capstone/` folder.

* * *

# 6\. Checkpoints & Self-Tests

**Module-end checklist (12 measurable items):**

1. Derive SVD from AᵀA eigen-decomposition and prove existence.

2. Prove Eckart–Young theorem (Frobenius norm).

3. Show how Moore–Penrose pseudoinverse arises from SVD and prove minimal-norm property.

4. Explain Courant–Fischer / Rayleigh quotient and use it to bound eigenvalues.

5. Implement power method and inverse-power method and demonstrate convergence rates dependent on spectral gap.

6. Implement truncated SVD by power-method-like algorithm and compare to `np.linalg.svd` on time and accuracy.

7. Derive gradients (index notation → matrix form) for MSE and logistic loss.

8. Use matrix calculus to compute Jacobian for (X\\mapsto AXB).

9. Demonstrate numerically that normal equations amplify condition-number-squared and recommend stable alternative.

10. Implement spectral clustering and explain Fiedler vector meaning on small graph.

11. Provide numeric gradient checks showing analytic gradients within 1e-6 of finite differences.

12. Produce reproducible capstone notebook that reproduces key figures with fixed seed.

**Oral exam prompts (5–10):**

- Explain why SVD exists for any rectangular matrix — sketch proof.

- What is condition number; how does it affect least-squares?

- How does Eckart–Young justify PCA as best linear compression?

- Why prefer SVD-based solve over normal equations for ill-conditioned systems?

- Interpret a singular value spectrum that sharply drops vs slowly decays.

- How does the Rayleigh quotient relate to the largest eigenvalue?

- What is the difference between left and right singular vectors physically?

**Code mastery tests (timed):**

1. *30 minutes:* Implement power method for dominant eigenpair for symmetric matrix and return eigenvalue within relative error 1e-6.

2. *90 minutes:* Implement truncated SVD for top-k singular vectors using randomized initialization + 2 power iterations; reconstruct and report Frobenius error vs np.linalg.svd on a 1000×500 synthetic matrix.

3. *45 minutes:* Implement pseudoinverse via SVD and solve a rank-deficient least-squares problem; demonstrate minimal-norm solution.

* * *

# 7\. Integration & Dependencies

**This module strongly depends on:** basic calculus (multivariable derivatives), some exposure to probability (for interpretation of random matrices and noise), and competent Python/Numpy skills.

**This module is crucial for later modules:** optimization (understanding Hessians and conditioning), probabilistic ML (PCA, factor models), deep learning (initialization, spectral norm regularization), and systems (numerical stability of training).

* * *

# 8\. Pitfalls & Adjustment Rules

**Typical failure modes:**

- *Proof paralysis:* spending all time on rigorous proofs and never implementing.

- *Tool addiction:* over-relying on NumPy/SciPy black boxes without understanding numerical tradeoffs.

- *Surface memorization:* knowing identities but not when to use them.

**Adjustment rules (practical):**

- If by end of **Block 2** you cannot implement a working power-method and show convergence behavior, **pause proofs** and spend next block on coding exercises only.

- If by end of **Block 3** reconstruction errors differ significantly from `np.linalg.svd` (>1e-6 relative) for small matrices, re-run numerical stability diagnostics (orthogonality checks, reseed RNG, use double precision).

- If ahead of schedule: implement randomized SVD power-iteration variant and add a small CI test comparing accuracy/time tradeoffs.

- If stuck on a derivation >4 hours, write down where the gap is, move on, and revisit after coding a related experiment — often numeric experiments illuminate the math.

* * *

&nbsp;

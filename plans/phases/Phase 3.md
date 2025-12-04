---
title: Phase 3
updated: 2025-11-19 18:22:20Z
created: 2025-11-19 18:22:05Z
latitude: 23.21563540
longitude: 72.63694150
altitude: 0.0000
---

# PHASE 3 — ADVANCED METHODS & THEORY (RESEARCH-GRADE, EXHAUSTIVE PLAN)

**Duration:** 12–20 weeks (recommendation: 16 weeks for depth; accelerate to 12 if you already know parts)
**Outcome:** Research-level mastery of optimization theory for deep models, probabilistic deep learning, modern generative modeling families, representation learning theory and practice, foundational causality for ML, and advanced RL paradigms. You will be able to read, dissect, reproduce, and extend top-conference papers in these areas and design rigorous experiments that test theoretical claims.

---

## PHASE OVERVIEW

Phase 3 turns engineering competence into principled research capability. It equips you to:

* reason about *why* modern training recipes work (sharp vs flat minima, implicit bias),
* build, compare, and analyze probabilistic models (VAEs, flows, EBMs, score-based/diffusion),
* run and evaluate generative models with rigorous metrics,
* use representation learning methods (contrastive, self-supervised) with theoretical grounding (InfoNCE etc.),
* apply causal tools to propose and evaluate interventions/robustness, and
* design RL algorithms beyond textbook basics (model-based, offline, imitation).

This phase unlocks principled contributions: novel optimization recipes, better generative objectives, improved representation methods, causal robustness investigations, and solid RL baselines. It’s the bridge from “implementer” to “researcher who can invent and critique.”

---

## MODULES (7 total)

Each module below is self-contained but interdependent. Expect to move between modules for cross-pollination (e.g., use contrastive embeddings inside generative or RL projects).

---

### Module 1 — Optimization Theory for Deep Learning

**Purpose**
Understand optimization geometry, implicit biases of algorithms, and advanced optimizers so you can propose and test training improvements with theoretical backing.

**Fundamental concepts**

* Loss landscape geometry: critical points, saddle points, basin widths
* Sharp vs. flat minima, generalization connections (Hessian spectrum)
* Implicit regularization of SGD and small-batch noise
* Natural gradient (Fisher information metric)
* Trust-region methods (TRPO ideas, second-order approximations)
* Mirror descent and multiplicative updates
* Lipschitz continuity, smoothness, strong convexity (definitions & uses)

**Deep theoretical coverage**

* Show relationship between Hessian trace/eigenvalues and "flatness" measures
* Derive natural gradient update: connection to KL constraint and Fisher matrix
* Prove simple mirror descent convergence in convex setting; interpret p-norm mirror maps
* Trust-region derivation: constrained quadratic model and Lagrangian dual
* Present formal statements (and sketches of proofs) relating SGD noise to implicit bias (e.g., bias towards minimum norm or low-rank solutions under certain parametrizations)

**Mathematical requirements**

* Matrix calculus, eigenvalue perturbation theory, convex analysis, stochastic differential equations (basic), variational calculus.

**Hands-on practical requirements**

* Compute empirical Hessian spectra for small networks (Lanczos / power iterations)
* Implement natural gradient with Fisher-vector products (no explicit matrix inversion)
* Implement simple trust-region optimizer and mirror descent variants
* Empirically test geometry: SGD vs. Adam basin geometry and generalization

**Expert details people miss**

* Flatness definitions depend on parameterization and norm — be precise (sharpness via max eigenvalue vs local trace).
* Naively inverting Fisher is catastrophic; use conjugate-gradient / FVP approximations.
* Trust-region gains require good curvature estimates; noisy curvature explodes without damping.

**Mini-projects / proofs**

* Prove under simple quadratic model how SGD noise amplitude relates to stationary distribution (Langevin approximation).
* Implement Fisher-vector product and show effectiveness on a small MLP.

**Deliverables**

* Notebook: Hessian analysis across optimizers + natural gradient implementation + small report.

---

### Module 2 — Probabilistic Deep Learning (VAEs, Flows, EBMs, Score Matching)

**Purpose**
Master probabilistic generative modeling frameworks that connect likelihoods, variational bounds, and sampling.

**Fundamental concepts**

* Latent variable models, evidence lower bound (ELBO) and variational inference
* VAEs: encoder/decoder, amortized inference, KL annealing, posterior collapse
* Normalizing flows: change-of-variables, Jacobian determinants, coupling/spline flows
* Score matching & denoising score matching; connection to diffusion models
* Energy-Based Models (EBMs): unnormalized densities, contrastive divergence, sampling issues

**Deep theoretical coverage**

* Derive ELBO; analyze tightness and amortization gaps
* Flow expressivity proofs: universal approximation of diffeomorphisms under mild conditions
* Score matching derivation and why denoising helps (Hyvärinen, Vincent)
* EBM training instabilities: gradient estimation and partition function issues; contrastive divergence derivation

**Mathematical requirements**

* Change of variables, Jacobian determinants, measure theory intuition, variational calculus, SDE basics for score/diffusion.

**Hands-on**

* Implement VAE (convolutional) with careful posterior diagnostics (importance weighted ELBO)
* Implement RealNVP or a simple coupling flow and validate likelihoods
* Implement denoising score matching on small datasets and use annealed Langevin sampling
* Fit simple EBM (1-D / 2-D toy) with SGLD sampling and visualize modes

**Expert details people miss**

* Posterior collapse: diagnose with mutual information between latents & data; fixes (KL-free allocation, free bits, hierarchical priors).
* Flows suffer from Jacobian cost — coupling/split strategies and permutation layers matter.
* EBMs require careful sampler tuning and initialization; failure modes commonly silent.

**Mini-projects**

* Compare sample quality and held-out likelihood: VAE vs Flow on a small image dataset (e.g., CIFAR10-32 subset).
* Implement annealed Langevin dynamics and show generated samples converge from noise.

**Deliverables**

* Repo: VAE + Flow + Score model + sampling notebooks + quantitative comparisons (FID, log-likelihood estimates where applicable).

---

### Module 3 — Generative Models: GANs, Diffusion, Advanced Topics

**Purpose**
Understand adversarial and diffusion families, their theoretical properties, training pathologies, and evaluation metrics.

**Fundamental concepts**

* GAN objective families (JS, WGAN, f-GAN) and instability reasons
* Mode collapse diagnostics and mitigation (minibatch discrimination, Unrolled GANs)
* Diffusion models (DDPM): forward noising SDE, reverse process, parameterizations
* DDIM (non-Markovian deterministic sampling) and latent diffusion architectures
* Wasserstein distance and optimal transport connections to generative learning
* PixelCNN and autoregressive likelihoods (PixelCNN++)

**Deep theoretical coverage**

* Derive GAN discriminator/generator Nash equilibrium; show when it recovers data distribution in infinite-capacity limit
* DDPM derivation: ELBO decomposition and equivalence with score matching in continuous limit
* WGAN: Kantorovich duality sketch and its impact on critic capacity and gradient estimates

**Mathematical requirements**

* Probability measures, KL/JS/Wasserstein divergences, SDEs, stochastic sampling analysis, optimal transport basics.

**Hands-on**

* Implement a stable GAN training recipe: hinge loss, spectral normalization, TTUR (two-time scale updates).
* Implement DDPM training and sample generation; implement DDIM sampling for faster inference.
* Implement or use an autoregressive PixelCNN++ model for likelihood baselines.

**Expert details people miss**

* GAN evaluation: Inception/FID limitations; use precision/recall for generative models.
* Diffusion sample quality depends on training noise schedule and network parameterization; latents reduce compute dramatically (latent diffusion).
* WGAN requires careful critic regularization; weight clipping is crude — use gradient penalty or spectral normalization.

**Mini-projects**

* Train DDPM on small image dataset and compare FID/Precision vs a GAN baseline.
* Implement latent diffusion on a lower-resolution dataset (learn an autoencoder first).

**Deliverables**

* Comparative paper-style notebook: GAN vs Diffusion vs PixelCNN on same data with metrics and ablations.

---

### Module 4 — Representation Learning & Contrastive Methods

**Purpose**
Acquire theoretical and experimental mastery of self-supervised and contrastive methods and how they produce transferable embeddings.

**Fundamental concepts**

* Contrastive loss families (InfoNCE), MoCo, SimCLR, BYOL, DINO
* Pretext tasks and architecture choices
* Negative sampling, temperature scaling, batch sizes effects
* Multi-modal contrastive learning (CLIP style)

**Deep theoretical coverage**

* Derive InfoNCE and show relation to bound on mutual information; analyze temperature’s role.
* Formalize collapse modes for BYOL-style methods and why implicit asymmetry helps.
* Theoretical connections between contrastive learning and spectral clustering / manifold learning.

**Mathematical requirements**

* Mutual information bounds, concentration of measure, matrix spectral properties for similarity matrices.

**Hands-on**

* Implement SimCLR pipeline with careful data augmentation and evaluate transfer performance.
* Implement MoCo with a queue and momentum encoder.
* Run multi-modal contrastive training on a text–image toy dataset (small subset of COCO).

**Expert details people miss**

* InfoNCE lower bounds MI only under certain assumptions; empirical success doesn’t equal MI maximization.
* Negative sample quality matters far more than raw quantity for some domains; hard negatives and mining strategies help.
* Fine-tuning strategy strongly influences transfer results — always test linear probe & full fine-tune.

**Mini-projects**

* Ablation: temperature, batch size, augmentations on SimCLR linear probe performance.
* Implement multi-modal CLIP-style model on small dataset and analyze attention maps.

**Deliverables**

* Transfer benchmark suite + reproducible code + analysis report.

---

### Module 5 — Causality for ML

**Purpose**
Give you the tools to reason about interventions, robustness, and identifiability; necessary for robust ML and principled domain transfer.

**Fundamental concepts**

* Structural causal models (SCMs), DAGs, do-calculus
* Confounding, backdoor/frontdoor adjustments
* Identification conditions and instrumental variables
* Causal discovery basics (PC algorithm, score-based)
* Causal effect estimation: propensity scores, inverse probability weighting, causal forests

**Deep theoretical coverage**

* Formal do-calculus rules and proof sketches for identifiability theorems
* Derive unbiasedness of IPW and double-robust estimators
* Analyze limits: unidentifiable models and sensitivity analysis

**Mathematical requirements**

* Probability theory, conditional independence, counterfactual notation, nonparametric identification.

**Hands-on**

* Use simulated SCMs to show confounding and identification failures
* Implement propensity score matching and double robust estimator on semi-synthetic datasets
* Apply causal discovery to small datasets and inspect false positives/negatives

**Expert details people miss**

* Causal effect estimators rely on untestable assumptions (no unmeasured confounders); always report sensitivity.
* Interventions vs conditional distributions are different: do(P(x|do(z))) vs P(x|z).

**Mini-projects**

* Build a simple SCM with confounder and instrument; demonstrate when IPW fails and how instrumental variables recover effect.

**Deliverables**

* Notebook: causal identification experiments + guidelines for ML practitioners.

---

### Module 6 — Reinforcement Learning: Advanced Topics

**Purpose**
Move beyond policy gradients to model-based, offline, and imitation learning with research quality experiments.

**Fundamental concepts**

* Policy gradients, actor-critic, advantage estimation (A2C/A3C, PPO basics)
* Trust-region methods (TRPO) and proximal objectives
* Off-policy value estimation, importance sampling corrections
* Model-based RL: dynamics models, planning (MPC), Dyna approaches
* Offline RL: distributional shift, conservative objectives (CQL), batch RL pitfalls
* Imitation learning: behavioral cloning, inverse RL, GAIL
* POMDPs: belief states, filtering, memory architectures

**Deep theoretical coverage**

* Variance properties of policy gradient estimators and variance reduction techniques
* Derive TRPO surrogate objective and KL constraint formulation
* Theoretical pitfalls of offline RL: extrapolation error and pessimism principle

**Mathematical requirements**

* Markov decision processes, Bellman equations, stochastic control, concentration bounds for off-policy evaluation.

**Hands-on**

* Implement PPO from scratch on OpenAI Gym (CartPole → Continuous control)
* Implement a simple model-based planner (learn dynamics on state, predict rollout) and compare sample efficiency with model-free baseline
* Implement CQL or another offline RL baseline and evaluate on D4RL datasets (small scale)

**Expert details people miss**

* Hyperparameters (entropy, clipping, learning rate ratios) drastically change policy gradient behavior — must run wide sweeps.
* Offline RL evaluation requires careful OPE (importance sampling, FQE) and confidence bounds.

**Mini-projects**

* Compare model-based vs model-free methods on a low-dim control task with identical compute budget.
* Implement GAIL imitation on an expert dataset and analyze mode coverage.

**Deliverables**

* RL repo with PPO, a model-based baseline, and at least one offline RL experiment.

---

### Module 7 — Integration & Cross-Module Experiments

**Purpose**
Synthesize modules: use representation learning inside generative models or RL; test causal robustness on representation transfer; explore optimization tricks across families.

**Suggested experiments**

* Use contrastive pretraining as encoder for diffusion model; test sample diversity vs baseline.
* Test effect of optimizer (natural gradient approx) on VAE training stability.
* Evaluate causal intervention (do) robustness of representations: perturb system and measure downstream transfer.

**Deliverables**

* One integrated research notebook per experiment with code and reproducible metrics.

---

## WEEKLY PLAN (16-week template; compress to 12 by merging weeks)

This plan assumes ~20–30 hours/week. Adjust tempo if you are in semester.

**Week 1 — Optimization fundamentals**

* Read NTK/SGD noise basics; implement Hessian spectral tools.
* Mini: compute Hessian spectrum for small CNN.

**Week 2 — Advanced optimizers**

* Natural gradient derivation; implement FVP; mirror descent exercise.

**Week 3 — Probabilistic foundations**

* ELBO derivation, VAE implementation (basic).

**Week 4 — Flows & score matching**

* Implement RealNVP or simple coupling flow; derive Jacobian computation.
* Implement denoising score matching toy.

**Week 5 — EBMs & sampling**

* Implement toy EBM with SGLD; analyze sampler mixing.

**Week 6 — GANs theory + practice**

* Implement stable GAN recipe; test on small dataset; study failure modes.

**Week 7 — Diffusion models (DDPM)**

* Implement DDPM training and sampling; compute likelihood proxy metrics.

**Week 8 — DDIM & Latent Diffusion**

* Implement DDIM sampler; build simple latent autoencoder pipeline.

**Week 9 — Representation learning (contrastive)**

* Implement SimCLR & MoCo variants; run linear probe experiments.

**Week 10 — Multi-modal contrastive & InfoNCE theory**

* Implement small CLIP-style model on tiny image/text pair dataset.

**Week 11 — Causality basics & estimation**

* Do-calculus exercises; implement IPW and double robust estimators on semi-synthetic data.

**Week 12 — Advanced RL: policy gradients & TRPO/PPO**

* Implement PPO; run on continuous control toy.

**Week 13 — Model-based & offline RL**

* Implement learned dynamics planner; run CQL offline baseline on small dataset.

**Week 14 — Integration experiments**

* Run one cross-module experiment (e.g., use contrastive encoder in VAE/diffusion).

**Week 15 — Ablations & rigorous evaluations**

* Run ablation sets, metric suites (FID, IS, precision/recall, likelihoods, downstream linear probe).

**Week 16 — Capstone consolidation & writeup**

* Finalize capstone report, reproducible code, plots, and analysis.

---

## MASTERY CHECKS

To pass Phase 3 you must demonstrate each of the following:

**Theory / Proofs**

* Provide a written proof sketch showing why natural gradient arises from a KL constraint (Fisher metric).
* Derive the ELBO and show how importance weighting tightens the bound; show when IWAE improves latent utilization.
* Explain the DDPM objective derivation and its connection to score matching (mathematical steps).
* Prove the representer theorem for kernel regression (used as background for contrastive spectral views).
* Prove simple mirror descent convergence under convexity assumptions.

**Coding / Reproduction**

* Implement and train: VAE, one Flow, one Diffusion model, and one GAN on the same dataset subset; produce comparison metrics (qualitative + quantitative).
* Implement natural gradient approx via FVP and verify on small network — show faster or more stable convergence in at least one setting.
* Implement SimCLR and report linear probe accuracy vs supervised baseline.
* Implement PPO and a model-based planner and compare sample efficiency on a control task.

**Experimental / Research**

* Run at least three ablations that test a clear hypothesis (e.g., noise schedule effect in DDPM, temperature effect in InfoNCE, optimizer choice effect on VAE posterior collapse).
* Produce a reproducibility artifact: a README with exact seeds, hyperparams, and scripts that reproduce one major experiment end-to-end.

**Communication**

* 8–12 page technical report (or 6–8 pages + appendices) detailing capstone experiments, proofs, ablations, and code references.

---

## CAPSTONE PROJECT (publication-quality)

**Title (example):** *“A Comparative Study and Unified Analysis of Likelihood, Adversarial, and Score-based Generative Models with Representation-based Conditioning”*

**Scope**

* Train and compare four families on the same dataset (e.g., 64×64 subset of FFHQ or CelebA): VAE (hierarchical), Normalizing Flow (coupling), GAN (hinge + SN), Diffusion (DDPM + DDIM).
* Use a shared encoder (contrastive pretraining) to produce conditional generative results — test whether representation conditioning improves sample fidelity and sample efficiency across families.
* Run rigorous evaluations: FID, IS, precision/recall, likelihood estimates (where applicable), and downstream linear probe performance of learned representations.
* Add at least one novel technical contribution (e.g., natural-gradient-based optimizer for diffusion training, or a hybrid objective combining ELBO + score loss that improves sample quality).
* Provide theoretical analysis for the proposed contribution (derivation + intuitive justification) and ablation studies demonstrating gains.

**Deliverables**

* Public GitHub repo with scripts to reproduce training & evaluation.
* 8–12 page manuscript styled for a workshop submission (methods, experiments, ablations, limitations).
* Reproducible Docker/conda environment + small checkpoint files.

**Why this capstone?**
It forces mastery across probabilistic modeling, diffusion theory, GAN stability, representation pretraining, optimization tricks, and rigorous evaluation — everything Phase 3 aims to teach.

---

## WHAT NOT TO SKIP (strict)

* Do **not** treat ELBO or likelihood claims as black boxes — compute and inspect importance weights and IWAE behavior.
* Do **not** compare sample images visually only — compute multiple metrics and report confidence intervals.
* Do **not** skip sampler diagnostics (mixing, mode coverage) for EBMs/diffusion/GANs.
* Do **not** use single random seeds for claims; always run 3+ seeds for major experiments.
* Do **not** ignore theoretical assumptions when making empirical claims (e.g., IID assumption, capacity limits).

---

## PREREQUISITES (explicit)

You must have completed and be **fluent** in the following prior to Phase 3:

* Phase 0 (Foundations): Linear algebra (SVD, pseudoinverse), multivariable calculus, probability basics, numerical stability, matrix calculus.
* Phase 1 (Core ML): GLMs, SVMs, trees/ensemble intuition, cross-validation & experimental hygiene.
* Phase 2 (Deep Learning Core): Backprop/autodiff, CNN/RNN/Transformer internals, optimizer basics (SGD/Adam), experience building full training pipelines, ResNet/LSTM/Transformer implementations and ablations.

If any prerequisite is weak, allocate time to remediate — these are assumed knowledge.

---

## FAILURE MODES & FIXES

**Failure:** Claiming improvement from noisy single-seed runs.
**Fix:** Run 3+ seeds, report mean ± std, and perform statistical tests where appropriate.

**Failure:** Interpreting low FID as sole evidence of generality.
**Fix:** Use multiple metrics (precision/recall, IS, downstream tasks), and show qualitative diversity.

**Failure:** Deploying samplers with inadequate burn-in or step sizes (EBMs/diffusion).
**Fix:** Tune sampler hyperparameters, show chain diagnostics (autocorrelation, ESS).

**Failure:** Optimizer misuse: comparing Adam with SGD without hyperparameter tuning.
**Fix:** Tune optimizers thoroughly (LR sweeps, weight decay), and show learning curves, not just final metrics.

**Failure:** Ignoring identifiability in causal claims.
**Fix:** Provide identification argument (backdoor/frontdoor/instrument) or sensitivity analysis.

---

## RESOURCES (depth → textbooks, canonical papers, and advanced notes)

### Optimization

* **Books / Notes:** *Convex Optimization* — Boyd & Vandenberghe (background); *Optimization for Machine Learning* (edited volume)
* **Papers:**

  * S. Amari — *Natural Gradient Works Efficiently in Learning* (1998)
  * Hardt et al. — *Train faster, generalize better: Stability of SGD* (2016)
  * Mandt, Hoffman & Blei — *Stochastic Gradient Descent as Approximate Bayesian Inference* (2017)

### Probabilistic Deep Learning

* **Books / Chapters:** *Pattern Recognition and Machine Learning* — Bishop (EM, latent models); *Information Theory, Inference, and Learning Algorithms* — MacKay
* **Papers:**

  * Kingma & Welling — *Auto-Encoding Variational Bayes* (2014)
  * Rezende & Mohamed — *Variational Inference with Normalizing Flows* (2015)
  * Hyvärinen — *Score Matching* papers; Vincent — *Denoising Score Matching*

### EBMs & Score / Diffusion

* **Papers:**

  * Song & Ermon — *Generative Modeling by Estimating Gradients of the Data Distribution*
  * Ho et al. — *Denoising Diffusion Probabilistic Models* (DDPM)
  * Song et al. — *Score-based Generative Modeling through SDEs*
  * Dhariwal & Nichol — *Diffusion Models Beat GANs on Image Synthesis*

### GANs & Autoregressive

* **Papers:**

  * Goodfellow et al. — *GANs* (2014)
  * Arjovsky et al. — *Wasserstein GAN*
  * Salimans et al. — *Improved Techniques for Training GANs*
  * van den Oord et al. — *PixelCNN/PixelRNN*; Salimans et al. — *PixelCNN++*

### Representation Learning / Contrastive

* **Papers:**

  * Oord, Li & Vinyals — *Representation Learning with Contrastive Predictive Coding*
  * Chen et al. — *SimCLR*
  * He et al. — *MoCo*
  * Grill et al. — *BYOL*; Caron et al. — *DINO*
  * Radford et al. — *CLIP* (multi-modal)

### Causality

* **Books / Papers:**

  * *Causality* — Judea Pearl (chapters on do-calculus)
  * Peters, Janzing & Schölkopf — *Elements of Causal Inference*
  * Hernán & Robins — *Causal Inference* (select chapters)

### Reinforcement Learning

* **Books / Papers:**

  * *Reinforcement Learning: An Introduction* — Sutton & Barto (foundations)
  * Schulman et al. — *TRPO/PPO*
  * Fujimoto et al. — *Off-Policy Deep RL: CQL* (offline RL)
  * Pathak et al. — *Model-based RL papers*

---

If you want, I will now:

* generate the **exact experiment list, datasets, hyperparameters, and shell scripts** for the capstone, or
* produce a **16-week calendar with daily actionable tasks and checkpoints** for this phase.

Which do you want me to produce next?

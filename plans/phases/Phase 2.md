---
title: Phase 2
updated: 2025-11-19 18:22:03Z
created: 2025-11-19 18:21:29Z
latitude: 23.21563540
longitude: 72.63694150
altitude: 0.0000
---

# **PHASE 2 — DEEP LEARNING CORE (RESEARCH-GRADE, EXHAUSTIVE PLAN)**

**Duration:** 12–16 weeks
**Outcome:** You should emerge with the ability to *implement*, *analyze*, and *extend* modern deep networks—CNNs, RNNs, and Transformers—at a level suitable for state-of-the-art research. This phase transforms you from someone who “uses deep learning frameworks” to someone who **understands the architecture, math, and optimization dynamics inside them**.

---

# **PHASE OVERVIEW**

Deep learning research requires three rare capabilities:

1. **Architectural mastery** — internalize the mathematical and engineering mechanisms behind MLPs, CNNs, RNNs, attention, and early Transformers.
2. **Optimization intuition** — understand geometry of loss landscapes, initialization, normalization and training dynamics.
3. **Reproducible engineering** — build training pipelines from scratch with stable training, logging, and ablation interfaces.

This phase develops all three.

After completing Phase 2, you should:

* write a full training loop (forward, backward, optimizer, LR schedules) from scratch using PyTorch
* implement ResNet, LSTM, GRU, attention, and a Transformer encoder
* derive gradients manually for core operations
* understand NTK and UAT at a research level
* reproduce CIFAR-10 ResNet results
* perform controlled ablations like a serious researcher

This prepares you for Phase 3 (Advanced Methods & Theory), Phase 3.5 (LLMs & RAG), and Phase 4 (Research Skills).

---

# **MODULES (6 total)**

---

## **Module A — Foundations of Neural Networks**

### **Purpose**

Master the fundamentals: forward/backward propagation, architectures, initialization theory, normalization, and regularization.

### **Fundamental Concepts**

* Forward & backward pass
* Autodiff (computational graph mechanics)
* Initialization: Xavier, Kaiming, LSUV
* BatchNorm, LayerNorm — what they do and *why*
* Dropout: regularization interpretation + **Dropout as Bayesian approximation**
* Universal Approximation Theorem (UAT) and its consequences/limits

### **Deep Theoretical Coverage**

* Derive full backprop manually for a 2-layer MLP using matrix calculus
* Compute gradients for ReLU, GELU, LayerNorm
* Show how initialization affects variance propagation
* Prove UAT for continuous functions on compact subsets via Cybenko’s theorem (at least a sketch)
* Bayesian view of dropout: approximate variational inference

### **Mathematical Requirements**

* Chain rule in matrix form
* Jacobian/Hessian intuition
* Variance propagation in deep nets

### **Hands-on Practical Requirements**

* Implement MLP with and without autodiff
* Compare training dynamics under different initialization strategies
* Visualize activation distributions and gradient magnitudes through layers

### **Expert-level Details People Miss**

* Why BatchNorm introduces stochasticity (batch-level noise)
* How dropout affects activation scaling at inference
* Subtle but important differences between ReLU and GELU

### **Mini-Projects**

* Build MLP from scratch using only NumPy
* Derive gradient formulas for LayerNorm
* Explore unstable initialization via exploding gradients

### **Deliverables**

* “NN Core Math Notebook” summarizing derivations & experiments
* Clean PyTorch MLP implementation with manual backprop check

---

## **Module B — Convolutional & Recurrent Models**

### **Purpose**

Master convolution, residual networks, recurrent nets, gating mechanisms, and their theoretical motivations.

---

### **Part 1 — CNNs**

**Fundamentals**

* Convolutions, padding, stride
* LeNet → VGG → ResNet evolution
* Residual connections
* Feature hierarchy
* Global average pooling

**Deep Theory**

* Why residuals cure vanishing gradients
* Expressivity vs. parameter sharing
* Initialization for deep CNNs
* Effective receptive field computation

**Hands-on**

* Implement LeNet manually
* Reimplement ResNet-18 from scratch in PyTorch
* Train ResNet on CIFAR-10

**Mini-Projects**

* Show gradient flow difference between plain CNN and ResNet
* Compute effective receptive field

---

### **Part 2 — RNNs**

**Fundamentals**

* RNNs, LSTM, GRU
* Gating mechanisms
* **LSTM peephole variants**
* **Gated CNNs (WaveNet)**

**Deep Theory**

* BPTT gradient dynamics (vanishing/exploding)
* LSTM gating math
* GRU vs. LSTM equivalence conditions
* WaveNet causal convolution math

**Hands-on**

* Implement vanilla RNN, LSTM, and GRU manually
* Train on small sequential dataset (e.g., char-level model)
* Implement a small WaveNet

**Mini-Projects**

* Visualize gradient norms across time steps
* Compare expressive power of LSTM vs GRU

### **Deliverables**

* CNN & RNN implementations + CIFAR-10 logs
* Experiments demonstrating gradient behavior in RNNs

---

## **Module C — Activation Functions & Their Mathematics**

### **Purpose**

Understand the role, math, and optimization implications of nonlinearities.

### **Fundamental Concepts**

* ReLU, LeakyReLU, GELU, Sigmoid, Tanh
* Saturation, gradient flow, nonlinearity families

### **Deep Theoretical Coverage**

* **GELU derivation** from Gaussian noise reinterpretation
* Why sigmoid/tanh saturate and kill gradients
* Smoothness and curvature’s effect on optimization
* Comparison of Swish & GELU

### **Hands-on**

* Plot activation functions, derivatives
* Train identical networks with different activations; compare convergence

### **Mini-Projects**

* Implement custom PyTorch activation
* Derive derivative of GELU (exact vs approximation)

### **Deliverables**

* Activation comparison notebook with training curves

---

## **Module D — Optimization for Deep Networks**

### **Purpose**

Build research-level intuition for how deep nets train, why optimizers behave differently, and how schedules shape convergence.

### **Fundamental Concepts**

* SGD, momentum, Adam, RMSProp
* Warm restarts
* Cosine decay, step LR, one-cycle LR
* Gradient clipping
* Weight decay vs. L2 regularization
* **Neural Tangent Kernel (NTK)** and wide-network behavior

### **Deep Theoretical Coverage**

* Derive SGD + momentum updates
* Analyze Adam’s bias correction
* Learning rate schedules and optimization trajectories
* NTK: wide neural networks converge like kernel machines

### **Hands-on**

* Write custom optimizers
* Compare trajectories of SGD/Adam on same model
* Compute empirical NTK for a small network

### **Mini-Projects**

* Visualize loss landscape slices
* Numerically approximate NTK for a small MLP

### **Deliverables**

* “Deep Optimization Experiments Report” documenting behavior

---

## **Module E — Transformer Internals**

### **Purpose**

Build the intellectual foundation for modern sequence modeling and LLM architectures.

### **Fundamental Concepts**

* Scaled dot-product attention
* Multi-head attention
* Residual + LayerNorm
* Feed-forward sublayer
* **Positional encodings** (sinusoidal & learned)
* **Positional encoding mathematical theory** — Fourier interpretation

### **Deep Theoretical Coverage**

* Full derivation of QKV projections
* Attention matrix and softmax stability
* Derivation of sinusoidal encoding from continuous Fourier basis
* Why residual streams stabilize very deep Transformer training

### **Hands-on**

* Implement attention from scratch (no PyTorch attention API)
* Implement MHA manually
* Implement positional encodings

### **Mini-Projects**

* Show difference between absolute vs relative positional encodings
* Visualize attention heads for synthetic tasks

### **Deliverables**

* Full Transformer encoder block implemented manually

---

## **Module F — Building Full Training Pipelines**

### **Purpose**

Engineering discipline: you must be able to train deep models reproducibly like a professional researcher.

### **Components**

* Training loop with manual forward/backward
* Gradient accumulation
* Mixed precision training
* Logging (TensorBoard / W&B)
* Checkpointing
* Data augmentation pipelines for CIFAR-10
* Deterministic seeding

### **Hands-on**

* Build complete PyTorch training loop
* Add metrics: accuracy, loss smoothing, LR tracking
* Train ResNet-18 on CIFAR-10 with competitive accuracy

### **Deliverables**

* Full CIFAR-10 training pipeline repository

---

# **WEEKLY PLAN (12–16 weeks)**

---

### **Week 1 — NN Core Foundations**

* Backprop derivations
* Implement MLP with manual gradients
* Explore initialization strategies

### **Week 2 — Normalization & Regularization**

* BatchNorm, LayerNorm derivations
* Dropout experiments + Bayesian interpretation

### **Week 3 — CNN Basics**

* Implement LeNet
* Study effective receptive field
* Start ResNet architecture study

### **Week 4 — ResNet Implementation**

* Implement ResNet-18
* Begin CIFAR-10 training
* Compare with/without residuals

### **Week 5 — RNN Theory**

* Derive BPTT
* Implement vanilla RNN
* Visualize gradient norms

### **Week 6 — LSTMs, GRUs, Peepholes**

* Implement LSTM, GRU, peephole LSTM
* Train char-level model

### **Week 7 — WaveNet & Sequence CNNs**

* Implement dilated convolutions
* Build mini-WaveNet model

### **Week 8 — Activations**

* Study derivatives and curvature
* Run activation comparison experiments

### **Week 9 — Optimization**

* Implement SGD, Adam
* LR schedules
* NTK experiments

### **Week 10 — Transformer Internals**

* Implement attention
* Implement MHA
* Implement positional encoding

### **Week 11 — Transformer Encoder**

* Build full encoder block
* Test on toy tasks

### **Week 12 — Training Pipeline Engineering**

* Build reproducible CIFAR-10 training loop
* Add logging, checkpointing, augmentation

### **Weeks 13–16 — Refinement & Capstone**

* Reproduce ResNet CIFAR-10 results
* Conduct ablations
* Prepare final research-quality report

---

# **MASTERY CHECKS**

You must satisfy all of the following:

### **Conceptual**

* Explain backprop with Jacobian formalism
* Prove properties of BatchNorm and LayerNorm
* Derive LSTM gradient for a single time-step
* Explain residual stability in deep networks
* Derive sinusoidal encoding from Fourier basis

### **Coding**

* Manual backprop MLP
* Full LSTM and GRU implementations
* Transformer attention implementation without PyTorch built-ins
* Working CIFAR-10 pipeline with competitive accuracy

### **Research Ability**

* Perform ablations on activation functions, initialization, LR schedules
* Analyze gradient norms and training stability
* Write a 4–8 page report comparing training strategies

### **Reproduction**

* Reproduce ResNet-18 CIFAR-10 accuracy within 1–3% of reference numbers

---

# **CAPSTONE PROJECT — “End-to-End Deep Architecture Reproduction”**

**Goal:** Reproduce the training setup, experiments, and findings of a foundational deep learning paper.
Recommended papers:

1. **He et al. (2016)** — *Deep Residual Learning for Image Recognition*
2. **Hochreiter & Schmidhuber (1997)** — *LSTM original paper*
3. **Vaswani et al. (2017)** — *Attention is All You Need* (restricted to encoder)

**Scope**

* Re-implement architecture from scratch
* Reproduce main experimental results
* Add at least two ablations (e.g., norm types, activation, LR schedule)
* Produce a reproducibility report with figures, logs, and insights

**Deliverables**

* GitHub repo (clean, reproducible)
* 6–10 page PDF report styled like a workshop submission

---

# **PREREQUISITES FROM PHASE 1**

* Complete fluency with classical ML
* Solid understanding of gradient-based optimization
* Comfort writing clean, vectorized code
* Experience with nested CV and experiment hygiene
* Probability & linear algebra from Phase 0

If any prerequisite is weak, stop and reinforce before continuing.

---

# **FAILURE MODES & FIXES**

### **Failure 1: Relying on PyTorch for core operations**

Fix: Implement layers manually before using higher APIs.

### **Failure 2: Memorizing Transformer architecture without understanding attention math**

Fix: Derive softmax(QKᵀ/√d) from scratch.

### **Failure 3: Not logging metrics properly**

Fix: Track LR, gradients, weight norms, loss curves.

### **Failure 4: Not understanding training instability**

Fix: Visualize gradient flow, activation distributions, and LR schedules.

### **Failure 5: Comparing only final accuracy**

Fix: Always run ablations and learning curves.

---

# **RESOURCES (DEPTH ONLY)**

### **Books / References**

* *Deep Learning* — Goodfellow, Bengio, Courville
* *Dive into Deep Learning* — Zhang et al.
* *Neural Networks and Deep Learning* — Nielsen (for classical derivations)

### **Papers**

* He et al. (ResNet)
* Hochreiter & Schmidhuber (LSTM)
* Vaswani et al. (Transformer)
* Ba et al. — Layer Normalization
* Srivastava et al. — Dropout
* Ioffe & Szegedy — BatchNorm
* Jacot et al. — NTK paper

### **Tooling**

* PyTorch docs
* TensorBoard / W&B
* NVIDIA AMP documentation



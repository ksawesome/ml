---
title: Master Roadmap
updated: 2025-11-23 19:07:52Z
created: 2025-11-19 18:18:50Z
latitude: 23.21563540
longitude: 72.63694150
altitude: 0.0000
---

# 🚀 \*\*MASTER ROADMAP \*\*

# **PHASE 0 — FOUNDATIONS (WITH ALL MISSED TOPICS ADDED)**

**Duration:** 4–8 weeks  
**Goal:** Zero mathematical or coding blind spots.

### **Core Topics**

- Linear algebra

  - Vectors, matrices, matrix calculus
  - Eigen/SVD
  - PCA
  - **Pseudoinverses**
  - **Spectral graph theory**
  - **Functional analysis basics**
- Calculus

  - Multivariable calculus
  - Taylor expansions
  - **Measure theory basics**
- Probability & Statistics

  - Random variables, distributions, expectation
  - MLE, MAP
  - Hypothesis testing
  - KL divergence
  - Concentration inequalities
  - **Markov chains**
  - **Stochastic processes**
  - **Poisson processes**
  - **Bayesian networks (detailed)**
  - **Information theory (entropy, MI, Fano inequality)**
- Optimization

  - Gradient descent, Adam, RMSProp
  - Convexity
  - **Numerical stability & conditioning**
- Programming & Tooling

  - Python (advanced)
  - NumPy internals
  - Pandas
  - Git, Bash
  - Virtualenv, Anaconda
  - **C++ basics**
  - **CUDA basics**
  - **Compilers (MLIR/XLA/TVM overview)**
  - **Parallel programming concepts**

### **Projects**

- Logistic regression from scratch
- Implement matrix operations
- Gradient derivation hand-written notebook

* * *

# **PHASE 1 — CORE MACHINE LEARNING (WITH MISSED TOPICS ADDED)**

**Duration:** 8–12 weeks

### **Supervised Learning**

- Linear models

- Logistic regression

- SVMs

  - **Dual formulation**
  - **Kernel trick proofs**
- Decision trees

  - Gini, entropy, pruning
  - **Full impurity derivation**
- Ensemble methods

  - Bagging, boosting
  - XGBoost, Random Forest
  - **CatBoost**
  - **LightGBM leaf-wise growth theory**

### **Unsupervised**

- K-means
- GMM
- EM algorithm
- PCA, ICA
- Spectral clustering
- **Manifold learning (Laplacian eigenmaps, Isomap)**
- **Metric learning (pre-deep era)**

### **Anomaly Detection**

- **One-Class SVM**
- **Isolation Forest**

### **Time-Series ML**

- **ARIMA**
- **SARIMAX**
- **Holt-Winters**

### **Other Skills**

- Feature engineering
- Cross-validation
- Bias-variance
- Calibration

### **Projects**

- Kaggle tabular pipeline
- Implement a mini-boosting algorithm
- Reproduce a classical ML paper

* * *

# **PHASE 2 — DEEP LEARNING CORE (WITH MISSED TOPICS ADDED)**

**Duration:** 12–16 weeks

### **Neural Networks**

- Backprop, autodiff

- Initialization methods

- BatchNorm, LayerNorm

- Dropout

  - **Dropout as Bayesian approximation**
- Transformers basics

### **CNNs, RNNs**

- CNN (LeNet, ResNet)

- RNNs

  - LSTM, GRU
  - **LSTM peephole variants**
  - **Gated CNNs (WaveNet)**
- Attention mechanisms

- **Universal Approximation Theorem**

### **Activation Functions**

- ReLU, GELU
- Sigmoid, Tanh
- **GELU mathematical derivation**

### **Optimization**

- SGD, Adam
- Warm restarts
- LR schedules
- **Neural Tangent Kernel (NTK)**

### **Transformer Internals**

- Multi-head attention

- QKV projections

- Positional encodings

  - **Positional encoding mathematical theory**

### **Projects**

- Implement Transformer encoder
- Build training loop with logging
- Reproduce ResNet on CIFAR-10

* * *

# **PHASE 3 — ADVANCED METHODS & THEORY (WITH ALL MISSED TOPICS ADDED)**

**Duration:** 12–20 weeks

### **Optimization Theory**

- Sharp vs flat minima
- Implicit regularization
- **Natural gradient**
- **Trust-region methods**
- **Mirror descent**
- **Lipschitz continuity proofs**

### **Probabilistic Deep Learning**

- VAEs
- Normalizing flows
- **Score matching**
- **Energy-based models (deep theory)**

### **Generative Models**

- GANs

- Diffusion models

  - DDPM
  - **DDIM**
  - **Latent diffusion theory**
- **PixelCNN++**

- **Wasserstein distances deep theory**

- **NeRFs (Neural Radiance Fields)**

### **Representation Learning**

- Contrastive learning
- Self-supervised
- **InfoNCE theoretical derivation**
- **Clustered contrastive learning**
- **Multi-modal contrastive learning**

### **Causality**

- Basic do-calculus
- Daggers
- Confounding
- Identification

### **Reinforcement Learning**

- Policy gradients
- Actor-critic
- **POMDPs**
- **Model-based RL**
- **Offline RL**
- **Imitation learning**

* * *

# ⭐ **PHASE 3.5 — LLMs, TRANSFORMERS & RAG (NEW PHASE)**

**Duration:** 4–12 weeks

### **LLM Fundamentals**

- Full Transformer stack
- Causal attention
- KV caching
- Sampling methods
- Tokenization
- **Scaling laws**
- **Alignment, RLHF, DPO**

### **Embedding Models**

- BERT embeddings
- Sentence-transformers
- LLM embedding heads
- **Vector normalization, whitening, quantization**

### **Retrieval Systems**

- Exact search

- Approximate search

  - FAISS
  - HNSW
  - ScaNN
  - Annoy

### **RAG Architecture**

- Chunking
- Embedding
- Indexing
- Retrieval
- Re-ranking
- Prompt construction
- Query rewriting (HyDE, Rewriter models)
- **Graph RAG**
- **Self-RAG**
- **Agentic RAG**
- **Long-context vs. RAG tradeoffs**

### **Evaluation**

- Precision@k, recall@k
- Faithfulness & hallucination
- Groundedness metrics
- Human evaluation

### **Engineering**

- Vector DBs (Milvus, Pinecone, Weaviate)
- Latency optimization
- Batching, caching
- Logging + tracing
- Monitoring failure cases

### **Advanced**

- **Memory-augmented LLMs**
- **Tool-using LLM agents**
- **Hybrid RAG + finetuning models**

* * *

# **PHASE 4 — RESEARCH SKILLS (WITH MISSED TOPICS ADDED)**

**Duration:** Continuous

### **Core Research Skills**

- Paper reading
- Distillation of contributions
- Reproducible experiments
- Baseline selection
- Ablation design

### **Advanced Methods**

- **Bayesian hyperparameter optimization**

- **ASHA / PBT / NAS basics**

- **Interpretability**

  - SHAP
  - LIME
  - Integrated Gradients
  - DeepLIFT
- **Experimental design (DOE)**

- **Statistical reproducibility**

  - Bootstrap
  - Jackknife
  - Permutation tests
- **Research ethics**

- **Deep academic writing**

- Versioning (Git, DVC)

- MLFlow / W&B

* * *

# **PHASE 5 — SPECIALIZATION (WITH ALL MISSED TOPICS MERGED)**

Choose 1–2 domains. Examples:

### **NLP**

- BERT, GPT, T5 internals
- **DeBERTa, UL2, T5 theory**
- **Retrieval-augmented models in depth**
- **Scaling laws**
- Attention variants
- Sequence training

### **Computer Vision**

- Detection, segmentation
- **Diffusion for vision**
- **Video modeling**
- **3D vision**
- **Point cloud networks**

### **ML Systems**

- Distributed training
- **Compilers (TVM, Triton, XLA)**
- **Tensor cores and accelerator hardware**
- **Quantization (INT8, INT4, INT2)**
- **Knowledge distillation theory**

### **RL/Robotics**

- **Sim2Real**
- **Hierarchical RL**
- **System identification**
- **Reward shaping theory**

* * *

# **PHASE 6 — PUBLICATION & COMMUNITY (WITH MISSED TOPICS)**

### **Paper Writing**

- Structure of academic papers

- Argument flow

- Latex mastery

  - **Advanced templates**
  - **BibTeX automation**

### **Conference Process**

- Workshop submissions
- Reviewer guidelines
- **Rebuttal negotiation strategies**
- Poster design
- Talk delivery

* * *

# **PHASE 7 — SYSTEM DESIGN & DEPLOYMENT (WITH MISSED TOPICS)**

### **Serving & Infrastructure**

- Model serving frameworks

  - **vLLM**
  - **TensorRT-LLM**
- Model optimization

  - Quantization
  - Pruning
  - Knowledge distillation
- Privacy-preserving ML

  - **DP-SGD**
  - **Federated learning**
  - **Secure aggregation**

### **MLOps**

- CI/CD pipelines
- Dataset versioning
- Monitoring
- Logging & metrics
- A/B testing

### **Safety**

- Red teaming
- Adversarial attacks
- Hallucination control

* * *

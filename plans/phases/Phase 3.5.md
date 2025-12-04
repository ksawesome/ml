---
title: Phase 3.5
updated: 2025-11-19 18:26:13Z
created: 2025-11-19 18:22:29Z
latitude: 23.21563540
longitude: 72.63694150
altitude: 0.0000
---

# PHASE 3.5 — LLMs, TRANSFORMERS & RAG (RESEARCH-GRADE, EXHAUSTIVE PLAN)

**Duration:** 4–12 weeks (recommended 8 weeks for full depth)
**Goal:** move from Transformer literacy → production-ready, research-quality Retrieval-Augmented Generation (RAG) systems and LLM engineering. You will be able to design novel retrieval+generation architectures, evaluate faithfulness/hallucinations, optimize latency/throughput for production, and propose/mechanize research contributions (e.g., hybrid RAG+finetune, agentic retrieval, memory-augmented LLMs).

---

## PHASE OVERVIEW

Phase 3.5 sits at the intersection of model architectures, information retrieval, and systems engineering. It requires:

* deep understanding of Transformer internals and LLM inference,
* theory + practice of embedding spaces and vector search,
* retrieval pipeline design (chunking, indexing, reranking),
* careful empirical evaluation of grounding/faithfulness, and
* applied engineering (vector DBs, batching, caching, monitoring).

Completing this phase unlocks the ability to build RAG systems that are research-grade (novel evaluations, hybrid techniques) and production-grade (low-latency, robust, auditable).

---

## MODULES (7 total)

### Module 1 — Transformer & LLM Fundamentals (Architecture → Inference)

**Purpose:** absolute mastery of the Transformer as used in modern LLMs and practical inference optimizations.

**Fundamental concepts**

* Transformer encoder/decoder and causal-only variants
* QKV math, scaled dot-product attention, attention masks
* Causal attention specifics (autoregressive generation)
* KV-caching mechanics for fast autoregressive decoding
* Sampling methods: greedy, beam, top-k, top-p, temperature, nucleus + calibration tricks
* Tokenization: BPE, SentencePiece, byte-level BPE, tokenization edge-cases
* Scaling laws: empirical relationships (model size, compute, data) — implications for design/tradeoffs
* Alignment basics: RLHF workflow, reward model training, DPO (Direct Preference Optimization) concepts

**Deep theory**

* Derive attention gradients and memory complexity (time/space)
* Analyze how KV-cache reduces recomputation and how memory scales in long-context
* Formalize sampling distributions (distribution shaping; temperature effect on entropy)
* Mathematical view of RLHF: policy gradient on LLMs with a learned reward, surrogate objectives

**Hands-on**

* Implement causal attention + KV cache in PyTorch from scratch and measure speedups vs naive generation.
* Implement sampling methods and quantify diversity vs quality tradeoffs on small LM.
* Tokenize datasets with multiple tokenizers; measure tokenization efficiency and edge-case behavior.

**Expert-level details often missed**

* Effective batching with KV-cache — the tradeoff between batch size and per-request latency.
* How subword tokenization choices change perplexity and downstream retrieval chunking.
* Practical instability of reward models in RLHF and DPO sensitivity to prompt phrasing.

**Mini-projects / deliverables**

* Small repo containing: causal attention + KV-cache module, sampling tools, tokenizer comparison report.

---

### Module 2 — Embedding Models & Representation Engineering

**Purpose:** build, evaluate, and fine-tune embedding models suitable for retrieval and semantic matching.

**Fundamental concepts**

* Sentence-BERT families, contrastive-finetuned encoders, siamese encoders
* LLM-based embeddings (pooled LM outputs, instruction-finetuned embedding heads)
* Distance measures: cosine vs dot-product vs Euclidean; effect of normalization
* Embedding post-processing: whitening, PCA projection, OPQ, quantization basics

**Deep theory**

* Contrastive loss (InfoNCE) applied to embedding learning; role of temperature and negatives
* Why normalization (unit norm) changes retrieval behavior under dot vs cosine
* Quantization theory: product quantization, asymmetric distance computation, impact on recall/latency

**Hands-on**

* Fine-tune a sentence-transformer via contrastive learning on a small domain dataset; evaluate Recall@k and MRR.
* Implement whitening and measure retrieval improvements (or regressions).
* Apply OPQ / PQ quantization on embeddings and benchmark recall vs memory.

**Expert-level details**

* Hard-negative mining strategies strongly impact quality; random negatives often insufficient.
* Batch size interacts with contrastive loss (effective negative pool).
* Embedding dimensionality tradeoffs for vector DBs (64/128/768 vs latency/memory).

**Mini-projects / deliverables**

* Embedding benchmark notebook: multiple encoders (BERT, SBERT, LLM-pool), postprocessing variants, quantitative comparison.

---

### Module 3 — Retrieval Systems & Indexing

**Purpose:** master exact and approximate nearest neighbor search, index design, and practical tradeoffs.

**Fundamental concepts**

* Exact search (brute-force) vs ANN strategies
* Graph-based: HNSW internals, parameters (M, efConstruction, efSearch)
* Quantization-based and product quantization (FAISS index families)
* Sharding, replication, vector compression, disk-backed indices
* Index maintenance: additions, deletes, reindexing strategies

**Deep theory**

* HNSW graph properties and search complexity; navigating small-world graphs
* PQ error bounds and asymmetric distance computation
* Index parameter tuning and theoretical impact on recall/latency

**Hands-on**

* Build FAISS indices: flat, IVF+PQ, HNSW; measure Recall@k, latency, RAM usage.
* Implement a toy HNSW or inspect internals via FAISS/HNSWlib.
* Practice index updates: real-time insertion batching and re-embedding strategies.

**Expert-level details**

* How efSearch vs efConstruction tradeoffs behave across dataset sizes and dimensionalities.
* Disk-backed index tradeoffs: use of memory-mapped indices and warmup strategies.
* Multi-index and hybrid (exact+ANN) fallbacks for safety-critical recall.

**Mini-projects / deliverables**

* Index comparison report with scripts to reproduce benchmarking (various dataset sizes and dims).

---

### Module 4 — Building RAG Pipelines (Chunking → Re-ranking → Generation)

**Purpose:** design and implement complete RAG pipeline, experiment with fusion strategies and reranking.

**Fundamental concepts**

* Chunking strategies: sliding window, semantic chunking, passage length vs retrieval quality
* Indexing pipeline (embedding store + metadata)
* Retrieval → re-rank (cross-encoder reranker vs bi-encoder)
* Fusion-in-decoder vs fusion-in-encoder patterns; late fusion strategies
* Query rewriting / query expansion (HyDE-style hallucinated docs, query reformulation)
* Reranking losses and training (pairwise/ranknet, listwise losses)

**Deep theory**

* Analyze retriever–reader coupling: tradeoffs between recall (retriever) and precision (reader)
* Formalize reranker training as learning-to-rank; loss functions and calibration
* Theoretical understanding of HyDE: generating hypothetical answers to improve retrieval context

**Hands-on**

* Build an end-to-end RAG pipeline with: chunker, embedder, FAISS/HNSW index, bi-encoder retriever, cross-encoder reranker, generator (small LM).
* Implement fusion-in-decoder where generator conditions on concatenated retrieved chunks.
* Implement HyDE (generate hypothetical answers with LM -> embed -> retrieve) and evaluate.

**Expert-level details**

* Chunk overlap vs redundancy: more overlap causes longer contexts but may improve grounding.
* Re-ranker latency costs vs retrieval precision: use cascade designs (fast retriever → expensive reranker for top-K).
* Prompt construction design choices (source attribution, provenance prompts) drastically affect hallucination.

**Mini-projects / deliverables**

* Full RAG repo that ingests a document corpus, builds indexes, runs retrieval+reranking, and produces grounded answers with provenance.

---

### Module 5 — Evaluation, Hallucination, and Alignment

**Purpose:** build rigorous, reproducible evaluation pipelines for faithfulness, groundedness, and safety; understand alignment techniques.

**Fundamental concepts**

* Retrieval metrics: Recall@k, MRR, nDCG
* Generative evaluation: factuality/faithfulness, hallucination detection, groundedness scoring
* Human evaluation protocols: instruction design, inter-annotator agreement, A/B testing
* Alignment: RLHF pipeline (preference data collection, reward model training, policy update), DPO overview

**Deep theory**

* Define groundedness formally: generation grounded in retrieved passages vs hallucinated facts; statistical tests to detect hallucination.
* Reward model bias and calibration; over-optimization risks (alignment gaming).
* Statistical design of human evals: power analysis, confidence intervals, bootstrap.

**Hands-on**

* Implement automated faithfulness checks: entailment-based verification (using NLI models), source overlap heuristics, citation checks.
* Set up small human evaluation pipeline (instruction templates, annotation schema) and compute agreement.
* Train a small reward model on pairwise preference data and run DPO/behavioral comparison.

**Expert-level details**

* NLI-based verification is brittle: entailed vs neutral vs contradiction scores need thresholds per domain.
* Reward models amplify dataset biases; careful dataset curation required.
* Human evals require careful prompt and instruction engineering to avoid annotator drift.

**Mini-projects / deliverables**

* Evaluation harness that computes retrieval+generation metrics, runs automated faithfulness checks, and stores human annotations.

---

### Module 6 — Engineering, Latency, and Productionization

**Purpose:** make RAG systems robust, low-latency, auditable and monitorable in production.

**Fundamental concepts**

* Vector DB choices and tradeoffs (Milvus, Weaviate, Pinecone, Milvus) — latency, consistency, cost
* Batching, async IO, KV-cache for LLMs, request-level concurrency
* Caching strategies: query result caching, document caching, reranker caching
* Observability: logging prompts, retrieval traces, provenance, user feedback loops
* SLOs, rate-limiting, graceful degradation strategies

**Deep theory**

* Queueing theory basics for latency modeling under load
* Cache hit-rate modeling vs cache size and request distribution
* Consistency models for vector DB replicas and their implication on freshness

**Hands-on**

* Deploy a prototype RAG stack (vector DB + service layer + LM inference) locally or on a cloud VM.
* Implement batching and async request handling; measure tail latency.
* Add logging/observability: per-request provenance, retrieval traces, metrics dashboards.

**Expert-level details**

* Tail latency often dominated by reranker or cross-encoder; use cascade or asynchronous reranking to mitigate.
* Vector DB indexing knobs depend on read/write patterns; reindex scheduling matters for freshness vs throughput.
* Monitoring must include detection for “silent regressions” (e.g., sudden increase in hallucination rate).

**Mini-projects / deliverables**

* Production-ready demo with SLO-focused benchmarks and monitoring dashboard.

---

### Module 7 — Advanced Topics: Memory, Agents, Hybrid Models

**Purpose:** research-level techniques that push RAG beyond retrieval: long-term memory, agentic behaviors, hybrids.

**Fundamental concepts**

* Memory-augmented LLMs: long-term vector memory with retrieval+decay policies
* Tool-using agents: function calling, tool APIs, safe sandboxing
* Self-RAG / Self-querying agents: models that iteratively query, update memory, and synthesize
* Hybrid: fine-tuning LLMs on retrieved context or distilling retrieved knowledge into model weights

**Deep theory**

* Memory consistency: policies for write/delete/forget; theoretical considerations for drift and stale facts
* Agent safety: formalize tool sandboxing, revertible actions, and constrained action spaces
* Hybrid approaches: when to fine-tune vs rely on retrieval (cost/latency/robustness tradeoffs)

**Hands-on**

* Implement a simple memory store that accumulates user interactions and retrieval-weighted decay.
* Implement a toy agent that calls an external calculator tool and verifies outputs.
* Implement a hybrid experiment: fine-tune small LM on RAG-augmented data and compare to pure RAG.

**Expert-level details**

* Memory must be audited—write policies to surface provenance and delete requests.
* Tool-using agents need strict input/output typing and error handling to avoid cascading failures.
* Hybrid fine-tuning risks overfitting to retrieval distribution; balance with held-out canonical data.

**Mini-projects / deliverables**

* Notebook & demo: memory-augmented RAG with example agent using a toy tool; hybrid finetune vs pure RAG comparison.

---

## WEEKLY PLAN (8-week recommended schedule)

(Adjust to 4–12 weeks by compressing/expanding modules; assume ~20–30 hours/week)

**Week 1 — Transformer & LLM fundamentals**

* Implement causal attention + KV-cache; sampling suite; tokenizer experiments.
* Deliverable: KV-cache module + sampling evaluation.

**Week 2 — Embeddings**

* Fine-tune SBERT/contrastive encoder; implement whitening/quantization experiments.
* Deliverable: embedding benchmark (Recall@k, MRR).

**Week 3 — Retrieval & Indexing**

* Build FAISS indexes (flat, IVF+PQ, HNSW); tune ef/M/efSearch; measure latency and recall.
* Deliverable: index benchmark scripts.

**Week 4 — RAG core pipeline**

* Implement chunker, embedder, retriever, basic generator fusion (concat top-K), implement reranker (cross-encoder).
* Deliverable: end-to-end RAG prototype on a small corpus.

**Week 5 — HyDE / Query rewriting & Re-ranking**

* Implement HyDE pipeline and cascade reranking (bi-encoder → cross-encoder).
* Deliverable: HyDE ablation & retrieval+answer quality comparison.

**Week 6 — Evaluation & Alignment**

* Build automated faithfulness checks, set up small human eval, train a reward model; run DPO/compare.
* Deliverable: evaluation harness + small human eval results.

**Week 7 — Engineering & Latency**

* Deploy prototype with vector DB, implement batching & caching, measure tail latency, implement monitoring.
* Deliverable: deployment demo + latency report.

**Week 8 — Advanced experiments & Capstone prep**

* Memory-augmented or agentic RAG experiment; finalize capstone report and reproducible scripts.
* Deliverable: capstone repo & paper-style writeup.

---

## MASTERY CHECKS (must satisfy all)

**Conceptual / Written**

* Explain mathematically how KV-cache changes generation complexity and provide pseudo-code.
* Derive InfoNCE-style contrastive loss impact on inner-product similarity and explain temperature role for retrieval embeddings.
* Formalize HyDE: write objective and intended retrieval improvement mechanism; analyze failure modes.

**Coding / Repro**

* End-to-end RAG system that: ingests documents (≥10k docs), builds index, retrieves top-K, reranks, and generates answers with provenance logging.
* Embedding & index benchmark achieving >X recall@10 (choose dataset; e.g., domain-specific) with measured latency <Y ms (define Y based on local hardware).
* Deployed demo (local/VM) with monitoring and a simple human evaluation flow.

**Evaluation**

* Run automated faithfulness metrics plus at least one human evaluation (n≥30 examples) and report agreement and confidence intervals.
* Show ablation demonstrating HyDE (or reranker) improves grounded answer rate.

---

## CAPSTONE (publication-quality)

**Title (example):** *“Hybrid RAG + Fine-tune: Improving Faithful Long-Context QA with Memory-Augmented Retrieval and Cascade Reranking”*

**Scope**

* Corpus: domain-specific collection (e.g., research papers, legal docs) ~10k–50k docs.
* Build: chunking pipeline, embedding model (contrastive finetune), FAISS/HNSW index, cascade retriever (bi-encoder → cross-encoder), generator (small LLM) with KV-cache.
* Propose and implement one research contribution (choose one):

  * Hybrid: distill retrieval context into LM via light-weight fine-tune and compare to pure RAG; or
  * Memory policy: design and evaluate decay/forget policy improving long-term factual consistency; or
  * Agentic RAG: iterative retrieval+planning pipeline for multi-step queries.
* Evaluate: Recall@k, precision@k, multiple faithfulness metrics, human eval (n≥200), latency & cost analysis.
* Deliverables: reproducible code, trained indexes, evaluation harness, 8–12 page paper + appendix with hyperparameters and seeds.

---

## WHAT NOT TO SKIP (strict)

* Don’t rely only on automatic metrics (FID/Rouge). Always run human eval for hallucination/factuality.
* Don’t ignore retrieval freshness & index maintenance; stale data breaks RAG silently.
* Don’t use high-dimensional embeddings with no quantization plan—memory blowup will block experiments.
* Don’t omit cascade reranking—many papers that report good RAG quality rely on cross-encoder rerankers for final precision.

---

## PREREQUISITES (explicit)

You must already be fluent in:

* Phase 2 (Deep Learning Core): Transformer internals, attention math, training pipelines.
* Phase 3 (Advanced Methods): optimization fundamentals, representation learning (contrastive basics).
* Production basics: containerization, basic web APIs, and database concepts.

If any of these are weak, remediate before Phase 3.5.

---

## FAILURE MODES & FIXES

**Failure:** High retrieval recall but generator still hallucinates.
**Fix:** Add cross-encoder reranker + hard constraint prompts that require source-supported tokens; test with entailment-based verification.

**Failure:** Index memory/execution budget exceeded.
**Fix:** Apply PQ/OPQ quantization, reduce embedding dims, shard index, or use hybrid exact+ANN.

**Failure:** Human eval shows low inter-annotator agreement.
**Fix:** Improve annotation schema, provide clear instructions, run pilot rounds and compute Krippendorff’s alpha.

**Failure:** Reward model/DPO optimization leads to degenerate outputs.
**Fix:** Regularize reward training, keep behavior cloning baseline, and run safety filters.

**Failure:** Tail latency spikes under load.
**Fix:** Implement batching, async reranking, circuit breakers, and degrade gracefully (fallback to cached answers).

---

## RESOURCES (depth-only — must-read / primary references & docs)

**Transformers & LLMs**

* Vaswani et al., *Attention is All You Need* (2017) — Transformer fundamentals.
* Relevant Transformer followups on positional encodings, scaling (e.g., papers on scaling laws).

**RAG, Retrieval & Embeddings**

* Lewis et al., *RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP* (2020) — architecture & evaluation motifs.
* Karpukhin et al., *Dense Passage Retrieval (DPR)* — dense retriever training.
* Reimers & Gurevych — *Sentence-BERT* — contrastive embeddings for sentences.
* FAQ / docs on HyDE and retrieval-then-generate approaches (original HyDE-like papers/notes).

**ANN & Indexing**

* FAISS documentation & HNSW papers (Malkov & Yashunin) — practical internals and parameter guidance.
* Product quantization & OPQ literature (Jegou et al., 2011).

**Evaluation & Alignment**

* Papers on hallucination/factuality evaluation (various recent NLP/LLM papers).
* RLHF and preference learning literature; DPO (Direct Preference Optimization) papers and blog posts.

**Systems & Production**

* Official docs for Pinecone, Milvus, Weaviate (for vector DB tradeoffs).
* Engineering posts/papers on batching, KV caching, and low-latency LLM serving.

---

If you want, I will now:

* generate the **exact capstone experiment plan** (datasets, hyperparams, scripts, compute estimates) ready to run; or
* produce a **detailed 8-week daily calendar** with step-by-step coding tasks, test cases, and checkpoints.

Which next artifact do you want immediately?

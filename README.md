# ORFEO vLLM Inference Optimization

This repository contains the work produced during my internship at **Area Science Park (LADE)** on the **ORFEO** HPC/AI platform.  
The goal is to **benchmark, analyze, and optimize LLM inference** served with **vLLM** on a **Kubernetes GPU cluster**, focusing on the trade-off between **throughput/latency**, **GPU memory usage**, and **benchmark accuracy**.

---

## Project Goals

- Deploy and operate **vLLM OpenAI-compatible endpoints** on Kubernetes.
- Measure **performance under load** (throughput, latency p50/p95/p99, error rate) using **Locust**.
- Measure **model quality** using **lm-evaluation-harness** on standard tasks (e.g., ARC, GSM8K, HellaSwag, TruthfulQA, WinoGrande).
- Run controlled tuning experiments (grid / one-parameter-at-a-time) on inference parameters such as:
  - `max_num_seqs`
  - `max_num_batched_tokens`
  - `enable_chunked_prefill`
  - `max_model_len`
  - `gpu_memory_utilization`
- Generate plots and reports to support **before vs after** comparisons and final recommendations.

---

## Repository Contents

- **Kubernetes deployment configs** for vLLM services
- **Locust scripts** to reproduce load test scenarios
- **Benchmarking scripts** (run orchestration, CSV aggregation)
- **Analysis notebooks / Python scripts** to generate plots and summarize results
- **Reports and slides** used to present findings and recommendations

---

## High-Level Workflow

1. **Baseline deployment**: deploy vLLM on Kubernetes with a reference configuration.
2. **Performance benchmarking**: run Locust load tests and collect throughput/latency/error metrics.
3. **Accuracy benchmarking**: run LM Eval on a fixed set of tasks and collect benchmark scores.
4. **Aggregation & analysis**: consolidate results into CSVs, generate plots, and compare models/configurations.
5. **Optimization & validation**: tune parameters and validate improvements with **Before vs After** analysis.

---

## Status / Next Steps

- GPT optimization completed with measurable improvements in throughput/latency/memory.
- Qwen3 and Llama 4 tuning planned as next steps (not completed due to time constraints).
- Future improvements: quantization studies, multi-replica serving, autoscaling, and cost/performance analysis (tokens/s per GPU and per watt).

---

## Acknowledgements

Special thanks to **Alberto Cazzaniga** (internship supervisor) for guidance and support throughout the project, and to **Isac** and **Niccolò** for their help with infrastructure, report reviews, and valuable advice.

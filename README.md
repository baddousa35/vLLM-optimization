ORFEO vLLM Inference Optimization

This repository contains the work produced during my internship at Area Science Park (LADE) on the ORFEO HPC/AI platform. The goal is to benchmark, analyze, and optimize LLM inference served with vLLM on a Kubernetes GPU cluster, focusing on the trade-off between throughput/latency, GPU memory usage, and benchmark accuracy.

Key objectives

Deploy and operate vLLM OpenAI-compatible endpoints on Kubernetes.

Measure performance under load (throughput, latency p50/p95/p99, error rate) using Locust.

Measure model quality using lm-evaluation-harness on standard tasks (e.g., ARC, GSM8K, HellaSwag, TruthfulQA, WinoGrande).

Run controlled tuning experiments (grid / one-parameter-at-a-time) on inference parameters such as:

max_num_seqs

max_num_batched_tokens

enable_chunked_prefill

max_model_len

gpu_memory_utilization

Generate clear plots and reports to support before vs after decisions.

What’s inside

Kubernetes manifests / deployment configs for vLLM services.

Locust scripts to reproduce load scenarios.

Benchmarking scripts (run orchestration, CSV aggregation).

Analysis notebooks / Python scripts to generate plots and summarize results.

Reports and slides used to present findings and recommendations.

Typical workflow (high level)

Deploy vLLM on Kubernetes (baseline).

Run Locust load tests → collect performance metrics.

Run LM Eval → collect accuracy metrics.

Aggregate results (CSV) → generate plots and trade-off analysis.

Tune parameters → validate improvements (Before vs After).

Status / next steps

GPT optimization completed with measurable improvements in throughput/latency/memory.

Qwen3 and Llama 4 tuning planned as next steps (time constraints during internship).

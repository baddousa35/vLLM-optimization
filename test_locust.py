from locust import HttpUser, task, between
import random
import json
import time
"""
Multi-Model LLM Stress Testing with Locust
------------------------------------------
This setup allows you to stress test multiple LLMs simultaneously
and compare performance across them.

Supported patterns:
- Weighted traffic per model
- Mixed prompt workloads
- Per-model performance labeling
- Token usage monitoring
"""
JWT_TOKEN= "..."
import os
import csv
from gevent.lock import Semaphore

TPS_CSV = os.getenv("TPS_CSV", "tokens_per_request.csv")
_csv_lock = Semaphore()

def append_csv_row(row: dict, path: str = TPS_CSV):
    with _csv_lock:
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                w.writeheader()
            w.writerow(row)

MODELS = [
    {
        "name": "GPT-OSS-120B",
        "host": "https://orfeo-llm.areasciencepark.it/vllm",
        "endpoint": "/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JWT_TOKEN}"
        },
        "payload": lambda prompt: {
            "model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        },
        "weight": 1
    },
    {
        "name": "Qwen3-VL-235B-Thinking",
        "host": "https://orfeo-llm.areasciencepark.it/vllm",
        "endpoint": "/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JWT_TOKEN}"
        },
        "payload": lambda prompt: {
            "model": "Qwen/Qwen3-VL-235B-A22B-Thinking-FP8",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        },
        "weight": 1
    },
    {
        "name": "Llama-4-Scout-17B",
        "host": "https://orfeo-llm.areasciencepark.it/vllm",
        "endpoint": "/v1/chat/completions",
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {JWT_TOKEN}"
        },
        "payload": lambda prompt: {
            "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000
        },
        "weight": 1
    }
]

PROMPTS = [
"Explain Kubernetes architecture in depth, including components, networking, and deployment strategies.",
"Write a Python module with multiple functions that reverse strings, handle Unicode, and include unit tests and docstrings.",
"Provide a comprehensive report on the impact of AI on healthcare, covering ethics, regulations, economics, and case studies.",
"Generate detailed marketing copy for a fintech startup, including product positioning, target audience analysis, and multiple tagline options."
]


def weighted_model_choice(models):
    weighted = []
    for m in models:
        weighted.extend([m] * m.get("weight", 1))
    return random.choice(weighted)

import logging
from locust import events
import logging
import os

# ---------- JWT CONFIG (single variable for all models) ----------
JWT_TOKEN = os.getenv("JWT_TOKEN", "...")


# ---------- ERROR LOGGING SETUP (Locust v2+ compatible) ----------
logging.basicConfig(
    filename="llm_errors.log",
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# Locust 2.x removed request_failure event, so we hook into the generic request event
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    if exception:
        logging.error(f"FAIL | {name} | {request_type} | {response_time}ms | {exception}")

class MultiModelUser(HttpUser):
    wait_time = between(1, 5)
    connection_timeout = 60
    network_timeout = 180
    host = "https://orfeo-llm.areasciencepark.it/vllm"

    @task
    def test_multiple_models(self):
        model = weighted_model_choice(MODELS)
        prompt = random.choice(PROMPTS)
        payload = model["payload"](prompt)

        request_name = f"{model['name']} Request"
        t0 = time.perf_counter()

        with self.client.post(
            model["endpoint"],
            json=payload,
            headers=model["headers"],   # tu gardes ton JWT comme avant
            name=request_name,
            catch_response=True
        ) as response:
            elapsed_s = max(time.perf_counter() - t0, 1e-9)
            ts_unix = int(time.time())

            status_code = response.status_code
            ok = (status_code == 200)

            prompt_tokens = completion_tokens = total_tokens = 0
            completion_tps = total_tps = 0.0
            parse_error = ""

            if ok:
                try:
                    data = response.json()
                    usage = data.get("usage") or {}

                    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

                    completion_tps = (completion_tokens / elapsed_s) if completion_tokens else 0.0
                    total_tps = (total_tokens / elapsed_s) if total_tokens else 0.0

                    response.success()

                except Exception as e:
                    parse_error = str(e)
                    response.failure(f"{model['name']} parse error: {e}")
            else:
                snippet = (response.text or "")[:200]
                response.failure(f"{model['name']} failed: {status_code} | {snippet}")

            append_csv_row({
                "ts_unix": ts_unix,
                "model": model["name"],
                "request_name": request_name,
                "endpoint": model["endpoint"],
                "status_code": status_code,
                "elapsed_s": round(elapsed_s, 6),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "completion_tps": round(completion_tps, 6),
                "total_tps": round(total_tps, 6),
                "parse_error": parse_error,
            })


# NOTE:
# Run with different hosts using --host per model OR use a reverse proxy if needed.
# For true parallel infra testing, run multiple locust workers per host.

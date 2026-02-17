# README — How to run an eval + log memory + generate CSVs

This project follows a simple workflow for each run **X**:

1) Launch the evaluation 
2) Log memory/GPU metrics while it runs
3) Stop logging when the run ends
4) Generate CSV files from the logs

---

## Run ID

Each run is identified by an integer **X** (example: `1`, `2`, `3`, ...).

---

## Step 1 — Start the evaluation

In a terminal:

./run-eval-X

## Step 2

Open a second terminal while the eval is running:

./log-memory-X

## Step 3

Stop memory logging when the eval finishes

Ctrl + C

## Step 4 — Generate CSVs from the run folder

After the run is done (and logging stopped), generate the CSVs:

python3 make_run_csvs.py runs/QWEN-run-<X>
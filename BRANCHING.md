# Suggested branching strategy for multiple runs

Use `main` for code + documentation.
For each experiment/run, create a branch that includes:
- the generated `phyphox_full_report/` folder OR the contents of `docs/sample-output/`
- a short note (e.g., `RESULTS.md` updates) describing what you changed (device, mounting, speed setting).

## Example workflow

```bash
# 1) Start from main
git checkout main
git pull

# 2) Create a new branch for a run (use a date-based name)
git checkout -b runs/hairdryer-2026-02-24

# 3) Put your two CSV files in ./data/
# 4) Run the analysis
python src/phyphox_all_in_one.py

# 5) Commit outputs for this run (choose ONE of the options below)

# Option A (recommended): keep raw CSV local, commit only the generated report + plots
git add phyphox_full_report/ report.html summary_metrics.csv 2>/dev/null || true
git add phyphox_full_report/
git commit -m "Add outputs for hairdryer run (low vs max)"

# Option B: copy key outputs into docs/sample-output (keeps repo smaller)
rm -rf docs/sample-output
mkdir -p docs/sample-output
cp -r phyphox_full_report/* docs/sample-output/
git add docs/sample-output/
git commit -m "Add sample output for hairdryer run"
```

## Why branches help

- Each branch becomes a snapshot you can share (e.g., in a job application)
- `main` stays clean and reusable for future experiments

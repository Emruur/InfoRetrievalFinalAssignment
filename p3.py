# -*- coding: utf-8 -*-
"""
Ensemble fusion of cross-encoder re-ranking runs using ranx.fuse.

5 methods across 3 families from Bassani & Romelli (2022):
  Score-based : CombSUM, CombMNZ
  Rank-based  : RRF, RBC
  Voting-based: BordaFuse
"""
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from ranx import Qrels, Run, fuse, evaluate, compare

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
DATA_ROOT     = "/data/s4402146"
base_path     = "./cross-encoder-reranker-ir-course-2026/"
finetuned_dir = os.path.join(base_path, "finetuned_models")
results_dir   = os.path.join(base_path, "fusion_results")
os.makedirs(results_dir, exist_ok=True)
data_folder  = os.path.join(DATA_ROOT, "trec2019-data")
qrels_path   = os.path.join(data_folder, "2019qrels-pass.txt")

# ----------------------------------------------------------------------
# 2. Load qrels
# ----------------------------------------------------------------------
qrels = Qrels.from_file(qrels_path, kind="trec")
print(f"Loaded qrels: {len(qrels.qrels)} queries")

# ----------------------------------------------------------------------
# 3. Auto-discover ranking runs produced by p2.py
# ----------------------------------------------------------------------
run_files = sorted([
    (d, os.path.join(finetuned_dir, d, "ranking.run"))
    for d in os.listdir(finetuned_dir)
    if os.path.exists(os.path.join(finetuned_dir, d, "ranking.run"))
])

if len(run_files) < 2:
    raise RuntimeError(f"Need at least 2 ranking.run files, found {len(run_files)}. Run p2.py first.")

runs = []
for model_dir, run_path in run_files:
    run = Run.from_file(run_path, kind="trec")
    run.name = model_dir  # used by compare()
    runs.append(run)
    print(f"  Loaded: {model_dir}")

# ----------------------------------------------------------------------
# 4. Five fusion methods (3 families, all unsupervised)
# ----------------------------------------------------------------------
# (method_id, norm,      display_label,            family)
METHODS = [
    ("sum",   "min-max", "CombSUM",   "score-based", {}),
    ("mnz",   "min-max", "CombMNZ",   "score-based", {}),
    ("rrf",   None,      "RRF",       "rank-based",  {}),
    ("rbc",   None,      "RBC",       "rank-based",  {"phi": 0.8}),
    ("borda", None,      "BordaFuse", "voting",      {}),
]

METRIC = "ndcg@10"

# ----------------------------------------------------------------------
# 5. Individual model baselines
# ----------------------------------------------------------------------
print(f"\n{'='*65}")
print("Individual model NDCG@10")
print(f"{'-'*65}")
rows = []
for run in runs:
    score = evaluate(qrels, run, METRIC)
    rows.append({"name": run.name, "family": "individual", METRIC: round(score * 100, 2)})
    print(f"  {run.name:<55}  {score*100:>6.2f}")

# ----------------------------------------------------------------------
# 6. Fusion
# ----------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"{'Method':<20} {'Family':<15} {'NDCG@10':>8}")
print(f"{'-'*65}")

fused_runs = []
for method_id, norm, label, family, params in METHODS:
    kwargs = {"runs": runs, "method": method_id}
    if norm:
        kwargs["norm"] = norm
    if params:
        kwargs["params"] = params
    combined = fuse(**kwargs)
    combined.name = label
    fused_runs.append(combined)

    score = evaluate(qrels, combined, METRIC)
    rows.append({"name": label, "family": family, METRIC: round(score * 100, 2)})
    print(f"  {label:<18} {family:<15} {score*100:>8.2f}")

    # Save fused ranking run
    combined.save(os.path.join(results_dir, f"{label}.run"), kind="trec")

# ----------------------------------------------------------------------
# 7. Save results to CSV and JSON
# ----------------------------------------------------------------------
df = pd.DataFrame(rows)
csv_path = os.path.join(results_dir, "fusion_results.csv")
df.to_csv(csv_path, index=False)
print(f"\nResults saved to: {csv_path}")

json_path = os.path.join(results_dir, "fusion_results.json")
with open(json_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"Results saved to: {json_path}")

# ----------------------------------------------------------------------
# 8. Bar chart
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#4c72b0" if r["family"] == "individual" else "#dd8452" for r in rows]
ax.bar([r["name"].split("-")[2] if r["family"] == "individual" else r["name"] for r in rows],
       [r[METRIC] for r in rows], color=colors)
ax.set_ylabel("NDCG@10")
ax.set_title("Individual Models vs Fusion Methods — NDCG@10")
ax.tick_params(axis='x', rotation=30)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#4c72b0", label="Individual"),
                   Patch(color="#dd8452", label="Fusion")], loc="lower right")
fig.tight_layout()
plot_path = os.path.join(results_dir, "fusion_comparison.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Plot saved to:    {plot_path}")

# ----------------------------------------------------------------------
# 9. Full comparison table (with significance testing)
# ----------------------------------------------------------------------
print(f"\n{'='*65}")
print("Full comparison (ranx compare, p<0.05)")
print(f"{'='*65}")
all_runs = runs + fused_runs
report = compare(qrels, runs=all_runs, metrics=[METRIC], max_p=0.05)
print(report)

report_path = os.path.join(results_dir, "compare_report.txt")
with open(report_path, "w") as f:
    f.write(str(report))
print(f"Compare report saved to: {report_path}")

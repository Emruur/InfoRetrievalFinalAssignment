# -*- coding: utf-8 -*-
"""
Ensemble fusion of cross-encoder re-ranking runs using ranx.fuse.

5 methods across 3 families from Bassani & Romelli (2022):
  Score-based : CombSUM, CombMNZ
  Rank-based  : RRF, RBC
  Voting-based: BordaFuse
"""
import os
from ranx import Qrels, Run, fuse, evaluate, compare

# ----------------------------------------------------------------------
# 1. Paths
# ----------------------------------------------------------------------
DATA_ROOT    = "/data/s4402146"
base_path    = "./cross-encoder-reranker-ir-course-2026/"
finetuned_dir = os.path.join(base_path, "finetuned_models")
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
    ("combsum",   "min-max", "CombSUM",    "score-based"),
    ("combmnz",   "min-max", "CombMNZ",    "score-based"),
    ("rrf",       None,      "RRF",         "rank-based"),
    ("rbc",       None,      "RBC",         "rank-based"),
    ("bordafuse", None,      "BordaFuse",   "voting"),
]

METRIC = "ndcg@10"

# ----------------------------------------------------------------------
# 5. Individual model baselines
# ----------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"Individual model NDCG@10")
print(f"{'-'*65}")
for run in runs:
    score = evaluate(qrels, run, METRIC)
    print(f"  {run.name:<55}  {score*100:>6.2f}")

# ----------------------------------------------------------------------
# 6. Fusion
# ----------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"{'Method':<20} {'Family':<15} {'NDCG@10':>8}")
print(f"{'-'*65}")

fused_runs = []
for method_id, norm, label, family in METHODS:
    kwargs = {"runs": runs, "method": method_id}
    if norm:
        kwargs["norm"] = norm
    combined = fuse(**kwargs)
    combined.name = label
    fused_runs.append(combined)

    score = evaluate(qrels, combined, METRIC)
    print(f"  {label:<18} {family:<15} {score*100:>8.2f}")

# ----------------------------------------------------------------------
# 7. Full comparison table (individual + fused, with significance)
# ----------------------------------------------------------------------
print(f"\n{'='*65}")
print("Full comparison (ranx compare)")
print(f"{'='*65}")
all_runs = runs + fused_runs
report = compare(
    qrels,
    runs=all_runs,
    metrics=[METRIC],
    max_p=0.05,
)
print(report)

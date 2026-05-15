# -*- coding: utf-8 -*-
import os
import gzip
import tarfile
import logging
import operator
from collections import defaultdict
import numpy as np
import tqdm
import pytrec_eval
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import util
from sentence_transformers.cross_encoder import CrossEncoder

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ----------------------------------------------------------------------
# 1. Configuration & Paths
# ----------------------------------------------------------------------
base_path = "./cross-encoder-reranker-ir-course-2026/"
finetuned_dir = os.path.join(base_path, "finetuned_models")

# Auto-discover completed model directories (must have model.safetensors or pytorch_model.bin)
model_dirs = sorted([
    os.path.join(finetuned_dir, d)
    for d in os.listdir(finetuned_dir)
    if os.path.isdir(os.path.join(finetuned_dir, d))
    and (
        os.path.exists(os.path.join(finetuned_dir, d, "model.safetensors"))
        or os.path.exists(os.path.join(finetuned_dir, d, "pytorch_model.bin"))
    )
])

logging.info(f"Found {len(model_dirs)} completed model(s):")
for d in model_dirs:
    logging.info(f"  {os.path.basename(d)}")

# ----------------------------------------------------------------------
# 2. Download Data (once)
# ----------------------------------------------------------------------
DATA_ROOT = "/data/s4402146"
queries_tar = os.path.join(DATA_ROOT, 'queries.tar.gz')
queries_train_path = os.path.join(DATA_ROOT, 'queries.train.tsv')
if not os.path.exists(queries_train_path):
    logging.info("Downloading queries.tar.gz")
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz', queries_tar)
    with tarfile.open(queries_tar, 'r:gz') as tar:
        tar.extractall(path=DATA_ROOT, filter='data')

data_folder = os.path.join(DATA_ROOT, 'trec2019-data')
os.makedirs(data_folder, exist_ok=True)

# Test queries
queries = {}
queries_filepath = os.path.join(data_folder, 'msmarco-test2019-queries.tsv.gz')
if not os.path.exists(queries_filepath):
    logging.info("Downloading msmarco-test2019-queries.tsv.gz")
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz', queries_filepath)

with gzip.open(queries_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# Relevance judgements
relevant_docs = defaultdict(lambda: defaultdict(int))
qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')
if not os.path.exists(qrels_filepath):
    logging.info("Downloading 2019qrels-pass.txt")
    util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)

with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, score = line.strip().split()
        if int(score) > 0:
            relevant_docs[qid][pid] = int(score)

relevant_qid = [qid for qid in queries if len(relevant_docs[qid]) > 0]

# Top-1000 candidates
passage_filepath = os.path.join(data_folder, 'msmarco-passagetest2019-top1000.tsv.gz')
if not os.path.exists(passage_filepath):
    logging.info("Downloading msmarco-passagetest2019-top1000.tsv.gz")
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz', passage_filepath)

passage_cand = {}
with gzip.open(passage_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, pid, query, passage = line.strip().split("\t")
        if qid not in passage_cand:
            passage_cand[qid] = []
        passage_cand[qid].append([pid, passage])

logging.info(f"Test queries with relevance judgements: {len(relevant_qid)}")

# ----------------------------------------------------------------------
# 3. Evaluate each model & plot training curves
# ----------------------------------------------------------------------
all_results = []

for model_save_path in model_dirs:
    model_name = os.path.basename(model_save_path)
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating: {model_name}")
    logging.info(f"{'='*60}")

    # --- Training curve ---
    log_path = os.path.join(model_save_path, "CERerankingEvaluator_train-eval_results_@10.csv")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df["steps"], df["MRR@10"], marker="o", label="MRR@10")
        ax.plot(df["steps"], df["NDCG@10"], marker="s", label="NDCG@10")
        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Score")
        ax.set_title(f"Training curve — {model_name}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = os.path.join(model_save_path, "training_curve.png")
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logging.info(f"Training plot saved to: {plot_path}")
    else:
        logging.warning(f"No training log found at {log_path}, skipping plot.")

    # --- Re-ranking ---
    model = CrossEncoder(model_save_path, max_length=512)

    run = {}
    for qid in tqdm.tqdm(relevant_qid, desc=f"Re-ranking [{model_name[:40]}]"):
        query = queries[qid]
        cand = passage_cand.get(qid, [])
        if not cand:
            continue

        pids = [c[0] for c in cand]
        cross_inp = [[query, c[1]] for c in cand]

        if model.config.num_labels > 1:
            cross_scores = model.predict(cross_inp, apply_softmax=True)[:, 1].tolist()
        else:
            cross_scores = model.predict(cross_inp).tolist()

        run[qid] = {pid: float(score) for pid, score in zip(pids, cross_scores)}

    # --- Metrics ---
    evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {'ndcg_cut.10', 'recall_100', 'map_cut.1000'})
    scores = evaluator.evaluate(run)

    ndcg  = np.mean([v["ndcg_cut_10"]    for v in scores.values()]) * 100
    rec   = np.mean([v["recall_100"]     for v in scores.values()]) * 100
    mAP   = np.mean([v["map_cut_1000"]   for v in scores.values()]) * 100

    all_results.append({"model": model_name, "NDCG@10": ndcg, "Recall@100": rec, "MAP@1000": mAP})

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"  NDCG@10:    {ndcg:.2f}")
    print(f"  Recall@100: {rec:.2f}")
    print(f"  MAP@1000:   {mAP:.2f}")
    print(f"{'='*40}\n")

    # --- Save ranking run ---
    sorted_run = []
    for qid, pid_scores in run.items():
        for rank, (pid, score) in enumerate(sorted(pid_scores.items(), key=operator.itemgetter(1), reverse=True)):
            sorted_run.append(f"{qid} Q0 {pid} {rank} {score} STANDARD")

    ranking_run_file_path = os.path.join(model_save_path, "ranking.run")
    with open(ranking_run_file_path, "w") as f:
        f.write("\n".join(sorted_run))
    logging.info(f"Ranking run saved to: {ranking_run_file_path}")

    del model

# ----------------------------------------------------------------------
# 4. Summary table
# ----------------------------------------------------------------------
print("\n" + "="*70)
print(f"{'Model':<50} {'NDCG@10':>8} {'Rec@100':>9} {'MAP@1000':>9}")
print("-"*70)
for r in all_results:
    print(f"{r['model']:<50} {r['NDCG@10']:>8.2f} {r['Recall@100']:>9.2f} {r['MAP@1000']:>9.2f}")
print("="*70)

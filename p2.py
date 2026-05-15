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
from sentence_transformers import util
from sentence_transformers.cross_encoder import CrossEncoder

# Setup logging so you can see the output in the terminal
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# ----------------------------------------------------------------------
# 1. Configuration & Paths
# ----------------------------------------------------------------------
base_path = "./cross-encoder-reranker-ir-course-2026/"
os.makedirs(base_path, exist_ok=True)

# TODO: REPLACE THIS STRING WITH THE ACTUAL FOLDER NAME FROM YOUR TRAINING RUN
model_save_path = os.path.join(base_path, "finetuned_models", "YOUR-MODEL-DIRECTORY-NAME-HERE") 

# ----------------------------------------------------------------------
# 2. Download Data
# ----------------------------------------------------------------------
# Download and extract queries
queries_tar = 'queries.tar.gz'
if not os.path.exists('queries.train.tsv'):
    logging.info("Downloading queries.tar.gz")
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz', queries_tar)
    with tarfile.open(queries_tar, 'r:gz') as tar:
        tar.extractall(filter='data')

data_folder = 'trec2019-data'
os.makedirs(data_folder, exist_ok=True)

# Read test queries
queries = {}
queries_filepath = os.path.join(data_folder, 'msmarco-test2019-queries.tsv.gz')
if not os.path.exists(queries_filepath):
    logging.info("Download " + os.path.basename(queries_filepath))
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-test2019-queries.tsv.gz', queries_filepath)

with gzip.open(queries_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# Read which passages are relevant
relevant_docs = defaultdict(lambda: defaultdict(int))
qrels_filepath = os.path.join(data_folder, '2019qrels-pass.txt')

if not os.path.exists(qrels_filepath):
    logging.info("Download " + os.path.basename(qrels_filepath))
    util.http_get('https://trec.nist.gov/data/deep/2019qrels-pass.txt', qrels_filepath)

with open(qrels_filepath) as fIn:
    for line in fIn:
        qid, _, pid, score = line.strip().split()
        score = int(score)
        if score > 0:
            relevant_docs[qid][pid] = score

# Only use queries that have at least one relevant passage
relevant_qid = []
for qid in queries:
    if len(relevant_docs[qid]) > 0:
        relevant_qid.append(qid)

# Read the top 1000 passages that are supposed to be re-ranked
passage_filepath = os.path.join(data_folder, 'msmarco-passagetest2019-top1000.tsv.gz')

if not os.path.exists(passage_filepath):
    logging.info("Download " + os.path.basename(passage_filepath))
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz', passage_filepath)

passage_cand = {}
with gzip.open(passage_filepath, 'rt', encoding='utf8') as fIn:
    for line in fIn:
        qid, pid, query, passage = line.strip().split("\t")
        if qid not in passage_cand:
            passage_cand[qid] = []
        passage_cand[qid].append([pid, passage])

logging.info(f"Queries: {len(queries)}")

# ----------------------------------------------------------------------
# 3. Prediction
# ----------------------------------------------------------------------
logging.info(f"Loading model from: {model_save_path}")
model = CrossEncoder(model_save_path, max_length=512)

run = {}
for qid in tqdm.tqdm(relevant_qid, desc="Evaluating Queries"):
    query = queries[qid]

    cand = passage_cand.get(qid, [])
    if not cand:
        continue
        
    pids = [c[0] for c in cand]
    corpus_sentences = [c[1] for c in cand]

    cross_inp = [[query, sent] for sent in corpus_sentences]

    if model.config.num_labels > 1: # Cross-Encoder that predicts more than 1 score
        cross_scores = model.predict(cross_inp, apply_softmax=True)[:, 1].tolist()
    else:
        cross_scores = model.predict(cross_inp).tolist()

    run[qid] = {}
    for idx, pid in enumerate(pids):
        run[qid][pid] = float(cross_scores[idx])

# ----------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------
evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, {'ndcg_cut.10', 'recall_100', 'map_cut.1000'})
scores = evaluator.evaluate(run)

print("\n" + "="*40)
print(f"Queries evaluated: {len(relevant_qid)}")
print("NDCG@10:    {:.2f}".format(np.mean([ele["ndcg_cut_10"] for ele in scores.values()]) * 100))
print("Recall@100: {:.2f}".format(np.mean([ele["recall_100"] for ele in scores.values()]) * 100))
print("MAP@1000:   {:.2f}".format(np.mean([ele["map_cut_1000"] for ele in scores.values()]) * 100))
print("="*40 + "\n")

# ----------------------------------------------------------------------
# 5. Store Ranking Run File
# ----------------------------------------------------------------------
# Sort candidate documents of each query based on their relevance score
for qid in run.keys():
    run[qid] = sorted(run[qid].items(), key=operator.itemgetter(1), reverse=True)

ranking_lines = []
for qid in run.keys():
    for rank, did_pred_score in enumerate(run[qid]):
        did, pred_score = did_pred_score
        line = f"{qid} Q0 {did} {rank} {pred_score} STANDARD"
        ranking_lines.append(line)

ranking_run_file_path = os.path.join(model_save_path, "ranking.run")
with open(ranking_run_file_path, "w+") as f_w:
    f_w.write("\n".join(ranking_lines))

logging.info(f"Ranking run file saved to: {ranking_run_file_path}")

# Print the first three lines (Replacing the !head bash command)
print("\nFirst 3 lines of the stored ranking run file:")
with open(ranking_run_file_path, "r") as f:
    for _ in range(3):
        print(f.readline().strip())

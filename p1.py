# -*- coding: utf-8 -*-
"""Cross-Encoder Fine-tuning for MS MARCO - Local Version

Original Colab: https://colab.research.google.com/drive/1yXKZGGsMBSnaGFeKVR20FqRNwyYV6Nw6

Run setup.sh first to create the virtual environment and install dependencies.
"""

import torch

# CUDA verification — runs before any heavy imports so failures are obvious
print("=" * 50)
print(f"PyTorch version : {torch.__version__}")
cuda_available = torch.cuda.is_available()
print(f"CUDA available  : {cuda_available}")
if cuda_available:
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    device = "cuda"
else:
    print("No GPU detected — running on CPU.")
    print("WARNING: 5e6 training samples on CPU will take days.")
    print("         Consider setting max_train_samples = 5e4 below.")
    device = "cpu"
print("=" * 50)

from torch.utils.data import DataLoader
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import InputExample
from datetime import datetime
import gzip
import os
import tarfile
import tqdm
import logging
from collections import defaultdict
import numpy as np
import sys
import pytrec_eval
import torch

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Local path for saving trained models (was Google Drive in Colab version)
base_path = "./cross-encoder-reranker-ir-course-2026/"
os.makedirs(base_path, exist_ok=True)

# Hyperparameters
train_batch_size = 32
num_epochs = 1
# 1 positive passage per 4 negative passages
pos_neg_ration = 4

# Maximal number of training samples.
# WARNING: 5e6 samples on a CPU-only machine will take many days.
# Reduce to ~5e4 for a quick test run on CPU.
max_train_samples = 5e6

# Load model — downloaded from HuggingFace Hub on first run,
# then cached locally at ~/.cache/huggingface/hub/
model_name = 'cross-encoder/ms-marco-MiniLM-L-2-v2'
# model_name = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'
# model_name = 'distilroberta-base'
model = CrossEncoder(model_name, num_labels=1, max_length=512)

model_save_path = os.path.join(
    base_path,
    'finetuned_models',
    'cross-encoder-' + model_name.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
os.makedirs(model_save_path, exist_ok=True)

# Download and extract MS MARCO queries if not already present
queries_tar = 'queries.tar.gz'
if not os.path.exists('queries.train.tsv'):
    logging.info("Downloading queries.tar.gz")
    util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz', queries_tar)
    with tarfile.open(queries_tar, 'r:gz') as tar:
        tar.extractall()

# Read the corpus (all passages)
data_folder = 'msmarco-data'
os.makedirs(data_folder, exist_ok=True)

corpus = {}
collection_filepath = os.path.join(data_folder, 'collection.tsv')
if not os.path.exists(collection_filepath):
    tar_filepath = os.path.join(data_folder, 'collection.tar.gz')
    if not os.path.exists(tar_filepath):
        logging.info("Download collection.tar.gz")
        util.http_get('https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz', tar_filepath)

    with tarfile.open(tar_filepath, "r:gz") as tar:
        tar.extractall(path=data_folder)

with open(collection_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        pid, passage = line.strip().split("\t")
        corpus[pid] = passage

# Read train queries
queries = {}
queries_filepath = os.path.join('queries.train.tsv')
with open(queries_filepath, 'r', encoding='utf8') as fIn:
    for line in fIn:
        qid, query = line.strip().split("\t")
        queries[qid] = query

# Build training & dev sets
train_samples = []
dev_samples = {}

num_dev_queries = 200
num_max_dev_negatives = 200

train_eval_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz')
if not os.path.exists(train_eval_filepath):
    logging.info("Download " + os.path.basename(train_eval_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train-eval.tsv.gz', train_eval_filepath)

with gzip.open(train_eval_filepath, 'rt') as fIn:
    for line in fIn:
        qid, pos_id, neg_id = line.strip().split()

        if qid not in dev_samples and len(dev_samples) < num_dev_queries:
            dev_samples[qid] = {'query': queries[qid], 'positive': set(), 'negative': set()}

        if qid in dev_samples:
            dev_samples[qid]['positive'].add(corpus[pos_id])

            if len(dev_samples[qid]['negative']) < num_max_dev_negatives:
                dev_samples[qid]['negative'].add(corpus[neg_id])

train_filepath = os.path.join(data_folder, 'msmarco-qidpidtriples.rnd-shuf.train.tsv.gz')
if not os.path.exists(train_filepath):
    logging.info("Download " + os.path.basename(train_filepath))
    util.http_get('https://sbert.net/datasets/msmarco-qidpidtriples.rnd-shuf.train.tsv.gz', train_filepath)

cnt = 0
with gzip.open(train_filepath, 'rt') as fIn:
    for line in tqdm.tqdm(fIn, unit_scale=True):
        qid, pos_id, neg_id = line.strip().split()

        if qid in dev_samples:
            continue

        query = queries[qid]
        if (cnt % (pos_neg_ration + 1)) == 0:
            passage = corpus[pos_id]
            label = 1
        else:
            passage = corpus[neg_id]
            label = 0

        train_samples.append(InputExample(texts=[query, passage], label=label))
        cnt += 1

        if cnt >= max_train_samples:
            break

# DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# Evaluator — runs every 1k steps on the dev set (MRR@10 by default)
evaluator = CERerankingEvaluator(dev_samples, name='train-eval')

# use_amp requires CUDA; falls back to False on CPU automatically in newer
# sentence-transformers, but be explicit to avoid warnings
use_amp = torch.cuda.is_available()

# Train and save the model locally under model_save_path
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=1000,
    warmup_steps=5000,
    output_path=model_save_path,
    use_amp=use_amp,
)

import sentence_transformers
print(sentence_transformers.__version__)
print(f"Model saved to: {model_save_path}")

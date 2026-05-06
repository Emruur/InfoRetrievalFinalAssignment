#!/usr/bin/env bash
set -e

DEST="/data/s4402146"

echo "==> Moving data files to $DEST"
mv -v msmarco-data/         "$DEST/msmarco-data"         2>/dev/null || true
mv -v queries.train.tsv     "$DEST/queries.train.tsv"    2>/dev/null || true
mv -v queries.dev.tsv       "$DEST/queries.dev.tsv"      2>/dev/null || true
mv -v queries.eval.tsv      "$DEST/queries.eval.tsv"     2>/dev/null || true
mv -v queries.tar.gz        "$DEST/queries.tar.gz"       2>/dev/null || true

echo "==> Done"

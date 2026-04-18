#!/bin/bash
# scripts/train_tokenizers.sh

set -e

CORPUS="data/tokenizer/sample_10M.txt"
MODELS_DIR="models/tokenizer"

mkdir -p "$MODELS_DIR"
mkdir -p data/tokenizer

echo "Sample sizes:"
du -h data/tokenizer/sample_*.txt

echo ""
echo "=== Training tokenizers on 10M sample ==="

# 32k
echo "[1/3] Training 32k vocabulary..."
spm_train \
  --input="$CORPUS" \
  --model_prefix="$MODELS_DIR/sp_32k" \
  --vocab_size=32000 \
  --character_coverage=0.9999 \
  --model_type=unigram \
  --max_sentence_length=1000000 \
  --shuffle_input_sentence=true \
  --num_threads=32 \
  --user_defined_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]"

# 64k
echo "[2/3] Training 64k vocabulary..."
spm_train \
  --input="$CORPUS" \
  --model_prefix="$MODELS_DIR/sp_64k" \
  --vocab_size=64000 \
  --character_coverage=0.9999 \
  --model_type=unigram \
  --max_sentence_length=1000000 \
  --shuffle_input_sentence=true \
  --num_threads=32 \
  --user_defined_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]"

# 100k
echo "[3/3] Training 100k vocabulary..."
spm_train \
  --input="$CORPUS" \
  --model_prefix="$MODELS_DIR/sp_100k" \
  --vocab_size=100000 \
  --character_coverage=0.9999 \
  --model_type=unigram \
  --max_sentence_length=1000000 \
  --shuffle_input_sentence=true \
  --num_threads=32 \
  --user_defined_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]"

echo ""
echo "=== Done ==="
ls -la "$MODELS_DIR"/*.model "$MODELS_DIR"/*.vocab
#!/bin/bash
# train_final_tokenizer.sh

VOCAB_SIZE=100000
CORPUS="data/corpus/corpus_cleaned.txt"
MODEL_DIR="models/tokenizer/final"

mkdir -p "$MODEL_DIR"

echo "=== Training final tokenizer (vocab=$VOCAB_SIZE) on full corpus ==="
echo "Input: $CORPUS"
echo "Output: $MODEL_DIR"

spm_train \
  --input="$CORPUS" \
  --model_prefix="$MODEL_DIR/sp_100k" \
  --vocab_size=$VOCAB_SIZE \
  --character_coverage=0.9999 \
  --model_type=unigram \
  --max_sentence_length=1000000 \
  --input_sentence_size=50000000 \
  --shuffle_input_sentence=true \
  --num_threads=32 \
  --train_extremely_large_corpus=true \
  --user_defined_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]"

echo "=== Done ==="
ls -la "$MODEL_DIR"/
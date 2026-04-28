#!/bin/bash
# train_final_tokenizer.sh

VOCAB_SIZE=64000
CORPUS="data/tokenizer/sample_20M.txt"
MODEL_DIR="models/tokenizer/multilingual"

mkdir -p "$MODEL_DIR"

echo "=== Training final tokenizer (vocab=$VOCAB_SIZE) on full corpus ==="
echo "Input: $CORPUS"
echo "Output: $MODEL_DIR"

spm_train \
  --input="$CORPUS" \
  --model_prefix="$MODEL_DIR/sp_64k" \
  --vocab_size=$VOCAB_SIZE \
  --character_coverage=0.9995 \
  --model_type=unigram \
  --max_sentence_length=1000000 \
  --input_sentence_size=20000000 \
  --train_extremely_large_corpus=true \
  --shuffle_input_sentence=true \
  --num_threads=32 \
  --user_defined_symbols="[PAD],[UNK],[CLS],[SEP],[MASK]"

echo "=== Done ==="
ls -la "$MODEL_DIR"/
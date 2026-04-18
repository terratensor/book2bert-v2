#!/usr/bin/env python3
# eval_tokenizers.py

import sentencepiece as spm
import json
import os
from collections import defaultdict

def evaluate_tokenizer(model_path, test_file, max_lines=100000):
    """Оценивает токенизатор на тестовом файле."""
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    total_words = 0
    total_tokens = 0
    unk_count = 0
    length_dist = defaultdict(int)
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            text = line.strip()
            if not text:
                continue
            
            # Слова (простое разбиение по пробелам)
            words = text.split()
            total_words += len(words)
            
            # Токены
            tokens = sp.EncodeAsIds(text)
            total_tokens += len(tokens)
            length_dist[len(tokens)] += 1
            
            # UNK
            unk_count += tokens.count(sp.unk_id())
    
    vocab_size = sp.GetPieceSize()
    coverage = 1 - (unk_count / total_tokens if total_tokens else 1)
    fertility = total_tokens / total_words if total_words else 0
    
    # Перцентили длин токенизированных последовательностей
    lengths = []
    for length, count in length_dist.items():
        lengths.extend([length] * count)
    lengths.sort()
    
    percentiles = {}
    for p in [50, 90, 95, 99]:
        idx = int(len(lengths) * p / 100)
        if idx < len(lengths):
            percentiles[f"p{p}"] = lengths[idx]
    
    return {
        'vocab_size': vocab_size,
        'coverage': round(coverage * 100, 2),
        'fertility': round(fertility, 2),
        'unk_rate': round((1 - coverage) * 100, 4),
        'total_tokens': total_tokens,
        'total_words': total_words,
        'avg_tokens_per_sentence': round(total_tokens / len(lengths) if lengths else 0, 1),
        'percentiles': percentiles,
    }

def main():
    models_dir = "models/tokenizer"
    test_file = "data/tokenizer/sample_test_1M.txt"
    
    models = {
        "32k": f"{models_dir}/sp_32k.model",
        "64k": f"{models_dir}/sp_64k.model",
        "100k": f"{models_dir}/sp_100k.model",
    }
    
    results = {}
    
    print("=== Evaluating Tokenizers ===\n")
    
    for name, model_path in models.items():
        if os.path.exists(model_path):
            print(f"Evaluating {name}...")
            results[name] = evaluate_tokenizer(model_path, test_file)
        else:
            print(f"Model {name} not found: {model_path}")
    
    print("\n=== Results ===\n")
    print(f"{'Metric':<30} {'32k':<15} {'64k':<15} {'100k':<15}")
    print("-" * 75)
    
    metrics = [
        ('vocab_size', 'Vocabulary size'),
        ('coverage', 'Coverage (%)'),
        ('fertility', 'Fertility (tokens/word)'),
        ('unk_rate', 'UNK rate (%)'),
        ('avg_tokens_per_sentence', 'Avg tokens/sentence'),
    ]
    
    for key, label in metrics:
        row = f"{label:<30}"
        for name in ['32k', '64k', '100k']:
            if name in results:
                val = results[name][key]
                row += f" {val:<15}"
            else:
                row += f" {'-':<15}"
        print(row)
    
    print("\n=== Token Length Percentiles ===\n")
    print(f"{'Percentile':<15} {'32k':<15} {'64k':<15} {'100k':<15}")
    print("-" * 60)
    
    for p in ['p50', 'p90', 'p95', 'p99']:
        label = p.replace('p', '') + '%'
        row = f"{label:<15}"
        for name in ['32k', '64k', '100k']:
            if name in results and 'percentiles' in results[name]:
                val = results[name]['percentiles'].get(p, '-')
                row += f" {val:<15}"
            else:
                row += f" {'-':<15}"
        print(row)
    
    # Сохраняем результаты
    with open(f"{models_dir}/comparison.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {models_dir}/comparison.json")

if __name__ == "__main__":
    main()
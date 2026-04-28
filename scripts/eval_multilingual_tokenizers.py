# eval_multilingual_tokenizers.py
import sentencepiece as spm
import json
import os
from collections import defaultdict

def evaluate_tokenizer(model_path, test_file, max_lines=100000):
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    # Метрики по языкам
    ru_words, ru_tokens, ru_unk = 0, 0, 0
    en_words, en_tokens, en_unk = 0, 0, 0
    mixed_words, mixed_tokens, mixed_unk = 0, 0, 0
    
    def count_letters(text):
        cyrillic = sum(1 for c in text if 'А' <= c <= 'я' or c in 'Ёё')
        latin = sum(1 for c in text if 'A' <= c <= 'z' or 'a' <= c <= 'z')
        return cyrillic, latin
    
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            
            text = line.strip()
            if not text:
                continue
            
            cyrillic, latin = count_letters(text)
            if cyrillic > latin * 5:
                lang = "ru"
            elif latin > cyrillic * 5:
                lang = "en"
            else:
                lang = "mixed"
            
            words = len(text.split())
            tokens = sp.EncodeAsIds(text)
            
            if lang == "ru":
                ru_words += words
                ru_tokens += len(tokens)
                ru_unk += tokens.count(sp.unk_id())
            elif lang == "en":
                en_words += words
                en_tokens += len(tokens)
                en_unk += tokens.count(sp.unk_id())
            else:
                mixed_words += words
                mixed_tokens += len(tokens)
                mixed_unk += tokens.count(sp.unk_id())
    
    return {
        'vocab_size': sp.GetPieceSize(),
        'ru': {
            'coverage': 1 - (ru_unk / ru_tokens if ru_tokens else 1),
            'fertility': ru_tokens / ru_words if ru_words else 0,
            'unk_rate': ru_unk / ru_tokens if ru_tokens else 0,
        },
        'en': {
            'coverage': 1 - (en_unk / en_tokens if en_tokens else 1),
            'fertility': en_tokens / en_words if en_words else 0,
            'unk_rate': en_unk / en_tokens if en_tokens else 0,
        },
        'mixed': {
            'coverage': 1 - (mixed_unk / mixed_tokens if mixed_tokens else 1),
            'fertility': mixed_tokens / mixed_words if mixed_words else 0,
            'unk_rate': mixed_unk / mixed_tokens if mixed_tokens else 0,
        },
    }

# Загружаем тестовый сэмпл
test_file = "data/tokenizer/sample_20M.txt"
models = {
    "32k": "models/tokenizer/final/32k/sp_32k.model",
    "42k": "models/tokenizer/multilingual/sp_42k.model",
    "50k": "models/tokenizer/multilingual/sp_50k.model",
    "64k": "models/tokenizer/multilingual/sp_64k.model",
}

results = {}
for name, path in models.items():
    if os.path.exists(path):
        print(f"Evaluating {name}...")
        results[name] = evaluate_tokenizer(path, test_file)
        print(f"  RU fertility: {results[name]['ru']['fertility']:.2f}, EN: {results[name]['en']['fertility']:.2f}")
        print(f"  RU unk_rate: {results[name]['ru']['unk_rate']:.4f}%, EN: {results[name]['en']['unk_rate']:.4f}%")

# Сохраняем
with open("models/tokenizer/multilingual/comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to models/tokenizer/multilingual/comparison.json")
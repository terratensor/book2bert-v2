#!/usr/bin/env python3
"""
Сравнение двух SentencePiece моделей:
- Мультиязычный 42k vs Русский 32k
"""

import sentencepiece as spm
import sys
import json
import unicodedata
from collections import Counter

def load_sp(model_path):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp

def analyze_text(sp, texts, label):
    stats = {
        "label": label,
        "total_sentences": len(texts),
        "total_words": 0,
        "total_tokens": 0,
        "unk_tokens": 0,
        "total_chars": 0,
        "token_freq": Counter(),
        "lengths": []
    }
    
    for text in texts:
        # Токенизация
        tokens = sp.encode_as_pieces(text)
        stats["total_tokens"] += len(tokens)
        stats["lengths"].append(len(tokens))
        
        # Подсчёт слов (пробелы)
        words = text.split()
        stats["total_words"] += len(words)
        stats["total_chars"] += len(text)
        
        # Частоты токенов
        for t in tokens:
            stats["token_freq"][t] += 1
            if t == "<unk>":
                stats["unk_tokens"] += 1
                
    return stats

def print_comparison(stats1, stats2):
    print("\n" + "="*80)
    print("📊 СРАВНИТЕЛЬНАЯ СТАТИСТИКА")
    print("="*80)
    
    for s in [stats1, stats2]:
        print(f"\n🔹 {s['label']}")
        print(f"   Предложений:      {s['total_sentences']:,}")
        print(f"   Слов:             {s['total_words']:,}")
        print(f"   Токенов:          {s['total_tokens']:,}")
        print(f"   Средн. токенов/слово: {s['total_tokens']/s['total_words']:.3f}")
        print(f"   UNK rate:         {s['unk_tokens']/s['total_tokens']*100:.4f}%")
        print(f"   Средн. длина:     {sum(s['lengths'])/len(s['lengths']):.1f} токенов/предложение")
        
        # Топ-5 самых частых токенов (исключая служебные)
        skip = {"<s>", "</s>", "<unk>", "<pad>"}
        common = [t for t, _ in s['token_freq'].most_common(10) if t not in skip]
        print(f"   Частые токены:    {', '.join(common[:5])}")

def show_tokenization_examples(sp1, sp2, texts, sp1_name, sp2_name):
    print("\n" + "="*80)
    print("🔍 ПРИМЕРЫ ТОКЕНИЗАЦИИ")
    print("="*80)
    
    for text in texts:
        t1 = sp1.encode_as_pieces(text)
        t2 = sp2.encode_as_pieces(text)
        
        print(f"\nОригинал: {text}")
        print(f"  {sp1_name:10} → {t1} ({len(t1)} ток)")
        print(f"  {sp2_name:10} → {t2} ({len(t2)} ток)")

def compare_vocab_overlap(sp1, sp2, sp1_name, sp2_name):
    vocab1 = set([sp1.id_to_piece(i) for i in range(sp1.get_piece_size())])
    vocab2 = set([sp2.id_to_piece(i) for i in range(sp2.get_piece_size())])
    
    common = vocab1 & vocab2
    only1 = vocab1 - vocab2
    only2 = vocab2 - vocab1
    
    print("\n" + "="*80)
    print("🔄 ПЕРЕКРЫТИЕ СЛОВАРЕЙ")
    print("="*80)
    print(f"  Размер {sp1_name}: {len(vocab1)}")
    print(f"  Размер {sp2_name}: {len(vocab2)}")
    print(f"  Общие токены:      {len(common)} ({len(common)/len(vocab1)*100:.1f}% от {sp1_name})")
    print(f"  Только в {sp1_name}: {len(only1)}")
    print(f"  Только в {sp2_name}: {len(only2)}")
    
    # Примеры уникальных токенов
    print(f"\n  Уникальные для {sp1_name} (первые 10): {sorted(list(only1))[:10]}")
    print(f"  Уникальные для {sp2_name} (первые 10): {sorted(list(only2))[:10]}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Использование: python compare_tokenizers.py <model_42k.model> <model_32k.model> <test_file.txt>")
        print("test_file.txt должен содержать по 1 предложению на строку")
        sys.exit(1)
        
    model_42k = load_sp(sys.argv[1])
    model_32k = load_sp(sys.argv[2])
    test_file = sys.argv[3]
    
    with open(test_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
        
    # Берём первые 10 000 строк для скорости
    sample_texts = texts[:10000]
    
    print(f"📂 Загружено {len(texts)} строк, анализируем {len(sample_texts)}...")
    
    stats_42k = analyze_text(model_42k, sample_texts, "Мультиязычный 42k")
    stats_32k = analyze_text(model_32k, sample_texts, "Русский 32k")
    
    print_comparison(stats_42k, stats_32k)
    compare_vocab_overlap(model_42k, model_32k, "42k", "32k")
    
    # Примеры для анализа
    examples = [
        "Программирование — это искусство.",
        "Debugging code is hard.",
        "Мы использовали Python и C++.",
        "Владимир Владимирович Путин.",
        "The quick brown fox jumps over the lazy dog.",
        "Это тест на смешанный текст с English.",
        "Коэффициент α = 0.05 при p-value < 0.05.",
        "API endpoint: /api/v1/users."
    ]
    show_tokenization_examples(model_42k, model_32k, examples, "42k", "32k")
    
    print("\n" + "="*80)
    print("💡 РЕКОМЕНДАЦИЯ")
    print("="*80)
    if stats_42k["unk_tokens"] < stats_32k["unk_tokens"]:
        print("✅ 42k лучше по UNK rate. Берите 42k, если память позволяет (+15M параметров).")
    elif stats_32k["total_tokens"]/stats_32k["total_words"] < 1.6:
        print("✅ 32k эффективнее по сжатию. Берите 32k, если важна скорость/память.")
    else:
        print("⚖️ Разница минимальна. 32k — безопасный выбор для русского.")
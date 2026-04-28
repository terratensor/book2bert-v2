#!/usr/bin/env python3
"""
Тестирование обученной BERT модели.
Показывает, чему научилась модель на разных этапах обучения.

Исправления v2:
- Маскирование на уровне токенов (не целых слов)
- Исправлен test_analogy (index out of bounds)
- Улучшенные тестовые фразы
"""

import os
import torch
import json
import random
import glob
import argparse
import numpy as np
from transformers import BertForMaskedLM
from simple_sp_tokenizer import SimpleSPTokenizer


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================================================

def load_model(model_path, tokenizer_path):
    """Загружает модель и токенизатор."""
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer


# ============================================================================
# MLM: ПРЕДСКАЗАНИЕ [MASK]
# ============================================================================

def predict_masked(text, model, tokenizer, top_k=5):
    """
    Предсказывает [MASK] в тексте.
    
    Example:
        text = "Москва — [MASK] России"
        returns: ["столица", "город", "сердце", "центр", "душа"]
    """
    encoded = tokenizer(text, max_length=512, return_tensors='pt')
    
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    
    # Находим позицию [MASK]
    mask_positions = (encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    
    if len(mask_positions) == 0:
        return []
    
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0, mask_positions[0]]
        probs = torch.softmax(logits, dim=-1)
        top_tokens = torch.topk(probs, top_k).indices.tolist()
        top_probs = torch.topk(probs, top_k).values.tolist()
    
    # Декодируем токены
    predictions = []
    for token_id, prob in zip(top_tokens, top_probs):
        token = tokenizer.convert_ids_to_tokens(token_id)
        token = token.replace('▁', '').strip()
        predictions.append((token, prob))
    
    return predictions


def predict_next_word(text, model, tokenizer, top_k=5):
    """
    Предсказывает следующее слово (без [MASK]).
    
    Example:
        text = "Москва — столица"
        returns: ["России", "мира", "СССР", "Руси", "государства"]
    """
    text_with_mask = text + " [MASK]"
    return predict_masked(text_with_mask, model, tokenizer, top_k)


# ============================================================================
# МАСКИРОВАНИЕ НА УРОВНЕ ТОКЕНОВ (НОВОЕ!)
# ============================================================================

def mask_random_token(text, tokenizer):
    """
    Маскирует случайный ЗНАЧИМЫЙ токен в предложении.
    
    Возвращает:
        masked_text: текст с [MASK] вместо токена
        true_id: ID замаскированного токена
        true_token: строка замаскированного токена (очищенная)
    
    Если не найдено подходящих токенов — возвращает (None, None, None).
    """
    # Токенизируем
    encoded = tokenizer(text, max_length=512, return_tensors='pt')
    input_ids = encoded['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    # Специальные токены, которые нельзя маскировать
    forbidden_ids = {
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.mask_token_id,
        tokenizer.unk_token_id,
    }
    
    # Пунктуация и одиночные символы
    punctuation = {',', '.', '!', '?', ':', ';', '-', '—', '…', '(', ')', '[', ']', '{', '}', 
                   '"', "'", '«', '»', '„', '“', '•', '/', '\\', '–', '―', '′', '″'}
    
    # Находим "хорошие" позиции для маскирования
    valid_positions = []
    for i, tid in enumerate(input_ids):
        if tid in forbidden_ids:
            continue
        
        token_str = tokens[i].replace('▁', '').strip()
        
        # Пропускаем пунктуацию
        if token_str in punctuation:
            continue
        
        # Пропускаем одиночные буквы/цифры (но не в составе слова)
        if len(token_str) <= 1 and not token_str.isalpha():
            continue
        
        valid_positions.append(i)
    
    if len(valid_positions) < 1:
        return None, None, None
    
    # Выбираем случайную позицию
    mask_pos = random.choice(valid_positions)
    true_token = tokens[mask_pos]
    true_id = input_ids[mask_pos]
    
    # Создаём замаскированную версию
    masked_ids = input_ids.copy()
    masked_ids[mask_pos] = tokenizer.mask_token_id
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True)
    
    # Очищаем true_token от SentencePiece артефактов
    true_token_clean = true_token.replace('▁', '').strip()
    
    return masked_text, true_id, true_token_clean


# ============================================================================
# ТЕСТ НА СЛУЧАЙНЫХ ПРЕДЛОЖЕНИЯХ (ОБНОВЛЁН)
# ============================================================================

def test_on_random_sentences(jsonl_path, model, tokenizer, num_examples=5):
    """
    Берёт случайные предложения из cleaned датасета и тестирует MLM
    с маскированием на уровне ТОКЕНОВ.
    """
    # Если передан путь к директории — берём случайный файл
    if os.path.isdir(jsonl_path):
        jsonl_files = glob.glob(f"{jsonl_path}/*.jsonl")
        if not jsonl_files:
            print("No JSONL files found!")
            return []
        jsonl_path = random.choice(jsonl_files)
    
    print(f"Using file: {jsonl_path}")
    
    sentences = []
    with open(jsonl_path, 'r') as f:
        all_lines = f.readlines()
        if len(all_lines) == 0:
            return []
        
        selected = random.sample(all_lines, min(num_examples * 10, len(all_lines)))
        
        for line in selected:
            data = json.loads(line)
            text = data.get('text', '').strip()
            # Берем предложения разумной длины
            if 30 < len(text) < 300:
                sentences.append(text)
                if len(sentences) >= num_examples:
                    break
    
    results = []
    for sent in sentences:
        masked_text, true_id, true_token = mask_random_token(sent, tokenizer)
        if masked_text is None:
            continue
        
        predictions = predict_masked(masked_text, model, tokenizer, top_k=10)
        
        results.append({
            'original': sent,
            'masked': masked_text,
            'true_token': true_token,
            'predictions': [p[0] for p in predictions],
            'predictions_with_probs': predictions,
        })
    
    return results


# ============================================================================
# ТЕСТ АНАЛОГИЙ (ИСПРАВЛЕН)
# ============================================================================

def test_analogy(word_a, word_b, word_c, model, tokenizer, top_k=5):
    """
    Тестирует аналогию: word_a - word_b + word_c = ?
    Использует ТОЛЬКО эмбеддинги первых токенов слов.
    """
    vocab_size = tokenizer.vocab_size
    hidden_size = model.config.hidden_size
    
    def get_single_embedding(word):
        """Получает эмбеддинг ПЕРВОГО токена слова."""
        tid = tokenizer.convert_tokens_to_ids(word)
        
        # Если слово не в словаре как единый токен — ищем его первый подтокен
        if tid == tokenizer.unk_token_id or tid >= vocab_size:
            ids = tokenizer.encode(word, max_length=512)
            ids = [i for i in ids if i not in [
                tokenizer.cls_token_id, tokenizer.sep_token_id,
                tokenizer.pad_token_id, tokenizer.unk_token_id
            ]]
            if len(ids) == 0:
                return None
            tid = ids[0]
        
        if tid < 0 or tid >= vocab_size:
            return None
        
        # Берём эмбеддинг из word_embeddings
        with torch.no_grad():
            embedding = model.bert.embeddings.word_embeddings.weight[tid]
            return embedding.clone()
    
    emb_a = get_single_embedding(word_a)
    emb_b = get_single_embedding(word_b)
    emb_c = get_single_embedding(word_c)
    
    if emb_a is None or emb_b is None or emb_c is None:
        print(f"  ⚠️  Could not find tokens for: {word_a}, {word_b}, {word_c}")
        return []
    
    # Все эмбеддинги должны быть одинаковой размерности [hidden_size]
    # Убедимся, что это так
    assert emb_a.shape == (hidden_size,), f"emb_a shape: {emb_a.shape}"
    assert emb_b.shape == (hidden_size,), f"emb_b shape: {emb_b.shape}"
    assert emb_c.shape == (hidden_size,), f"emb_c shape: {emb_c.shape}"
    
    # Вычисляем целевой эмбеддинг
    target_emb = emb_a - emb_b + emb_c  # [hidden_size]
    
    # Все эмбеддинги слов: [vocab_size, hidden_size]
    word_embeddings = model.bert.embeddings.word_embeddings.weight
    
    # Приводим target_emb к [1, hidden_size] для косинусного сходства
    target_emb = target_emb.unsqueeze(0)  # [1, hidden_size]
    
    # Косинусное сходство: [1, hidden_size] vs [vocab_size, hidden_size] → [vocab_size]
    similarity = torch.cosine_similarity(target_emb, word_embeddings)  # [vocab_size]
    
    # Исключаем входные слова
    exclude_ids = set()
    for word in [word_a, word_b, word_c]:
        tid = tokenizer.convert_tokens_to_ids(word)
        if isinstance(tid, int) and 0 <= tid < vocab_size:
            exclude_ids.add(tid)
    
    for eid in exclude_ids:
        similarity[eid] = -float('inf')
    
    # Берём top_k
    top_k = min(top_k, len(similarity) - len(exclude_ids))
    if top_k <= 0:
        return []
    
    top_ids = torch.topk(similarity, top_k).indices.tolist()
    
    results = []
    for tid in top_ids:
        if 0 <= tid < vocab_size:
            token = tokenizer.convert_ids_to_tokens(tid)
            token = token.replace('▁', '').strip()
            if token:
                results.append(token)
    
    return results

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Тестирование BERT модели")
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer/multilingual/sp_42k.model')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['interactive', 'random', 'analogy', 'all'])
    parser.add_argument('--cleaned', type=str, default='data/cleaned',
                       help='Директория с JSONL для случайных примеров')
    parser.add_argument('--num-examples', type=int, default=5,
                       help='Количество случайных примеров')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Loading model from {args.model}")
    model, tokenizer = load_model(args.model, args.tokenizer)
    print("Model loaded!")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, "
          f"PAD={tokenizer.pad_token_id}, MASK={tokenizer.mask_token_id}")
    print("=" * 80)
    
    if args.mode == 'interactive':
        print("\nИнтерактивный режим. Введите текст с [MASK] или 'exit' для выхода.\n")
        print("Примеры:")
        print("  Москва — [MASK] России")
        print("  2 + 2 = [MASK]")
        print("  Я [MASK] в магазин")
        print()
        
        while True:
            text = input(">>> ")
            if text.lower() == 'exit':
                break
            
            if '[MASK]' in text:
                predictions = predict_masked(text, model, tokenizer, top_k=10)
                print(f"\nПредсказания для [MASK]:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {word} ({prob:.3f})")
            else:
                predictions = predict_next_word(text, model, tokenizer, top_k=10)
                print(f"\nСледующее слово:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {word} ({prob:.3f})")
            print()
    
    elif args.mode == 'random':
        print(f"\nТестирование на случайных предложениях из {args.cleaned}")
        print(f"Маскирование на уровне ТОКЕНОВ (не целых слов)\n")
        
        results = test_on_random_sentences(
            args.cleaned, model, tokenizer, num_examples=args.num_examples
        )
        
        if not results:
            print("Не удалось найти подходящие предложения.")
            return
        
        for i, r in enumerate(results, 1):
            print(f"\n--- Пример {i} ---")
            print(f"Оригинал:     {r['original'][:200]}")
            print(f"С маской:     {r['masked'][:200]}")
            print(f"Заменили:     [{r['true_token']}]")
            print(f"Топ-10:       {', '.join(r['predictions'][:10])}")
            print(f"Результат:    {'✅' if r['true_token'] in r['predictions'][:5] else '❌'} "
                  f"(в топ-5: {r['true_token'] in r['predictions'][:5]}, "
                  f"в топ-10: {r['true_token'] in r['predictions'][:10]})")
    
    elif args.mode == 'analogy':
        print("\nТестирование аналогий:")
        
        analogies = [
            ("Москва", "Россия", "Париж", "Франция"),
            ("король", "мужчина", "королева", "женщина"),
            ("ходить", "шёл", "бегать", "бежал"),
            ("хороший", "лучше", "плохой", "хуже"),
        ]
        
        for a, b, c, expected in analogies:
            result = test_analogy(a, b, c, model, tokenizer, top_k=5)
            status = "✅" if expected in result else "❌"
            print(f"\n{a} - {b} + {c} = ? (ожидается: {expected}) {status}")
            print(f"  Предсказания: {result}")
    
    elif args.mode == 'all':
        # ====================================================================
        # 1. MLM ТЕСТ
        # ====================================================================
        print("\n" + "=" * 80)
        print("1. MLM тест (предсказание [MASK])")
        print("=" * 80)
        
        test_texts = [
            # Простые факты
            "Москва — [MASK] России.",
            "Столица России — [MASK].",
            "The capital of England is [MASK].",
            "[MASK] is the capital of France.",
            
            # Грамматика
            "Я [MASK] в магазин.",
            "Он [MASK] книгу.",
            
            # Числа и последовательности
            "1, 2, [MASK], 4.",
            "Понедельник, вторник, [MASK], четверг.",
            
            # Цвета
            "Красный, зелёный, [MASK], жёлтый.",
            
            # Известные люди
            "Владимир Владимирович [MASK].",
            "Президент России — Владимир [MASK].",
            
            # Наука
            "Вода состоит из водорода и [MASK].",
            
            # Математика
            "2 + 2 = [MASK]",
            "α = 0.[MASK]",
        ]
        
        for text in test_texts:
            predictions = predict_masked(text, model, tokenizer, top_k=10)
            print(f"\n{text}")
            for word, prob in predictions[:5]:
                print(f"  → {word} ({prob:.3f})")
        
        # ====================================================================
        # 2. СЛУЧАЙНЫЕ ПРЕДЛОЖЕНИЯ (МАСКИРОВАНИЕ ТОКЕНОВ)
        # ====================================================================
        print("\n" + "=" * 80)
        print("2. Маскирование случайных ТОКЕНОВ (не слов)")
        print("=" * 80)
        
        results = test_on_random_sentences(
            args.cleaned, model, tokenizer, num_examples=5
        )
        
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n--- Пример {i} ---")
                print(f"Оригинал:     {r['original'][:200]}")
                print(f"С маской:     {r['masked'][:200]}")
                print(f"Заменили:     [{r['true_token']}]")
                print(f"Топ-10:       {', '.join(r['predictions'][:10])}")
                in_top5 = r['true_token'] in r['predictions'][:5]
                in_top10 = r['true_token'] in r['predictions'][:10]
                print(f"Результат:    {'✅' if in_top5 else '❌'} "
                      f"(в топ-5: {in_top5}, в топ-10: {in_top10})")
        else:
            print("Не удалось найти подходящие предложения.")
        
        # ====================================================================
        # 3. АНАЛОГИИ
        # ====================================================================
        print("\n" + "=" * 80)
        print("3. Тест аналогий")
        print("=" * 80)
        
        analogies = [
            ("Москва", "Россия", "Париж", "Франция"),
            ("король", "мужчина", "королева", "женщина"),
            ("ходить", "шёл", "бегать", "бежал"),
        ]
        
        for a, b, c, expected in analogies:
            result = test_analogy(a, b, c, model, tokenizer, top_k=5)
            status = "✅" if expected in result else "❌"
            print(f"\n{a} - {b} + {c} = ? (ожидается: {expected}) {status}")
            print(f"  Предсказания: {result}")


if __name__ == "__main__":
    main()
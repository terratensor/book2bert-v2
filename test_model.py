#!/usr/bin/env python3
"""
Тестирование обученной BERT модели.
Поддерживает выбор языка, расширенный контекст, множественное маскирование.
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
# ЗАГРУЗКА МОДЕЛИ И ИНДЕКСА
# ============================================================================

def load_model(model_path, tokenizer_path):
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, tokenizer


def load_book_index(index_path):
    """Загружает индекс книг с информацией о языках."""
    if not index_path or not os.path.exists(index_path):
        return {}
    
    index = {}
    with open(index_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            book_id = data.get('book_id', '')
            if book_id:
                index[book_id] = data
    return index


# ============================================================================
# MLM: ПРЕДСКАЗАНИЕ [MASK] (поддержка нескольких масок)
# ============================================================================

def predict_masked(text, model, tokenizer, top_k=5):
    """Предсказывает ВСЕ [MASK] в тексте."""
    encoded = tokenizer(text, max_length=512, return_tensors='pt')
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    
    mask_positions = (encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return []
    
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0]
    
    all_predictions = []
    for pos in mask_positions:
        probs = torch.softmax(logits[pos], dim=-1)
        top_tokens = torch.topk(probs, top_k).indices.tolist()
        top_probs = torch.topk(probs, top_k).values.tolist()
        
        predictions = []
        for token_id, prob in zip(top_tokens, top_probs):
            token = tokenizer.convert_ids_to_tokens(token_id)
            token = token.replace('▁', '').strip()
            predictions.append((token, float(prob)))
        all_predictions.append(predictions)
    
    return all_predictions


# ============================================================================
# МАСКИРОВАНИЕ ТОКЕНОВ
# ============================================================================

def get_valid_positions(input_ids, tokens, tokenizer, min_length=2):
    """Возвращает список позиций, которые можно маскировать."""
    forbidden_ids = {
        tokenizer.cls_token_id, tokenizer.sep_token_id,
        tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.unk_token_id,
    }
    punctuation = {',', '.', '!', '?', ':', ';', '-', '—', '…', '(', ')', '[', ']', '{', '}',
                   '"', "'", '«', '»', '„', '“', '•', '/', '\\', '–', '―', '′', '″'}
    
    valid = []
    for i, tid in enumerate(input_ids):
        if tid in forbidden_ids:
            continue
        token_str = tokens[i].replace('▁', '').strip()
        if token_str in punctuation:
            continue
        if len(token_str) < min_length:
            continue
        valid.append(i)
    return valid


def mask_single_token(text, tokenizer):
    """Маскирует один случайный токен."""
    encoded = tokenizer(text, max_length=512, return_tensors='pt')
    input_ids = encoded['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    valid_positions = get_valid_positions(input_ids, tokens, tokenizer)
    if len(valid_positions) < 1:
        return None, None, None, None
    
    mask_pos = random.choice(valid_positions)
    true_token = tokens[mask_pos].replace('▁', '').strip()
    
    masked_ids = input_ids.copy()
    masked_ids[mask_pos] = tokenizer.mask_token_id
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True)
    
    return masked_text, true_token, [mask_pos], [true_token]


def mask_multiple_tokens(text, tokenizer, num_masks=3, min_distance=3):
    """Маскирует несколько случайных токенов."""
    encoded = tokenizer(text, max_length=512, return_tensors='pt')
    input_ids = encoded['input_ids'][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    valid_positions = get_valid_positions(input_ids, tokens, tokenizer)
    if len(valid_positions) < num_masks:
        return None, None, None, None
    
    # Выбираем позиции с минимальным расстоянием
    selected = []
    available = valid_positions.copy()
    random.shuffle(available)
    
    for pos in available:
        too_close = False
        for sel in selected:
            if abs(pos - sel) < min_distance:
                too_close = True
                break
        if not too_close:
            selected.append(pos)
        if len(selected) >= num_masks:
            break
    
    if len(selected) < 2:
        return mask_single_token(text, tokenizer)
    
    selected.sort()
    true_tokens = [tokens[p].replace('▁', '').strip() for p in selected]
    
    masked_ids = input_ids.copy()
    for pos in selected:
        masked_ids[pos] = tokenizer.mask_token_id
    masked_text = tokenizer.decode(masked_ids, skip_special_tokens=True)
    
    return masked_text, true_tokens, selected, true_tokens


# ============================================================================
# ВЫБОРКА ПРЕДЛОЖЕНИЙ ПО ЯЗЫКУ
# ============================================================================

def select_sentences_by_language(index, processed_dir, language, count, min_len=30, max_len=500):
    """Выбирает случайные предложения на заданном языке."""
    # Собираем книги нужного языка
    books = []
    for book_id, info in index.items():
        if info.get('language') == language:
            books.append(book_id)
    
    if not books:
        print(f"No books found for language: {language}")
        return []
    
    sentences = []
    attempts = 0
    max_attempts = count * 50
    
    while len(sentences) < count and attempts < max_attempts:
        book_id = random.choice(books)
        jsonl_path = os.path.join(processed_dir, f"{book_id}.jsonl")
        
        if not os.path.exists(jsonl_path):
            attempts += 1
            continue
        
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                attempts += 1
                continue
            
            # Берём случайное предложение
            line = random.choice(lines)
            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                if min_len <= len(text) <= max_len:
                    sentences.append(text)
            except:
                pass
        
        attempts += 1
    
    return sentences[:count]


# ============================================================================
# РАСШИРЕННЫЙ КОНТЕКСТ (3 предложения)
# ============================================================================

def get_extended_context(jsonl_path, position, window=1):
    """Получает контекст: предложение + соседние."""
    if not os.path.exists(jsonl_path):
        return None
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    if not lines or position >= len(lines):
        return None
    
    start = max(0, position - window)
    end = min(len(lines), position + window + 1)
    
    context = []
    for i in range(start, end):
        try:
            data = json.loads(lines[i])
            text = data.get('text', '').strip()
            if text:
                context.append(text)
        except:
            pass
    
    return ' '.join(context) if context else None


def test_with_extended_context(jsonl_path, model, tokenizer, num_examples=5):
    """Тест с расширенным контекстом (3 предложения)."""
    if os.path.isdir(jsonl_path):
        jsonl_files = glob.glob(f"{jsonl_path}/*.jsonl")
        if not jsonl_files:
            print("No JSONL files found!")
            return []
        jsonl_path = random.choice(jsonl_files)
    
    print(f"Using file: {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        all_lines = f.readlines()
    
    results = []
    attempts = 0
    
    while len(results) < num_examples and attempts < 500:
        pos = random.randint(0, len(all_lines) - 1)
        
        # Получаем контекст
        context_text = get_extended_context(jsonl_path, pos, window=1)
        if not context_text or len(context_text) < 50:
            attempts += 1
            continue
        
        # Получаем целевое предложение
        try:
            data = json.loads(all_lines[pos])
            target_sent = data.get('text', '').strip()
        except:
            attempts += 1
            continue
        
        # Маскируем токен в целевом предложении
        masked_target, true_token, _, _ = mask_single_token(target_sent, tokenizer)
        if masked_target is None:
            attempts += 1
            continue
        
        # Заменяем в контексте целевое предложение на замаскированное
        full_context = context_text.replace(target_sent, masked_target)
        
        predictions = predict_masked(full_context, model, tokenizer, top_k=10)
        if not predictions:
            attempts += 1
            continue
        
        results.append({
            'context': full_context[:300],
            'target_sentence': target_sent[:150],
            'masked_target': masked_target[:150],
            'true_token': true_token,
            'predictions': [p[0] for p in predictions[0]],
        })
        
        attempts += 1
    
    return results


# ============================================================================
# ТЕСТ НА СЛУЧАЙНЫХ ПРЕДЛОЖЕНИЯХ
# ============================================================================

def test_on_random_sentences(jsonl_path, model, tokenizer, num_examples=5, language=None, index=None, processed_dir=None):
    """Тест с маскированием токенов, опционально по языку."""
    
    # Выбор по языку
    if language and index and processed_dir:
        sentences = select_sentences_by_language(index, processed_dir, language, num_examples)
    else:
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
            if all_lines:
                selected = random.sample(all_lines, min(num_examples * 10, len(all_lines)))
                for line in selected:
                    data = json.loads(line)
                    text = data.get('text', '').strip()
                    if 30 < len(text) < 500:
                        sentences.append(text)
                        if len(sentences) >= num_examples:
                            break
    
    results = []
    for sent in sentences:
        masked_text, true_token, _, _ = mask_single_token(sent, tokenizer)
        if masked_text is None:
            continue
        
        predictions = predict_masked(masked_text, model, tokenizer, top_k=10)
        if not predictions:
            continue
        
        results.append({
            'original': sent,
            'masked': masked_text,
            'true_token': true_token,
            'predictions': [p[0] for p in predictions[0]],
        })
    
    return results


def test_multi_mask(jsonl_path, model, tokenizer, num_examples=5, num_masks=3):
    """Тест с множественным маскированием."""
    if os.path.isdir(jsonl_path):
        jsonl_files = glob.glob(f"{jsonl_path}/*.jsonl")
        if not jsonl_files:
            return []
        jsonl_path = random.choice(jsonl_files)
    
    print(f"Using file: {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        all_lines = f.readlines()
    
    results = []
    for _ in range(num_examples * 5):
        if len(results) >= num_examples:
            break
        
        line = random.choice(all_lines)
        data = json.loads(line)
        text = data.get('text', '').strip()
        
        if len(text) < 80:  # Нужен более длинный текст для нескольких масок
            continue
        
        masked_text, true_tokens, mask_positions, _ = mask_multiple_tokens(
            text, tokenizer, num_masks=num_masks
        )
        if masked_text is None:
            continue
        
        predictions = predict_masked(masked_text, model, tokenizer, top_k=5)
        if len(predictions) < len(true_tokens):
            continue
        
        results.append({
            'original': text[:300],
            'masked': masked_text[:300],
            'true_tokens': true_tokens,
            'predictions_per_mask': [[p[0] for p in pred] for pred in predictions],
            'correct': sum(1 for t, p in zip(true_tokens, predictions) if t in [x[0] for x in p]),
            'total': len(true_tokens),
        })
    
    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Тестирование BERT модели")
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer/multilingual/sp_42k.model')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['interactive', 'random', 'language', 'extended', 'multi', 'all'])
    parser.add_argument('--cleaned', type=str, default='data/processed', help='Директория с JSONL')
    parser.add_argument('--index', type=str, default='data/index/book_index.jsonl', help='Файл индекса')
    parser.add_argument('--num-examples', type=int, default=5)
    parser.add_argument('--language', type=str, default=None, 
                       choices=['ru', 'en', 'mixed'], help='Язык для теста')
    parser.add_argument('--num-masks', type=int, default=3, help='Количество масок для multi режима')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Loading model from {args.model}")
    model, tokenizer = load_model(args.model, args.tokenizer)
    print("Model loaded!")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print("=" * 80)
    
    # Загружаем индекс (если есть)
    index = load_book_index(args.index) if os.path.exists(args.index) else {}
    if index:
        print(f"Loaded index with {len(index)} books")
    
    if args.mode == 'interactive':
        print("\nИнтерактивный режим. Введите текст с [MASK] или 'exit'.\n")
        while True:
            text = input(">>> ")
            if text.lower() == 'exit':
                break
            predictions = predict_masked(text, model, tokenizer, top_k=10)
            if predictions:
                for i, mask_preds in enumerate(predictions):
                    print(f"\n[MASK {i+1}]:")
                    for word, prob in mask_preds[:5]:
                        print(f"  → {word} ({prob:.3f})")
            print()
    
    elif args.mode == 'random':
        lang_str = f" (language: {args.language})" if args.language else ""
        print(f"\nТест на случайных предложениях{lang_str}")
        
        results = test_on_random_sentences(
            args.cleaned, model, tokenizer, args.num_examples,
            language=args.language, index=index, processed_dir=args.cleaned
        )
        
        for i, r in enumerate(results, 1):
            in_top5 = r['true_token'] in r['predictions'][:5]
            in_top10 = r['true_token'] in r['predictions'][:10]
            print(f"\n--- {i} ---")
            print(f"Оригинал:  {r['original'][:200]}")
            print(f"Маска:     {r['masked'][:200]}")
            print(f"Токен:     [{r['true_token']}]")
            print(f"Топ-10:    {', '.join(r['predictions'][:10])}")
            print(f"Результат: {'✅' if in_top5 else '❌'} (топ-5: {in_top5}, топ-10: {in_top10})")
    
    elif args.mode == 'language':
        if not index:
            print("Index file required for language mode. Use --index")
            return
        
        for lang in ['ru', 'en', 'mixed']:
            print(f"\n{'='*60}")
            print(f"Language: {lang}")
            print(f"{'='*60}")
            
            results = test_on_random_sentences(
                args.cleaned, model, tokenizer, args.num_examples,
                language=lang, index=index, processed_dir=args.cleaned
            )
            
            correct_top5 = sum(1 for r in results if r['true_token'] in r['predictions'][:5])
            print(f"Top-5 accuracy: {correct_top5}/{len(results)}")
    
    elif args.mode == 'extended':
        print(f"\nТест с расширенным контекстом (3 предложения)")
        
        results = test_with_extended_context(
            args.cleaned, model, tokenizer, args.num_examples
        )
        
        for i, r in enumerate(results, 1):
            in_top5 = r['true_token'] in r['predictions'][:5]
            print(f"\n--- {i} ---")
            print(f"Контекст:   {r['context'][:250]}...")
            print(f"Цель:       {r['target_sentence'][:150]}")
            print(f"Маска:      {r['masked_target'][:150]}")
            print(f"Токен:      [{r['true_token']}]")
            print(f"Топ-10:     {', '.join(r['predictions'][:10])}")
            print(f"Результат:  {'✅' if in_top5 else '❌'}")
    
    elif args.mode == 'multi':
        print(f"\nТест с множественным маскированием ({args.num_masks} масок)")
        
        results = test_multi_mask(
            args.cleaned, model, tokenizer, args.num_examples, args.num_masks
        )
        
        for i, r in enumerate(results, 1):
            print(f"\n--- {i} ---")
            print(f"Оригинал:   {r['original'][:200]}...")
            print(f"Маска:      {r['masked'][:200]}...")
            print(f"Токены:     {r['true_tokens']}")
            for j, preds in enumerate(r['predictions_per_mask']):
                print(f"  [MASK {j+1}]: {', '.join(preds[:5])}")
            print(f"Точно:      {r['correct']}/{r['total']}")
    
    elif args.mode == 'all':
        # MLM тест
        print("\n" + "=" * 80)
        print("1. MLM тест")
        print("=" * 80)
        
        test_texts = [
            "Москва — [MASK] России.",
            "The capital of France is [MASK].",
            "Вода состоит из водорода и [MASK].",
            "Я [MASK] в магазин.",
            "Attention is all you [MASK].",
        ]
        
        for text in test_texts:
            predictions = predict_masked(text, model, tokenizer, top_k=5)
            print(f"\n{text}")
            for mask_preds in predictions:
                for word, prob in mask_preds[:5]:
                    print(f"  → {word} ({prob:.3f})")
        
        # По языкам
        if index:
            print("\n" + "=" * 80)
            print("2. Тест по языкам")
            print("=" * 80)
            
            for lang in ['ru', 'en', 'mixed']:
                results = test_on_random_sentences(
                    args.cleaned, model, tokenizer, args.num_examples,
                    language=lang, index=index, processed_dir=args.cleaned
                )
                correct = sum(1 for r in results if r['true_token'] in r['predictions'][:5])
                print(f"\n{lang}: {correct}/{len(results)} в топ-5")
        
        # Расширенный контекст
        print("\n" + "=" * 80)
        print("3. Расширенный контекст")
        print("=" * 80)
        
        results = test_with_extended_context(args.cleaned, model, tokenizer, 3)
        for r in results:
            in_top5 = r['true_token'] in r['predictions'][:5]
            print(f"\nТокен: [{r['true_token']}] → {'✅' if in_top5 else '❌'}")
            print(f"Топ-5: {', '.join(r['predictions'][:5])}")


if __name__ == "__main__":
    main()
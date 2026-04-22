#!/usr/bin/env python3
"""
Тестирование обученной BERT модели.
Показывает, чему научилась модель на разных этапах обучения.
"""

import torch
import json
import random
from transformers import BertForMaskedLM
from simple_sp_tokenizer import SimpleSPTokenizer
import numpy as np


def load_model(model_path, tokenizer_path):
    """Загружает модель и токенизатор."""
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path)
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer


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
    # Добавляем [MASK] в конец
    text_with_mask = text + " [MASK]"
    return predict_masked(text_with_mask, model, tokenizer, top_k)


def fill_cloze(text, model, tokenizer):
    """
    Заполняет все [MASK] в тексте.
    """
    encoded = tokenizer(text, max_length=512, return_tensors='pt')
    
    if torch.cuda.is_available():
        encoded = {k: v.cuda() for k, v in encoded.items()}
    
    mask_positions = (encoded['input_ids'][0] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
    
    if len(mask_positions) == 0:
        return text
    
    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits[0]
    
    result_tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0].tolist())
    
    for pos in mask_positions:
        token_id = torch.argmax(logits[pos]).item()
        result_tokens[pos] = tokenizer.convert_ids_to_tokens(token_id)
    
    # Декодируем обратно
    result_text = ''.join(result_tokens).replace('▁', ' ').replace('[CLS]', '').replace('[SEP]', '').strip()
    return result_text


def test_analogy(word_a, word_b, word_c, model, tokenizer, top_k=5):
    """
    Тестирует аналогию: word_a - word_b + word_c = ?
    
    Example:
        "Москва" - "Россия" + "Париж" = "Франция"
    """
    # Получаем эмбеддинги слов
    def get_embedding(word):
        ids = tokenizer.encode(word, max_length=512)
        ids = [i for i in ids if i not in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]]
        
        if torch.cuda.is_available():
            ids_tensor = torch.tensor([ids]).cuda()
        else:
            ids_tensor = torch.tensor([ids])
        
        with torch.no_grad():
            outputs = model.bert.embeddings.word_embeddings(ids_tensor)
            return outputs.mean(dim=0)
    
    emb_a = get_embedding(word_a)
    emb_b = get_embedding(word_b)
    emb_c = get_embedding(word_c)
    
    # Вычисляем целевой эмбеддинг
    target_emb = emb_a - emb_b + emb_c
    
    # Ищем ближайшие слова в словаре
    word_embeddings = model.bert.embeddings.word_embeddings.weight
    
    # Косинусное сходство
    similarity = torch.cosine_similarity(target_emb.unsqueeze(0), word_embeddings)
    
    # Исключаем входные слова
    exclude_ids = set()
    for word in [word_a, word_b, word_c]:
        exclude_ids.add(tokenizer.convert_tokens_to_ids(word))
    
    similarity[list(exclude_ids)] = -float('inf')
    
    top_ids = torch.topk(similarity, top_k).indices.tolist()
    
    results = []
    for tid in top_ids:
        token = tokenizer.convert_ids_to_tokens(tid)
        token = token.replace('▁', '').strip()
        results.append(token)
    
    return results


def test_on_random_sentences(jsonl_file, model, tokenizer, num_examples=5):
    """
    Берёт случайные предложения из cleaned датасета и тестирует MLM.
    """
    sentences = []
    
    # Читаем случайные предложения
    with open(jsonl_file, 'r') as f:
        all_lines = f.readlines()
        selected = random.sample(all_lines, min(num_examples * 10, len(all_lines)))
        
        for line in selected:
            data = json.loads(line)
            text = data.get('text', '').strip()
            if len(text) > 20 and len(text) < 200:
                sentences.append(text)
                if len(sentences) >= num_examples:
                    break
    
    results = []
    for sent in sentences:
        # Случайно маскируем одно слово
        words = sent.split()
        if len(words) < 3:
            continue
        
        mask_idx = random.randint(0, len(words) - 1)
        masked_word = words[mask_idx]
        words[mask_idx] = '[MASK]'
        masked_text = ' '.join(words)
        
        predictions = predict_masked(masked_text, model, tokenizer, top_k=10)
        
        results.append({
            'original': sent,
            'masked': masked_text,
            'true_word': masked_word,
            'predictions': [p[0] for p in predictions]
        })
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Путь к модели')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer/final/sp_100k.model')
    parser.add_argument('--mode', type=str, default='interactive',
                       choices=['interactive', 'random', 'analogy', 'all'])
    parser.add_argument('--cleaned', type=str, default='data/cleaned',
                       help='Директория с JSONL для случайных примеров')
    args = parser.parse_args()
    
    print("=" * 80)
    print(f"Loading model from {args.model}")
    model, tokenizer = load_model(args.model, args.tokenizer)
    print("Model loaded!")
    print("=" * 80)
    
    if args.mode == 'interactive':
        print("\nИнтерактивный режим. Введите текст с [MASK] или 'exit' для выхода.\n")
        print("Примеры:")
        print("  Москва — [MASK] России")
        print("  The capital of France is [MASK]")
        print("  α = 0.[MASK]")
        print()
        
        while True:
            text = input(">>> ")
            if text.lower() == 'exit':
                break
            
            if '[MASK]' in text:
                predictions = predict_masked(text, model, tokenizer, top_k=5)
                print(f"\nПредсказания для [MASK]:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {word} ({prob:.3f})")
            else:
                predictions = predict_next_word(text, model, tokenizer, top_k=5)
                print(f"\nСледующее слово:")
                for i, (word, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {word} ({prob:.3f})")
            print()
    
    elif args.mode == 'random':
        print(f"\nТестирование на случайных предложениях из {args.cleaned}")
        
        # Ищем первый попавшийся JSONL файл
        import glob
        jsonl_files = glob.glob(f"{args.cleaned}/*.jsonl")
        if not jsonl_files:
            print("No JSONL files found!")
            return
        
        test_file = random.choice(jsonl_files)
        print(f"Using file: {test_file}")
        
        results = test_on_random_sentences(test_file, model, tokenizer, num_examples=5)
        
        for i, r in enumerate(results, 1):
            print(f"\n--- Пример {i} ---")
            print(f"Оригинал:    {r['original']}")
            print(f"С маской:    {r['masked']}")
            print(f"Правильно:   {r['true_word']}")
            print(f"Предсказано: {', '.join(r['predictions'])}")
            print(f"{'✅' if r['true_word'] in r['predictions'] else '❌'}")
    
    elif args.mode == 'analogy':
        print("\nТестирование аналогий:")
        
        analogies = [
            ("Москва", "Россия", "Париж"),
            ("король", "мужчина", "королева"),
            ("ходить", "шёл", "бегать"),
        ]
        
        for a, b, c in analogies:
            result = test_analogy(a, b, c, model, tokenizer, top_k=3)
            print(f"\n{a} - {b} + {c} = ?")
            print(f"  Предсказания: {result}")
    
    elif args.mode == 'all':
        print("\n" + "=" * 80)
        print("1. MLM тест")
        print("=" * 80)
        
        test_texts = [
            "Москва — [MASK] России",
            "Столица России [MASK]",
            "The capital of France is [MASK]",
            "Внимание — это [MASK], что вам нужно",
            "α = 0.[MASK]",
            "2 + 2 = [MASK]",
            "Я [MASK] в магазин",           # ожидаем: пошёл, иду
            "Он [MASK] книгу",              # ожидаем: читает, взял
            "1, 2, [MASK], 4",              # ожидаем: 3
            "Красный, жёлтый, [MASK]",     # ожидаем: синий, жёлтый
            "Владимир Владимирович [MASK] ",
            "Президент США [MASK] ",
            "Здравствуйте, как [MASK]?",
        ]        
        
        for text in test_texts:
            predictions = predict_masked(text, model, tokenizer, top_k=10)
            print(f"\n{text}")
            for word, prob in predictions:
                print(f"  → {word} ({prob:.3f})")
        
        print("\n" + "=" * 80)
        print("2. Случайные предложения из корпуса")
        print("=" * 80)
        
        import glob
        jsonl_files = glob.glob(f"{args.cleaned}/*.jsonl")
        if jsonl_files:
            test_file = random.choice(jsonl_files)
            results = test_on_random_sentences(test_file, model, tokenizer, num_examples=3)
            
            for r in results:
                print(f"\nОригинал: {r['original']}")
                print(f"Маска:    {r['masked']}")
                print(f"Правда:   {r['true_word']}")
                print(f"Топ-3:    {r['predictions']}")
                print(f"Результат: {'✅' if r['true_word'] in r['predictions'] else '❌'}")


if __name__ == "__main__":
    main()
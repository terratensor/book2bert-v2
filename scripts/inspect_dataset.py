#!/usr/bin/env python3
"""Инспектирует созданный датасет для BERT."""

import json
from pathlib import Path
from datasets import load_from_disk
from simple_sp_tokenizer import SimpleSPTokenizer


def inspect_dataset(dataset_path: str, tokenizer_path: str, num_examples: int = 5):
    """Показывает содержимое датасета."""
    
    print("=" * 80)
    print(f"Inspecting dataset: {dataset_path}")
    print("=" * 80)
    
    # Загружаем датасет
    dataset = load_from_disk(dataset_path)
    print(f"\nDataset splits: {list(dataset.keys())}")
    
    for split_name in dataset.keys():
        print(f"\n  {split_name}: {len(dataset[split_name]):,} examples")
    
    # Загружаем токенизатор
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    
    # Показываем примеры из train
    print("\n" + "=" * 80)
    print("TRAIN EXAMPLES")
    print("=" * 80)
    
    for i, example in enumerate(dataset['train'].select(range(min(num_examples, len(dataset['train']))))):
        print(f"\n--- Example {i+1} ---")
        
        input_ids = example['input_ids']
        attention_mask = example['attention_mask']
        
        # Убираем паддинг для отображения
        actual_len = sum(attention_mask)
        actual_ids = input_ids[:actual_len]
        padding_ids = input_ids[actual_len:]
        
        print(f"Input IDs length: {len(input_ids)} (actual: {actual_len}, padding: {len(padding_ids)})")
        print(f"Attention mask: {attention_mask[:50]}{'...' if len(attention_mask) > 50 else ''}")
        
        # Декодируем
        decoded = tokenizer.decode(actual_ids, skip_special_tokens=False)
        print(f"\nDecoded text (with special tokens):\n{decoded[:500]}...")
        
        # Декодируем без специальных токенов
        decoded_clean = tokenizer.decode(actual_ids, skip_special_tokens=True)
        print(f"\nDecoded text (clean):\n{decoded_clean[:500]}...")
        
        # Показываем токены
        tokens = tokenizer.convert_ids_to_tokens(actual_ids[:30])
        print(f"\nFirst 30 tokens: {tokens}")
        
        # Проверяем специальные токены
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id
        
        print(f"\nSpecial tokens check:")
        print(f"  CLS token ID: {cls_id}")
        print(f"  SEP token ID: {sep_id}")
        print(f"  PAD token ID: {pad_id}")
        print(f"  First token: {actual_ids[0]} (should be CLS={cls_id}) -> {'✅' if actual_ids[0] == cls_id else '❌'}")
        print(f"  Last actual token: {actual_ids[-1]} (should be SEP={sep_id}) -> {'✅' if actual_ids[-1] == sep_id else '❌'}")
        
        # Проверяем паддинг
        if len(padding_ids) > 0:
            all_pad = all(pid == pad_id for pid in padding_ids)
            print(f"  Padding tokens: {padding_ids[:10]}... (all PAD={pad_id}) -> {'✅' if all_pad else '❌'}")
        else:
            print(f"  Padding tokens: none (sequence exactly {actual_len} tokens)")
        
        # Проверяем нет ли дублирования SEP
        sep_count = actual_ids.count(sep_id)
        sep_positions = [j for j, tid in enumerate(actual_ids) if tid == sep_id]
        print(f"  SEP count: {sep_count} (should be 1 at the end) -> {'✅' if sep_count == 1 else '❌'}")
        print(f"  SEP positions: {sep_positions}")
        
        # Проверяем CLS только в начале
        cls_positions = [j for j, tid in enumerate(actual_ids) if tid == cls_id]
        print(f"  CLS positions: {cls_positions} (should be [0]) -> {'✅' if cls_positions == [0] else '❌'}")


def check_batch_format(dataset_path: str, tokenizer_path: str):
    """Проверяет формат батча для обучения."""
    
    print("\n" + "=" * 80)
    print("BATCH FORMAT CHECK")
    print("=" * 80)
    
    dataset = load_from_disk(dataset_path)
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    pad_id = tokenizer.pad_token_id
    
    # Берем первые 4 примера как "батч"
    batch = dataset['train'][:4]
    
    print(f"\nBatch keys: {batch.keys()}")
    print(f"input_ids shape: {len(batch['input_ids'])} x {len(batch['input_ids'][0])}")
    print(f"attention_mask shape: {len(batch['attention_mask'])} x {len(batch['attention_mask'][0])}")
    
    all_valid = True
    
    for i in range(len(batch['input_ids'])):
        ids = batch['input_ids'][i]
        mask = batch['attention_mask'][i]
        
        # Длина по маске
        actual_len = sum(mask)
        
        # Считаем не-паддинговые токены (НЕ равные pad_id)
        non_pad = sum(1 for tid in ids if tid != pad_id)
        
        # Проверяем что паддинг в конце - все pad_id
        padding_ids = ids[actual_len:]
        all_pad = all(pid == pad_id for pid in padding_ids) if len(padding_ids) > 0 else True
        
        # Проверяем что маска корректна: 1 для actual, 0 для padding
        mask_correct = all(m == 1 for m in mask[:actual_len]) and all(m == 0 for m in mask[actual_len:])
        
        valid = (actual_len == non_pad) and all_pad and mask_correct
        
        print(f"\n  Example {i+1}:")
        print(f"    Total length: {len(ids)}")
        print(f"    Actual length (by mask): {actual_len}")
        print(f"    Non-pad tokens (by ID): {non_pad}")
        print(f"    Padding all PAD ({pad_id}): {'✅' if all_pad else '❌'}")
        print(f"    Mask correct: {'✅' if mask_correct else '❌'}")
        print(f"    Overall valid: {'✅' if valid else '❌'}")
        
        if not valid:
            all_valid = False
            print(f"    DEBUG: first padding ids: {padding_ids[:5]}")
            print(f"    DEBUG: mask at boundary: {mask[actual_len-2:actual_len+2]}")
    
    print(f"\n  All examples valid: {'✅' if all_valid else '❌'}")


def show_token_distribution(dataset_path: str):
    """Показывает распределение длин последовательностей."""
    
    print("\n" + "=" * 80)
    print("SEQUENCE LENGTH DISTRIBUTION")
    print("=" * 80)
    
    dataset = load_from_disk(dataset_path)
    
    for split_name in dataset.keys():
        lengths = []
        for example in dataset[split_name]:
            actual_len = sum(example['attention_mask'])
            lengths.append(actual_len)
        
        if lengths:
            import numpy as np
            print(f"\n  {split_name}:")
            print(f"    Count: {len(lengths):,}")
            print(f"    Min length: {min(lengths)}")
            print(f"    Max length: {max(lengths)}")
            print(f"    Mean length: {np.mean(lengths):.1f}")
            print(f"    Median length: {np.median(lengths):.0f}")
            print(f"    Std dev: {np.std(lengths):.1f}")
            
            percentiles = [50, 75, 90, 95, 99]
            for p in percentiles:
                val = np.percentile(lengths, p)
                print(f"    {p}th percentile: {val:.0f}")


def check_special_tokens_vocab(tokenizer_path: str):
    """Проверяет наличие специальных токенов в словаре."""
    
    print("\n" + "=" * 80)
    print("SPECIAL TOKENS IN VOCABULARY")
    print("=" * 80)
    
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    
    special_tokens = [
        ("[PAD]", tokenizer.pad_token_id),
        ("[UNK]", tokenizer.unk_token_id),
        ("[CLS]", tokenizer.cls_token_id),
        ("[SEP]", tokenizer.sep_token_id),
        ("[MASK]", tokenizer.mask_token_id),
    ]
    
    print("\n  Special tokens:")
    for name, tid in special_tokens:
        token_str = tokenizer.sp.IdToPiece(tid) if tid < tokenizer.vocab_size else "NOT FOUND"
        print(f"    {name}: ID={tid}, piece='{token_str}'")
    
    # Проверяем что все ID в пределах vocab_size
    all_valid = all(tid < tokenizer.vocab_size for _, tid in special_tokens)
    print(f"\n  All special tokens in vocab: {'✅' if all_valid else '❌'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/bert/bert_dataset_128")
    parser.add_argument("--tokenizer", default="models/tokenizer/final/sp_100k.model")
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()
    
    # Проверяем специальные токены
    check_special_tokens_vocab(args.tokenizer)
    
    # Инспектируем датасет
    inspect_dataset(args.dataset, args.tokenizer, args.examples)
    check_batch_format(args.dataset, args.tokenizer)
    show_token_distribution(args.dataset)
    
    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)
#!/usr/bin/env python3
"""Потоковая подготовка датасета из JSONL с сохранением контекста книг."""

import json
import random
from pathlib import Path
from typing import Iterator, List, Optional
import glob
import tempfile

from simple_sp_tokenizer import SimpleSPTokenizer
from datasets import Dataset, DatasetDict, Features, Value


def iter_chunks_from_book(
    book_file: Path,
    tokenizer: SimpleSPTokenizer,
    max_length: int,
    overlap: int = 0,
) -> Iterator[str]:
    """Генератор: создает чанки из одной книги по одному."""
    sentences = []
    
    with open(book_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                if text:
                    sentences.append(text)
            except:
                continue
    
    if not sentences:
        return
    
    current_chunk = []
    current_length = 0
    
    for text in sentences:
        ids = tokenizer.sp.EncodeAsIds(text)
        sent_len = len(ids)
        
        if sent_len > max_length - 2:
            continue
        
        if current_length + sent_len > max_length - 2:
            if current_chunk:
                yield ' '.join(current_chunk)
            
            if overlap > 0:
                overlap_chunk = []
                overlap_len = 0
                for s in reversed(current_chunk):
                    s_len = len(tokenizer.sp.EncodeAsIds(s))
                    if overlap_len + s_len > overlap:
                        break
                    overlap_chunk.insert(0, s)
                    overlap_len += s_len
                current_chunk = overlap_chunk
                current_length = overlap_len
            else:
                current_chunk = []
                current_length = 0
        
        current_chunk.append(text)
        current_length += sent_len
    
    if current_chunk:
        yield ' '.join(current_chunk)


def iter_all_chunks(
    book_files: List[str],
    tokenizer: SimpleSPTokenizer,
    max_length: int,
    overlap: int = 0,
) -> Iterator[str]:
    """Генератор: обходит все книги и отдает чанки по одному."""
    for i, book_file in enumerate(book_files):
        if (i + 1) % 100 == 0:
            print(f"    Processing book {i+1}/{len(book_files)}")
        
        for chunk in iter_chunks_from_book(
            Path(book_file), tokenizer, max_length, overlap
        ):
            yield chunk


def create_dataset_from_chunks(
    book_files: List[str],
    tokenizer: SimpleSPTokenizer,
    max_length: int,
    overlap: int = 0,
) -> Optional[Dataset]:
    """
    Создает Dataset из чанков.
    Если чанков нет — возвращает None.
    """
    
    # Сначала собираем чанки в список (для небольших сплитов это ок)
    # Для большого train используем другой подход
    chunks = []
    for i, book_file in enumerate(book_files):
        if (i + 1) % 1000 == 0:
            print(f"    Processing book {i+1}/{len(book_files)}, {len(chunks):,} chunks")
        
        for chunk in iter_chunks_from_book(
            Path(book_file), tokenizer, max_length, overlap
        ):
            chunks.append(chunk)
    
    if not chunks:
        print("    WARNING: No chunks generated!")
        return None
    
    print(f"    Total chunks: {len(chunks):,}")
    
    # Создаем Dataset
    dataset = Dataset.from_dict({'text': chunks})
    return dataset


def prepare_dataset_streaming(
    cleaned_dir: str,
    tokenizer_path: str,
    output_dir: str,
    max_lengths: List[int] = [128, 256, 512],
    overlap: int = 50,
    val_split: float = 0.025,
    test_split: float = 0.025,
    max_books: int = None,
):
    """
    Создание датасета с правильным разбиением.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем токенизатор
    print("Loading tokenizer...")
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    
    # Находим все JSONL файлы
    print(f"Searching for JSONL files in {cleaned_dir}...")
    all_books = sorted(glob.glob(f"{cleaned_dir}/*.jsonl"))
    print(f"Found {len(all_books):,} books")
    
    if max_books:
        all_books = all_books[:max_books]
        print(f"Limited to {len(all_books)} books")
    
    # Перемешиваем книги
    random.seed(42)
    random.shuffle(all_books)
    
    # Разбиваем на train/val/test по пропорциям
    n = len(all_books)
    test_n = int(n * test_split)
    val_n = int(n * val_split)
    train_n = n - val_n - test_n
    
    # Если книг мало — адаптируем разбиение
    if n < 10:
        train_books = all_books[:max(1, n-2)]
        val_books = all_books[len(train_books):len(train_books)+1] if n > len(train_books) else []
        test_books = all_books[len(train_books)+len(val_books):]
    else:
        train_books = all_books[:train_n]
        val_books = all_books[train_n:train_n + val_n]
        test_books = all_books[train_n + val_n:]
    
    print(f"Train books: {len(train_books):,}")
    print(f"Val books: {len(val_books):,}")
    print(f"Test books: {len(test_books):,}")
    
    splits = {
        'train': train_books,
        'validation': val_books,
        'test': test_books,
    }
    
    # Для каждой длины создаем датасет
    for max_len in max_lengths:
        print(f"\n{'='*60}")
        print(f"Preparing dataset for max_length={max_len}")
        print('='*60)
        
        datasets = {}
        
        for split_name, book_files in splits.items():
            if not book_files:
                print(f"\n  Skipping {split_name} (no books)")
                continue
                
            print(f"\n  Processing {split_name}...")
            
            dataset = create_dataset_from_chunks(
                book_files, tokenizer, max_len,
                overlap if max_len >= 256 else 0
            )
            
            if dataset is None:
                print(f"    ERROR: Could not create dataset for {split_name}")
                continue
            
            # Токенизируем
            print(f"    Tokenizing...")
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    max_length=max_len,
                    padding='max_length',
                    truncation=True,
                )
            
            dataset = dataset.map(
                tokenize_function,
                batched=True,
                batch_size=1000,
                remove_columns=['text'],
                desc=f"    Tokenizing {split_name}",
            )
            
            datasets[split_name] = dataset
        
        if not datasets:
            print("  ERROR: No datasets created!")
            continue
        
        # Сохраняем
        dataset_dict = DatasetDict(datasets)
        save_path = output_path / f"bert_dataset_{max_len}"
        dataset_dict.save_to_disk(str(save_path))
        print(f"\n  Saved to {save_path}")
        
        # Статистика
        for name, ds in datasets.items():
            print(f"    {name}: {len(ds):,} examples")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", default="data/cleaned")
    parser.add_argument("--tokenizer", default="models/tokenizer/final/sp_100k.model")
    parser.add_argument("--output", default="data/bert")
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--val-split", type=float, default=0.025)
    parser.add_argument("--test-split", type=float, default=0.025)
    args = parser.parse_args()
    
    prepare_dataset_streaming(
        cleaned_dir=args.cleaned,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        max_lengths=args.lengths,
        max_books=args.max_books,
        val_split=args.val_split,
        test_split=args.test_split,
    )
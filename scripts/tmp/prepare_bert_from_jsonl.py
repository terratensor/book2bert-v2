#!/usr/bin/env python3
"""Потоковая подготовка датасета с немедленным сохранением на диск."""

import json
import random
from pathlib import Path
from typing import Iterator, List
import glob
import tempfile
import os

from simple_sp_tokenizer import SimpleSPTokenizer
from datasets import Dataset, DatasetDict


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


def write_chunks_to_temp(
    book_files: List[str],
    tokenizer: SimpleSPTokenizer,
    max_length: int,
    overlap: int,
    temp_file: Path,
):
    """
    Записывает все чанки во временный файл ПОТОКОВО.
    НЕ хранит чанки в памяти!
    """
    with open(temp_file, 'w', encoding='utf-8') as f:
        chunk_count = 0
        for i, book_file in enumerate(book_files):
            if (i + 1) % 100 == 0:
                print(f"    Processed {i+1}/{len(book_files)} books, {chunk_count:,} chunks written")
            
            for chunk in iter_chunks_from_book(
                Path(book_file), tokenizer, max_length, overlap
            ):
                f.write(chunk + '\n')
                chunk_count += 1
        
        print(f"    Total chunks written: {chunk_count:,}")
        return chunk_count


def create_dataset_from_temp(
    temp_file: Path,
    tokenizer: SimpleSPTokenizer,
    max_length: int,
) -> Dataset:
    """
    Создает Dataset из временного файла.
    datasets умеет читать файл потоково, не загружая всё в память.
    """
    # Загружаем тексты из файла (datasets делает это эффективно)
    dataset = Dataset.from_text(str(temp_file))
    
    # Токенизируем
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            max_length=max_length,
            padding='max_length',
            truncation=True,
        )
    
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=['text'],
        desc="    Tokenizing",
    )
    
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
    keep_temp: bool = False,
):
    """Потоковое создание датасета с записью на диск."""
    
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
    
    # Разбиваем на train/val/test
    n = len(all_books)
    test_n = int(n * test_split)
    val_n = int(n * val_split)
    train_n = n - val_n - test_n
    
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
    
    # Создаем временную директорию
    temp_dir = Path(tempfile.mkdtemp(dir=output_path, prefix="temp_chunks_"))
    print(f"Temp directory: {temp_dir}")
    
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
            
            # Временный файл для этого сплита
            temp_file = temp_dir / f"{split_name}_{max_len}.txt"
            
            # Записываем чанки во временный файл (ПОТОКОВО!)
            print(f"    Writing chunks to {temp_file}...")
            chunk_count = write_chunks_to_temp(
                book_files, tokenizer, max_len,
                overlap if max_len >= 256 else 0,
                temp_file,
            )
            
            if chunk_count == 0:
                print(f"    WARNING: No chunks generated for {split_name}")
                continue
            
            # Создаем Dataset из временного файла
            print(f"    Creating dataset from temp file...")
            dataset = create_dataset_from_temp(temp_file, tokenizer, max_len)
            
            datasets[split_name] = dataset
            
            # Удаляем временный файл (если не нужно хранить)
            if not keep_temp:
                temp_file.unlink()
                print(f"    Temp file deleted")
        
        if not datasets:
            print("  ERROR: No datasets created!")
            continue
        
        # Сохраняем датасет
        dataset_dict = DatasetDict(datasets)
        save_path = output_path / f"bert_dataset_{max_len}"
        dataset_dict.save_to_disk(str(save_path))
        print(f"\n  Saved to {save_path}")
        
        # Статистика
        for name, ds in datasets.items():
            print(f"    {name}: {len(ds):,} examples")
    
    # Удаляем временную директорию
    if not keep_temp:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"\nTemp directory deleted: {temp_dir}")
    
    print("\n" + "=" * 60)
    print("All datasets prepared!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned", default="data/cleaned")
    parser.add_argument("--tokenizer", default="models/tokenizer/final/sp_100k.model")
    parser.add_argument("--output", default="data/bert")
    parser.add_argument("--max-books", type=int, default=None)
    parser.add_argument("--lengths", nargs="+", type=int, default=[128])
    parser.add_argument("--val-split", type=float, default=0.025)
    parser.add_argument("--test-split", type=float, default=0.025)
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary chunk files")
    args = parser.parse_args()
    
    prepare_dataset_streaming(
        cleaned_dir=args.cleaned,
        tokenizer_path=args.tokenizer,
        output_dir=args.output,
        max_lengths=args.lengths,
        max_books=args.max_books,
        val_split=args.val_split,
        test_split=args.test_split,
        keep_temp=args.keep_temp,
    )
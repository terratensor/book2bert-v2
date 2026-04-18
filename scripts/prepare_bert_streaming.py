#!/usr/bin/env python3
"""Параллельная подготовка датасета для BERT."""

import json
import random
from pathlib import Path
from typing import Iterator, List
import glob
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from simple_sp_tokenizer import SimpleSPTokenizer
from datasets import Dataset, DatasetDict


def iter_chunks_from_book(
    book_file: Path,
    tokenizer: SimpleSPTokenizer,
    max_length: int,
) -> Iterator[str]:
    """Генератор чанков из одной книги (без overlap)."""
    
    sentences = []
    with open(book_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                text = text.replace('\n', ' ').replace('\r', ' ')
                text = ' '.join(text.split())
                if text:
                    sentences.append(text)
            except Exception:
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
            current_chunk = [text]
            current_length = sent_len
        else:
            current_chunk.append(text)
            current_length += sent_len
    
    if current_chunk:
        yield ' '.join(current_chunk)


def process_single_book(args):
    """Обрабатывает одну книгу."""
    book_file, tokenizer_path, max_length = args
    
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    
    chunks = []
    for chunk in iter_chunks_from_book(Path(book_file), tokenizer, max_length):
        chunks.append(chunk)
    
    return chunks


def write_chunks_parallel(
    book_files: List[str],
    tokenizer_path: str,
    max_length: int,
    temp_file: Path,
    num_workers: int = 32,
) -> int:
    """Параллельная обработка книг."""
    
    chunk_count = 0
    args_list = [(f, tokenizer_path, max_length) for f in book_files]
    
    with open(temp_file, 'w', encoding='utf-8') as f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_single_book, args): args[0] 
                      for args in args_list}
            
            for i, future in enumerate(as_completed(futures)):
                book_file = futures[future]
                try:
                    chunks = future.result()
                    for chunk in chunks:
                        f.write(chunk + '\n')
                        chunk_count += 1
                    
                    if (i + 1) % 1000 == 0:
                        print(f"    Processed {i+1}/{len(book_files)} books, {chunk_count:,} chunks")
                        
                except Exception as e:
                    print(f"    ERROR {book_file}: {e}")
    
    print(f"    Total chunks written: {chunk_count:,}")
    return chunk_count


def create_dataset_from_temp(
    temp_file: Path,
    tokenizer: SimpleSPTokenizer,
    max_length: int,
) -> Dataset:
    """Создает Dataset из временного файла."""
    
    dataset = Dataset.from_text(str(temp_file))
    
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
    max_lengths: List[int] = [128],
    val_split: float = 0.025,
    test_split: float = 0.025,
    max_books: int = None,
    keep_temp: bool = False,
    num_workers: int = 32,
):
    """Параллельная подготовка датасета."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    print(f"Searching for JSONL files in {cleaned_dir}...")
    all_books = sorted(glob.glob(f"{cleaned_dir}/*.jsonl"))
    print(f"Found {len(all_books):,} books")
    
    if max_books:
        all_books = all_books[:max_books]
        print(f"Limited to {len(all_books)} books")
    
    random.seed(42)
    random.shuffle(all_books)
    
    n = len(all_books)
    test_n = int(n * test_split)
    val_n = int(n * val_split)
    
    if n < 10:
        train_books = all_books[:max(1, n-2)]
        val_books = all_books[len(train_books):len(train_books)+1] if n > len(train_books) else []
        test_books = all_books[len(train_books)+len(val_books):]
    else:
        train_books = all_books[:n - val_n - test_n]
        val_books = all_books[n - val_n - test_n:n - test_n]
        test_books = all_books[n - test_n:]
    
    print(f"Train books: {len(train_books):,}")
    print(f"Val books: {len(val_books):,}")
    print(f"Test books: {len(test_books):,}")
    
    splits = {'train': train_books, 'validation': val_books, 'test': test_books}
    
    temp_dir = Path(tempfile.mkdtemp(dir=output_path, prefix="temp_chunks_"))
    print(f"Temp directory: {temp_dir}")
    
    for max_len in max_lengths:
        print(f"\n{'='*60}")
        print(f"Preparing dataset for max_length={max_len}")
        print('='*60)
        
        datasets = {}
        
        for split_name, book_files in splits.items():
            if not book_files:
                continue
            
            print(f"\n  Processing {split_name} ({len(book_files):,} books)...")
            
            temp_file = temp_dir / f"{split_name}_{max_len}.txt"
            
            print(f"    Writing chunks in parallel ({num_workers} workers)...")
            chunk_count = write_chunks_parallel(
                book_files, 
                tokenizer_path, 
                max_len,
                temp_file, 
                num_workers
            )
            if chunk_count == 0:
                print(f"    WARNING: No chunks generated")
                continue
            
            print(f"    Creating dataset from temp file...")
            dataset = create_dataset_from_temp(temp_file, tokenizer, max_len)
            datasets[split_name] = dataset
            
            if not keep_temp:
                temp_file.unlink()
        
        if datasets:
            dataset_dict = DatasetDict(datasets)
            save_path = output_path / f"bert_dataset_{max_len}"
            dataset_dict.save_to_disk(str(save_path))
            print(f"\n  Saved to {save_path}")
            for name, ds in datasets.items():
                print(f"    {name}: {len(ds):,} examples")
    
    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)
    
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
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--workers", type=int, default=32)
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
        num_workers=args.workers,
    )
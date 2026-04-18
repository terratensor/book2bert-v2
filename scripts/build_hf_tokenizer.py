#!/usr/bin/env python3
"""Создает HuggingFace совместимый токенизатор из SentencePiece модели."""

import json
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm


def build_tokenizer_from_sp(sp_model_path: str, output_dir: str):
    """Создает токенизатор из SentencePiece модели."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем SentencePiece
    sp = spm.SentencePieceProcessor()
    sp.Load(sp_model_path)
    
    # Получаем словарь
    vocab = {}
    for i in range(sp.GetPieceSize()):
        piece = sp.IdToPiece(i)
        # Очищаем от мета-символов SentencePiece (▁ для пробела)
        clean_piece = piece.replace('▁', '')
        if clean_piece:
            vocab[piece] = i
    
    # Сохраняем vocab.json
    vocab_path = output_path / "vocab.json"
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    # Создаем tokenizer_config.json
    config = {
        "model_max_length": 512,
        "do_lower_case": False,
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
        "mask_token": "[MASK]",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": len(vocab),
        "name_or_path": "book2bert-tokenizer",
        "special_tokens_map_file": "special_tokens_map.json",
    }
    
    config_path = output_path / "tokenizer_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    # Создаем special_tokens_map.json
    special_tokens_map = {
        "unk_token": {"content": "[UNK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "pad_token": {"content": "[PAD]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "cls_token": {"content": "[CLS]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "sep_token": {"content": "[SEP]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
        "mask_token": {"content": "[MASK]", "single_word": False, "lstrip": False, "rstrip": False, "normalized": True},
    }
    
    special_tokens_path = output_path / "special_tokens_map.json"
    with open(special_tokens_path, 'w', encoding='utf-8') as f:
        json.dump(special_tokens_map, f, indent=2)
    
    # Копируем SentencePiece модель
    import shutil
    spm_dest = output_path / "spm.model"
    shutil.copy(sp_model_path, spm_dest)
    
    print(f"Tokenizer files saved to {output_path}")
    return output_path


def test_tokenizer(tokenizer_path: str):
    """Тестирует сохраненный токенизатор."""
    from transformers import PreTrainedTokenizerFast
    
    try:
        # Загружаем токенизатор
        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            tokenizer_path,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]",
        )
        
        # Добавляем специальные токены если их нет
        special_tokens = {
            "unk_token": "[UNK]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
        tokenizer.add_special_tokens(special_tokens)
        
        test_texts = [
            "Привет, мир!",
            "α = 0.05",
            "Москва—Петушки",
            "«Война и мир»",
            "H₂O + CO₂ → H₂CO₃",
        ]
        
        print("\n=== Testing tokenizer ===")
        for text in test_texts:
            tokens = tokenizer.tokenize(text)
            ids = tokenizer.encode(text)
            decoded = tokenizer.decode(ids)
            print(f"\nText: {text}")
            print(f"Tokens ({len(tokens)}): {tokens[:15]}")
            print(f"IDs ({len(ids)}): {ids[:15]}")
            print(f"Decoded: {decoded}")
        
        print("\n=== Tokenizer works! ===")
        return True
        
    except Exception as e:
        print(f"\nError loading tokenizer: {e}")
        print("\nBut files are saved. You can load tokenizer using:")
        print(f"  tokenizer = PreTrainedTokenizerFast.from_pretrained('{tokenizer_path}')")
        return False


if __name__ == "__main__":
    # Создаем токенизатор
    tokenizer_path = build_tokenizer_from_sp(
        sp_model_path="models/tokenizer/final/sp_100k.model",
        output_dir="models/tokenizer/hf",
    )
    
    # Тестируем
    test_tokenizer(str(tokenizer_path))
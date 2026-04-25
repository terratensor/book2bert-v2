#!/usr/bin/env python3
"""Простой токенизатор на SentencePiece, совместимый с datasets API."""

import json
from pathlib import Path
import sentencepiece as spm


class SimpleSPTokenizer:
    """Простой токенизатор для BERT, совместимый с HuggingFace datasets."""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # Специальные токены
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        
        self.cls_token_id = self._get_id("[CLS]")
        self.sep_token_id = self._get_id("[SEP]")
        self.pad_token_id = self._get_id("[PAD]")
        self.unk_token_id = self._get_id("[UNK]")
        self.mask_token_id = self._get_id("[MASK]")
        
        self.vocab_size = self.sp.GetPieceSize()
        
        # Создаем словарь для совместимости
        self.vocab = {}
        for i in range(self.vocab_size):
            self.vocab[self.sp.IdToPiece(i)] = i
        
        # Сохраняем специальные ID как атрибуты для совместимости
        self.pad_token_id = self.pad_token_id
        self.cls_token_id = self.cls_token_id
        self.sep_token_id = self.sep_token_id
        self.unk_token_id = self.unk_token_id
        self.mask_token_id = self.mask_token_id
    
    def _get_id(self, token: str) -> int:
        try:
            return self.sp.PieceToId(token)
        except:
            return 0
    
    def __call__(self, texts, max_length=512, padding='max_length', truncation=True, return_tensors=None, **kwargs):
        """Совместимость с HuggingFace Tokenizer API."""
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids = []
        attention_masks = []
        
        for text in texts:
            if not text:
                text = ""
            
            # Токенизируем через SentencePiece
            ids = self.sp.EncodeAsIds(text)
            
            # Добавляем специальные токены
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
            
            # Truncation
            if truncation and len(ids) > max_length:
                ids = ids[:max_length-1] + [self.sep_token_id]
            
            # Padding
            mask = [1] * len(ids)
            if padding == 'max_length':
                pad_len = max_length - len(ids)
                if pad_len > 0:
                    ids = ids + [self.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
        }
    
    def encode(self, text, max_length=512, **kwargs):
        result = self(text, max_length=max_length, **kwargs)
        return result['input_ids'][0]
    
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [i for i in ids if i not in {
                self.cls_token_id, self.sep_token_id, 
                self.pad_token_id, self.unk_token_id
            }]
        return self.sp.DecodeIds(ids)
    
    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)
    
    def convert_ids_to_tokens(self, ids):
        return [self.sp.IdToPiece(i) for i in ids]
    
    def convert_tokens_to_ids(self, tokens):
        return [self.sp.PieceToId(t) for t in tokens]
    
    def save_pretrained(self, path):
        """Сохранение в формате HuggingFace."""
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            "model_max_length": 512,
            "do_lower_case": False,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "vocab_size": self.vocab_size,
            "tokenizer_class": "SimpleSPTokenizer",
        }
        
        with open(output_path / "tokenizer_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        with open(output_path / "vocab.json", 'w') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        
        with open(output_path / "special_tokens_map.json", 'w') as f:
            json.dump({
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
            }, f, indent=2)
        
        print(f"Tokenizer saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path):
        """Загрузка токенизатора."""
        config_path = Path(path) / "tokenizer_config.json"
        with open(config_path) as f:
            config = json.load(f)
        
        # Ищем spm.model в той же директории или родительской
        spm_path = Path(path) / "spm.model"
        if not spm_path.exists():
            spm_path = Path(path).parent / "final" / "32k" / "sp_32k.model"
        
        return cls(str(spm_path))


# Тестируем
if __name__ == "__main__":
    tokenizer = SimpleSPTokenizer("models/tokenizer/final/32k/sp_32k.model")
    
    print("=== Testing SimpleSPTokenizer ===")
    test_texts = [
        "Привет, мир!",
        "α = 0.05",
        "Москва—Петушки",
        "«Война и мир»",
    ]
    
    for text in test_texts:
        print(f"\nText: {text}")
        print(f"Tokens: {tokenizer.tokenize(text)[:15]}")
        ids = tokenizer.encode(text, max_length=512)
        print(f"IDs: {ids[:20]}")
        print(f"Decoded: {tokenizer.decode(ids)}")
    
    # Сохраняем
    tokenizer.save_pretrained("models/tokenizer/simple_hf")
    
    # Проверяем загрузку
    print("\n=== Testing load ===")
    loaded = SimpleSPTokenizer.from_pretrained("models/tokenizer/simple_hf")
    print(f"Loaded tokenizer with vocab size: {loaded.vocab_size}")
    print(f"Test encode: {loaded.encode('Привет')[:15]}")
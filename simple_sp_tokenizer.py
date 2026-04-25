#!/usr/bin/env python3
"""Простой токенизатор на SentencePiece, совместимый с HuggingFace."""

import json
from pathlib import Path
import sentencepiece as spm


class SimpleSPTokenizer:
    """Простой токенизатор для BERT."""
    
    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
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
        self.vocab = {self.sp.IdToPiece(i): i for i in range(self.vocab_size)}
        self.model_max_length = 512

    def __len__(self):
        """Возвращает размер словаря (для совместимости с HuggingFace)."""
        return self.vocab_size  

    def _get_id(self, token: str) -> int:
        try:
            return self.sp.PieceToId(token)
        except:
            return 0
    
    def __call__(self, texts, max_length=512, padding='max_length', truncation=True, return_tensors=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        input_ids, attention_masks = [], []
        
        for text in texts:
            ids = self.sp.EncodeAsIds(text) if text else []
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
            
            if truncation and len(ids) > max_length:
                ids = ids[:max_length-1] + [self.sep_token_id]
            
            mask = [1] * len(ids)
            if padding == 'max_length':
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        import torch
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_masks, dtype=torch.long),
        }
    
    def pad(self, encoded_inputs, padding=True, max_length=None, pad_to_multiple_of=None, return_tensors=None):
        """
        Паддинг для совместимости с HuggingFace DataCollator.
        DataCollatorForLanguageModeling вызывает tokenizer.pad().
        """
        if isinstance(encoded_inputs, dict):
            return encoded_inputs
        
        if not encoded_inputs:
            return {}
        
        if max_length is None:
            max_length = self.model_max_length
        
        # Находим максимальную длину в батче
        batch_max_len = max(len(x['input_ids']) for x in encoded_inputs)
        if max_length is not None:
            batch_max_len = min(batch_max_len, max_length)
        
        input_ids = []
        attention_masks = []
        
        for item in encoded_inputs:
            ids = item['input_ids']
            mask = item['attention_mask']
            
            if isinstance(ids, list):
                # Truncate
                if len(ids) > batch_max_len:
                    ids = ids[:batch_max_len]
                    mask = mask[:batch_max_len]
                
                # Pad
                pad_len = batch_max_len - len(ids)
                if pad_len > 0:
                    ids = ids + [self.pad_token_id] * pad_len
                    mask = mask + [0] * pad_len
            else:
                # Уже тензор
                import torch
                if ids.shape[0] > batch_max_len:
                    ids = ids[:batch_max_len]
                    mask = mask[:batch_max_len]
                pad_len = batch_max_len - ids.shape[0]
                if pad_len > 0:
                    ids = torch.cat([ids, torch.full((pad_len,), self.pad_token_id, dtype=ids.dtype)])
                    mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
            
            input_ids.append(ids)
            attention_masks.append(mask)
        
        import torch
        # Конвертируем в тензоры если нужно
        if isinstance(input_ids[0], list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)
        else:
            input_ids = torch.stack(input_ids)
            attention_masks = torch.stack(attention_masks)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
        }
    
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Возвращает маску специальных токенов.
        Используется DataCollatorForLanguageModeling чтобы не маскировать CLS, SEP, PAD.
        
        Args:
            token_ids_0: список ID токенов
            token_ids_1: второй список (для пар предложений, не используется)
            already_has_special_tokens: добавлены ли уже CLS/SEP
        
        Returns:
            Список из 0 и 1, где 1 = специальный токен (не маскировать)
        """
        special_ids = {self.cls_token_id, self.sep_token_id, self.pad_token_id, self.unk_token_id, self.mask_token_id}
        
        if already_has_special_tokens:
            return [1 if tid in special_ids else 0 for tid in token_ids_0]
        else:
            # Если специальные токены ещё не добавлены, их нет в последовательности
            return [0] * len(token_ids_0)    
    
    def encode(self, text, max_length=512, **kwargs):
        result = self(text, max_length=max_length, **kwargs)
        return result['input_ids'][0].tolist()
    
    def decode(self, ids, skip_special_tokens=True):
        if skip_special_tokens:
            ids = [i for i in ids if i not in {self.cls_token_id, self.sep_token_id, self.pad_token_id}]
        return self.sp.DecodeIds(ids)
    
    def tokenize(self, text):
        return self.sp.EncodeAsPieces(text)
    
    def convert_ids_to_tokens(self, ids):
        """
        Конвертирует ID в токены.
        """
        if isinstance(ids, int):
            return self.sp.IdToPiece(ids)
        elif isinstance(ids, list):
            return [self.sp.IdToPiece(i) for i in ids]
        else:
            raise TypeError(f"Expected int or list, got {type(ids)}")
    
    def convert_tokens_to_ids(self, tokens):
        """
        Конвертирует токены в ID.
        
        Args:
            tokens: строка или список строк
        
        Returns:
            int (если tokens - строка) или list[int] (если список)
        """
        if isinstance(tokens, str):
            return self.sp.PieceToId(tokens)
        elif isinstance(tokens, list):
            return [self.sp.PieceToId(t) for t in tokens]
        else:
            raise TypeError(f"Expected str or list, got {type(tokens)}")
    
    def save_pretrained(self, path):
        output_path = Path(path)
        output_path.mkdir(parents=True, exist_ok=True)
        with open(output_path / "tokenizer_config.json", 'w') as f:
            json.dump({
                "model_max_length": self.model_max_length,
                "do_lower_case": False,
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "cls_token": self.cls_token,
                "sep_token": self.sep_token,
                "mask_token": self.mask_token,
                "vocab_size": self.vocab_size,
            }, f, indent=2)
        with open(output_path / "vocab.json", 'w') as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_pretrained(cls, path):
        config_path = Path(path) / "tokenizer_config.json"
        with open(config_path) as f:
            config = json.load(f)
        spm_path = Path(path).parent / "final" / "32k" / "sp_32k.model"
        if not spm_path.exists():
            spm_path = "models/tokenizer/final/32k/sp_32k.model"
        return cls(str(spm_path))
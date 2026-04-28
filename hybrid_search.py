#!/usr/bin/env python3
"""
Интерактивный гибридный поиск с Query Expansion для Geomatrix BERT + Manticore.
Использует BERT-base (64k multilingual) + SimpleSPTokenizer.
Query Expansion через MLM (Masked Language Modeling).
"""

import sys
import json
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Optional
import readline

import torch
import torch.nn.functional as F
from transformers import BertForMaskedLM, BertConfig
from simple_sp_tokenizer import SimpleSPTokenizer


class GeomatrixHybridSearch:
    """
    Гибридный поиск: BERT (семантика) + Manticore (ключевые слова).
    Query Expansion через MLM предсказания.
    """
    
    MAX_SEQ_LEN = 512
    HIDDEN_SIZE = 768
    
    def __init__(
        self,
        bert_model_path: str,
        tokenizer_path: str,
        manticore_url: str = "http://localhost:9308/search",
        table_name: str = "library2026",
        device: str = "cuda"
    ):
        self.manticore_url = manticore_url
        self.table_name = table_name
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Device: {self.device}")
        
        # Токенизатор
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = SimpleSPTokenizer(tokenizer_path)
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        
        # BERT
        print(f"Loading BERT from {bert_model_path}...")
        self._load_bert(bert_model_path)
        
        # Настройки
        self.settings = {
            'top_k': 10,
            'semantic_weight': 0.6,
            'keyword_weight': 0.4,
            'use_expansion': False,
            'expansion_top_k': 3,
            'manticore_limit': 100
        }
    
    def _load_bert(self, model_path: str):
        """Загрузка BertForMaskedLM (нужен для MLM expansion)."""
        try:
            self.bert = BertForMaskedLM.from_pretrained(model_path)
        except:
            config = BertConfig(
                vocab_size=64000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
            )
            self.bert = BertForMaskedLM(config)
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
            self.bert.load_state_dict(state_dict, strict=False)
        
        self.bert.to(self.device)
        self.bert.eval()
        
        params = sum(p.numel() for p in self.bert.parameters())
        print(f"BERT loaded: {params:,} parameters")
    
    # ========================================================================
    # QUERY EXPANSION через MLM
    # ========================================================================
    
    def expand_query_mlm(self, query: str, top_k: int = 5) -> List[str]:
        """
        Расширение запроса через MLM.
        Маскирует слова и предсказывает альтернативы.
        """
        words = query.split()
        if len(words) < 2:
            return [query]
        
        variations = set()
        variations.add(query)
        
        for i, word in enumerate(words):
            # Пропускаем короткие и служебные слова
            if len(word) <= 2:
                continue
            
            # Маскируем слово
            masked = words.copy()
            masked[i] = self.tokenizer.mask_token
            masked_query = ' '.join(masked)
            
            # Токенизируем
            encoded = self.tokenizer(masked_query, max_length=128, return_tensors='pt')
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Находим позицию [MASK]
            mask_positions = (encoded['input_ids'][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                continue
            
            with torch.no_grad():
                outputs = self.bert(**encoded)
                logits = outputs.logits[0, mask_positions[0]]
                probs = torch.softmax(logits, dim=-1)
                top_tokens = torch.topk(probs, top_k * 2).indices.tolist()
            
            # Декодируем и фильтруем
            for token_id in top_tokens:
                token = self.tokenizer.convert_ids_to_tokens(token_id)
                token = token.replace('▁', '').strip()
                
                # Фильтр: не пунктуация, не оригинальное слово, не слишком короткое
                if token and len(token) > 1 and token != word:
                    if not all(c in '.,!?;:()[]{}""''«»—–-…' for c in token):
                        new_words = words.copy()
                        new_words[i] = token
                        variations.add(' '.join(new_words))
        
        # Возвращаем топ-k (кроме оригинала)
        result = list(variations)
        if len(result) > 1:
            result.remove(query)
        return result[:top_k]
    
    # ========================================================================
    # ЭМБЕДДИНГИ
    # ========================================================================
    
    def get_embedding(self, text: str, pooling: str = "mean") -> np.ndarray:
        """Эмбеддинг текста через mean pooling."""
        if not text or len(text.strip()) == 0:
            return np.zeros(self.HIDDEN_SIZE)
        
        encoded = self.tokenizer(
            text, max_length=self.MAX_SEQ_LEN,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        if pooling == "cls":
            embedding = hidden_states[:, 0, :].cpu().numpy()
        else:
            mask = attention_mask.cpu().numpy()
            embeddings = hidden_states.cpu().numpy()
            sum_mask = np.sum(mask, axis=1, keepdims=True)
            sum_mask = np.where(sum_mask == 0, 1.0, sum_mask)
            embedding = np.sum(embeddings * mask[:, :, np.newaxis], axis=1) / sum_mask
        
        embedding = embedding.squeeze()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    # ========================================================================
    # MANTICORE
    # ========================================================================
    
    def search_manticore(self, query: str, limit: int = 1000) -> Dict:
        """Поиск в Manticore (BM25)."""
        payload = {
            "index": self.table_name,
            "query": {"query_string": f'@content {query}'},
            "limit": limit,
            "options": {"ranker": "bm25"}
        }
        
        try:
            response = requests.post(
                self.manticore_url,
                headers={'Content-Type': 'application/json'},
                json=payload,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"hits": {"hits": [], "total": 0}}
        except Exception:
            return {"hits": {"hits": [], "total": 0}}
    
    # ========================================================================
    # ГИБРИДНЫЙ ПОИСК
    # ========================================================================
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = None,
        keyword_weight: float = None,
        use_expansion: bool = None,
        verbose: bool = True
    ) -> List[Dict]:
        """Гибридный поиск с Query Expansion."""
        
        top_k = top_k or self.settings['top_k']
        semantic_weight = semantic_weight or self.settings['semantic_weight']
        keyword_weight = keyword_weight or self.settings['keyword_weight']
        use_expansion = use_expansion if use_expansion is not None else self.settings['use_expansion']
        limit = self.settings['manticore_limit']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SEARCH: '{query}'")
            print(f"{'='*60}")
        
        # 1. Query Expansion
        all_queries = [query]
        
        if use_expansion:
            if verbose:
                print("1. Query Expansion (MLM)...")
            variations = self.expand_query_mlm(query, top_k=self.settings['expansion_top_k'])
            all_queries.extend(variations)
            if verbose:
                print(f"   Expanded to: {all_queries}")
        else:
            if verbose:
                print("1. Query Expansion: OFF")
        
        # 2. Поиск в Manticore по всем запросам
        if verbose:
            print("2. Manticore search...")
        
        all_hits = {}
        for q in all_queries:
            results = self.search_manticore(q, limit=limit // len(all_queries))
            for hit in results.get('hits', {}).get('hits', []):
                doc_id = hit.get('_id')
                if doc_id not in all_hits:
                    all_hits[doc_id] = hit
                else:
                    # Сохраняем лучший score
                    if hit.get('_score', 0) > all_hits[doc_id].get('_score', 0):
                        all_hits[doc_id] = hit
        
        hits = list(all_hits.values())
        
        if verbose:
            print(f"   Total unique documents: {len(hits)}")
        
        if not hits:
            return []
        
        # 3. Эмбеддинги
        if verbose:
            print(f"3. Computing embeddings for {len(hits)} documents...")
        
        candidate_texts = []
        candidate_scores = []
        
        for hit in hits:
            source = hit.get('_source', {})
            content = source.get('content', '')
            if isinstance(content, str) and content.strip():
                candidate_texts.append(content)
                candidate_scores.append(hit.get('_score', 0))
        
        if not candidate_texts:
            return []
        
        # Эмбеддинг запроса
        query_emb = self.get_embedding(query)
        
        # Эмбеддинги документов
        doc_embs = []
        for i, text in enumerate(candidate_texts):
            doc_embs.append(self.get_embedding(text))
            if verbose and (i+1) % 500 == 0:
                print(f"   {i+1}/{len(candidate_texts)}")
        
        doc_embs = np.array(doc_embs)
        
        # 4. Скоринг
        semantic_scores = np.dot(doc_embs, query_emb)
        
        max_kw = max(candidate_scores) if candidate_scores else 1
        keyword_scores_norm = np.array([s / max_kw for s in candidate_scores])
        
        combined = semantic_weight * semantic_scores + keyword_weight * keyword_scores_norm
        
        # 5. Топ-K
        top_indices = np.argsort(combined)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            hit = hits[idx]
            source = hit.get('_source', {})
            
            results.append({
                'id': hit.get('_id'),
                'score': float(combined[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'keyword_score': float(keyword_scores_norm[idx]),
                'title': source.get('title', 'No title'),
                'author': source.get('author', 'Unknown'),
                'content': source.get('content', '')[:2000],
                'genre': source.get('genre', ''),
                'book_id': source.get('book_id', ''),
            })
        
        return results


def print_results(results: List[Dict], query: str):
    """Вывод результатов."""
    if not results:
        print("\nNo results found")
        return
    
    print(f"\n{'='*60}")
    print(f"RESULTS FOR: '{query}'")
    print(f"{'='*60}")
    
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [Score: {r['score']:.4f} | SEM: {r['semantic_score']:.4f} | KW: {r['keyword_score']:.4f}]")
        print(f"   Title: {r['title']}")
        print(f"   Author: {r['author']}")
        print(f"   Genre: {r['genre']}")
        print(f"   Content: {r['content'][:2000]}...")
        print(f"   ---")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to BERT checkpoint')
    parser.add_argument('--tokenizer', type=str, default='models/tokenizer/multilingual/sp_42k.model')
    parser.add_argument('--manticore', type=str, default='http://localhost:9308/search')
    parser.add_argument('--table', type=str, default='library2026')
    args = parser.parse_args()
    
    print("Loading Geomatrix Hybrid Search...")
    
    searcher = GeomatrixHybridSearch(
        bert_model_path=args.model,
        tokenizer_path=args.tokenizer,
        manticore_url=args.manticore,
        table_name=args.table,
    )
    
    print("\n" + "="*60)
    print("Interactive Hybrid Search with Query Expansion")
    print("Commands: /quit, /help, /settings, /set, /expansion on/off")
    print("="*60)
    
    while True:
        try:
            query = input("\nSearch> ").strip()
            if not query:
                continue
            
            if query.lower() in ['/quit', '/exit', '/q']:
                print("Goodbye!")
                break
            elif query.lower() == '/help':
                print("""
Commands:
  /quit, /exit, /q - Exit
  /help            - Help
  /settings        - Show settings
  /expansion on/off - Enable/disable query expansion
  /set top_k N     - Number of results
  /set semantic_weight 0.6 - Semantic weight
  /set keyword_weight 0.4  - Keyword weight
""")
                continue
            elif query.lower() == '/settings':
                s = searcher.settings
                print(f"top_k={s['top_k']}, sem={s['semantic_weight']}, kw={s['keyword_weight']}, "
                      f"expansion={'ON' if s['use_expansion'] else 'OFF'}, "
                      f"expansion_top_k={s['expansion_top_k']}")
                continue
            elif query.lower().startswith('/expansion'):
                parts = query.split()
                if len(parts) == 2:
                    if parts[1].lower() == 'on':
                        searcher.settings['use_expansion'] = True
                        print("Query Expansion: ON")
                    elif parts[1].lower() == 'off':
                        searcher.settings['use_expansion'] = False
                        print("Query Expansion: OFF")
                continue
            elif query.lower().startswith('/set'):
                parts = query.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    if param == 'top_k':
                        searcher.settings['top_k'] = int(value)
                    elif param == 'semantic_weight':
                        searcher.settings['semantic_weight'] = float(value)
                    elif param == 'keyword_weight':
                        searcher.settings['keyword_weight'] = float(value)
                    elif param == 'expansion_top_k':
                        searcher.settings['expansion_top_k'] = int(value)
                    print(f"Set {param}={value}")
                continue
            
            results = searcher.hybrid_search(query)
            print_results(results, query)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
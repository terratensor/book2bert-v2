#!/usr/bin/env python3
"""
Интерактивный гибридный поиск для Geomatrix BERT + Manticore.
Использует нашу BERT-base модель (64k multilingual) и SimpleSPTokenizer.
"""

import sys
import json
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Optional
import readline

import torch
from transformers import BertModel, BertConfig
from simple_sp_tokenizer import SimpleSPTokenizer


class GeomatrixHybridSearch:
    """
    Гибридный поиск для библиотеки Geomatrix.
    Использует BERT-base (64k multilingual) + Manticore Search.
    """
    
    MAX_SEQ_LEN = 512
    HIDDEN_SIZE = 768  # BERT-base
    
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
        
        # Загружаем токенизатор
        print(f"Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = SimpleSPTokenizer(tokenizer_path)
        print(f"Vocab size: {self.tokenizer.vocab_size}")
        
        # Загружаем BERT
        print(f"Loading BERT from {bert_model_path}...")
        self._load_bert(bert_model_path)
        
        # Настройки поиска
        self.settings = {
            'top_k': 10,
            'semantic_weight': 0.6,
            'keyword_weight': 0.4,
            'manticore_limit': 100
        }
    
    def _load_bert(self, model_path: str):
        """Загрузка BERT-base модели."""
        # Пробуем загрузить как полную модель HuggingFace
        try:
            self.bert = BertModel.from_pretrained(model_path)
        except:
            # Если не получается — создаём с нуля и загружаем веса
            print("Creating model from config...")
            config = BertConfig(
                vocab_size=64000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=512,
            )
            self.bert = BertModel(config)
            
            # Загружаем веса из чекпоинта
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Убираем префикс 'bert.' если есть
            state_dict = {k.replace('bert.', ''): v for k, v in state_dict.items()}
            self.bert.load_state_dict(state_dict, strict=False)
        
        self.bert.to(self.device)
        self.bert.eval()
        
        params = sum(p.numel() for p in self.bert.parameters())
        print(f"BERT loaded: {params:,} parameters")
    
    def get_embedding(self, text: str, pooling: str = "mean") -> np.ndarray:
        """Получение эмбеддинга текста."""
        if not text or len(text.strip()) == 0:
            return np.zeros(self.HIDDEN_SIZE)
        
        # Токенизируем
        encoded = self.tokenizer(
            text,
            max_length=self.MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        
        if pooling == "cls":
            # [CLS] токен
            embedding = hidden_states[:, 0, :].cpu().numpy()
        else:
            # Mean pooling с учётом attention mask
            mask = attention_mask.cpu().numpy()
            embeddings = hidden_states.cpu().numpy()
            sum_mask = np.sum(mask, axis=1, keepdims=True)
            sum_mask = np.where(sum_mask == 0, 1.0, sum_mask)
            embedding = np.sum(embeddings * mask[:, :, np.newaxis], axis=1) / sum_mask
        
        # Нормализуем
        embedding = embedding.squeeze()
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def search_manticore(self, query: str, limit: int = 1000) -> Dict:
        """Поиск в Manticore."""
        payload = {
            "index": self.table_name,
            "query": {"query_string": query},
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
                print(f"Manticore error: {response.status_code}")
                return {"hits": {"hits": [], "total": 0}}
        except Exception as e:
            print(f"Manticore connection error: {e}")
            return {"hits": {"hits": [], "total": 0}}
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = None,
        keyword_weight: float = None,
        verbose: bool = True
    ) -> List[Dict]:
        """Гибридный поиск."""
        
        top_k = top_k or self.settings['top_k']
        semantic_weight = semantic_weight or self.settings['semantic_weight']
        keyword_weight = keyword_weight or self.settings['keyword_weight']
        limit = self.settings['manticore_limit']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"SEARCH: '{query}'")
            print(f"{'='*60}")
        
        # 1. Поиск в Manticore
        if verbose:
            print("1. Manticore search...")
        
        results = self.search_manticore(query, limit=limit)
        hits = results.get('hits', {}).get('hits', [])
        total = results.get('hits', {}).get('total', 0)
        
        if verbose:
            print(f"   Found: {total} documents, retrieved: {len(hits)}")
        
        if not hits:
            return []
        
        # 2. Вычисляем эмбеддинги
        if verbose:
            print(f"2. Computing embeddings for {len(hits)} documents...")
        
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
        if verbose:
            print("3. Computing query embedding...")
        query_emb = self.get_embedding(query)
        
        # Эмбеддинги документов
        doc_embs = []
        batch_size = 2
        
        for i in range(0, len(candidate_texts), batch_size):
            batch = candidate_texts[i:i+batch_size]
            for text in batch:
                emb = self.get_embedding(text)
                doc_embs.append(emb)
            
            if verbose:
                print(f"   Processed {min(i+batch_size, len(candidate_texts))}/{len(candidate_texts)}")
        
        doc_embs = np.array(doc_embs)
        
        # 3. Скоринг
        semantic_scores = np.dot(doc_embs, query_emb)
        
        max_kw = max(candidate_scores) if candidate_scores else 1
        keyword_scores_norm = np.array([s / max_kw for s in candidate_scores])
        
        combined = semantic_weight * semantic_scores + keyword_weight * keyword_scores_norm
        
        # 4. Сортировка
        top_indices = np.argsort(combined)[-top_k:][::-1]
        
        # 5. Результат
        final_results = []
        for idx in top_indices:
            hit = hits[idx]
            source = hit.get('_source', {})
            
            final_results.append({
                'id': hit.get('_id'),
                'score': float(combined[idx]),
                'semantic_score': float(semantic_scores[idx]),
                'keyword_score': float(keyword_scores_norm[idx]),
                'title': source.get('title', 'No title'),
                'author': source.get('author', 'Unknown'),
                'content': source.get('content', '')[:3000],
                'genre': source.get('genre', ''),
                'book_id': source.get('book_id', ''),
            })
        
        return final_results


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
        print(f"   Content: {r['content'][:3000]}...")
        print(f"   ---")


def main():
    """Интерактивный поиск."""
    
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
    print("Interactive Hybrid Search")
    print("Commands: /quit, /help, /settings, /set")
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
  /set top_k N     - Number of results
  /set semantic_weight 0.6 - Semantic weight (0-1)
  /set keyword_weight 0.4  - Keyword weight (0-1)
""")
                continue
            
            elif query.lower() == '/settings':
                s = searcher.settings
                print(f"top_k={s['top_k']}, sem={s['semantic_weight']}, kw={s['keyword_weight']}")
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
                    print(f"Set {param}={value}")
                continue
            
            # Поиск
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
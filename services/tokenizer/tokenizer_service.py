#!/usr/bin/env python3
"""
HTTP-сервис токенизации через SentencePiece.
Запуск: gunicorn -w 16 -b 0.0.0.0:8093 --timeout 30 tokenizer_service:app
"""

import json
import logging
import time
import os
import sys

# Добавляем путь к модели
MODEL_PATH = os.environ.get("SP_MODEL_PATH", "models/tokenizer/final/sp_100k.model")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Пытаемся импортировать Flask и sentencepiece
try:
    from flask import Flask, request, jsonify
except ImportError:
    logger.error("Flask not installed. Run: pip install flask")
    sys.exit(1)

try:
    import sentencepiece as spm
except ImportError:
    logger.error("sentencepiece not installed. Run: pip install sentencepiece")
    sys.exit(1)

app = Flask(__name__)

# Глобальная модель (загружается один раз на воркер)
_sp = None


def get_sp():
    """Ленивая загрузка модели SentencePiece."""
    global _sp
    if _sp is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        
        _sp = spm.SentencePieceProcessor()
        _sp.Load(MODEL_PATH)
        logger.info(f"SentencePiece loaded: vocab_size={_sp.GetPieceSize()}, model={MODEL_PATH}")
    return _sp


@app.route("/health", methods=["GET"])
def health():
    """Проверка работоспособности."""
    try:
        sp = get_sp()
        return jsonify({
            "status": "ok",
            "model": MODEL_PATH,
            "vocab_size": sp.GetPieceSize(),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/tokenize", methods=["POST"])
def tokenize():
    """
    Токенизация одного текста.
    Ожидает: {"text": "текст..."}
    Возвращает: {"ids": [5, 20286, ...]}
    """
    start_time = time.time()
    
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data["text"]
    if not isinstance(text, str):
        return jsonify({"error": "Text must be string"}), 400
    
    if not text.strip():
        return jsonify({"ids": []})
    
    try:
        sp = get_sp()
        ids = sp.EncodeAsIds(text)
        
        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"Tokenized {len(text)} chars -> {len(ids)} ids in {elapsed:.2f} ms")
        
        return jsonify({"ids": ids})
        
    except Exception as e:
        logger.exception(f"Tokenization failed for text length {len(text)}")
        return jsonify({"error": str(e)}), 500


@app.route("/tokenize_batch", methods=["POST"])
def tokenize_batch():
    """
    Пакетная токенизация.
    Ожидает: {"texts": ["текст1", "текст2", ...]}
    Возвращает: {"results": [[id1, id2], ...]}
    """
    start_time = time.time()
    
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
    
    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    
    # Ограничение на размер батча
    if len(texts) > 1000:
        return jsonify({"error": f"Batch size {len(texts)} exceeds limit 1000"}), 400
    
    sp = get_sp()
    results = []
    failed = 0
    
    for text in texts:
        if isinstance(text, str) and text.strip():
            try:
                ids = sp.EncodeAsIds(text)
                results.append(ids)
            except Exception:
                results.append([])
                failed += 1
        else:
            results.append([])
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Batch tokenized {len(texts)} texts, {failed} failed in {elapsed:.2f} ms")
    
    return jsonify({"results": results, "failed": failed})


@app.route("/special_tokens", methods=["GET"])
def special_tokens():
    """Возвращает ID специальных токенов."""
    sp = get_sp()
    return jsonify({
        "cls": sp.PieceToId("[CLS]"),
        "sep": sp.PieceToId("[SEP]"),
        "pad": sp.PieceToId("[PAD]"),
        "unk": sp.PieceToId("[UNK]"),
        "mask": sp.PieceToId("[MASK]"),
        "vocab_size": sp.GetPieceSize(),
    })


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    # Для локальной отладки
    port = int(os.environ.get("PORT", 8093))
    app.run(host="0.0.0.0", port=port, debug=False)
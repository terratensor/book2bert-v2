#!/usr/bin/env python3
"""
HTTP-сервис определения языка через fastText.
Запуск: gunicorn -w 8 -b 0.0.0.0:8092 --timeout 10 app:app
"""

import os
import logging
import time
import fasttext
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Путь к модели fastText
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(BASE_DIR, "models", "fasttext", "lid.176.ftz")
MODEL_PATH = os.environ.get("FASTTEXT_MODEL", DEFAULT_MODEL)

logger.info(f"Loading fastText model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    logger.error(f"Model not found at {MODEL_PATH}")
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = fasttext.load_model(MODEL_PATH)
logger.info("Model loaded!")


def get_cyrillic_ratio(text: str) -> float:
    """Возвращает долю кириллических букв среди всех букв текста."""
    cyrillic = sum(1 for c in text if 'А' <= c <= 'я' or c in ('Ё', 'ё'))
    latin = sum(1 for c in text if 'A' <= c <= 'Z' or 'a' <= c <= 'z')
    total = cyrillic + latin
    if total == 0:
        return 0.0
    return cyrillic / total


def clean_text_for_fasttext(text: str) -> str:
    """Удаляет \n и лишние пробелы для fastText."""
    text = text.replace('\n', ' ').replace('\r', ' ')
    return ' '.join(text.split())


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": MODEL_PATH})


@app.route("/detect", methods=["POST"])
def detect():
    """
    Определяет язык текста.
    Ожидает: {"text": "текст..."}
    Возвращает: {"lang": "ru", "confidence": 0.99, "cyrillic_ratio": 0.95, "keep": true}
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data["text"].strip()
    if not text:
        return jsonify({"lang": "unknown", "confidence": 0.0, "cyrillic_ratio": 0.0, "keep": False})
    
    # Очищаем текст для fastText
    text = clean_text_for_fasttext(text)
    if not text:
        return jsonify({"lang": "unknown", "confidence": 0.0, "cyrillic_ratio": 0.0, "keep": False})
    
    # fastText определяет язык
    lang, conf = model.predict(text, k=1)
    lang = lang[0].replace("__label__", "")
    conf = float(conf[0])  # ← конвертируем np.float32 в Python float
    
    # Дополнительная эвристика
    cyrillic_ratio = get_cyrillic_ratio(text)
    
    # Правило удержания
    keep = (lang == "ru" and conf > 0.5) or cyrillic_ratio > 0.5
    
    return jsonify({
        "lang": lang,
        "confidence": round(conf, 4),
        "cyrillic_ratio": round(cyrillic_ratio, 4),
        "keep": bool(keep)  # ← конвертируем np.bool_ в Python bool
    })


@app.route("/detect_batch", methods=["POST"])
def detect_batch():
    """
    Пакетное определение языка.
    Ожидает: {"texts": ["текст1", "текст2", ...]}
    Возвращает: {"results": [{"lang": "ru", ...}, ...]}
    """
    start_time = time.time()
    
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
    
    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    
    if len(texts) > 1000:
        return jsonify({"error": f"Batch size {len(texts)} exceeds limit 1000"}), 400
    
    results = []
    for text in texts:
        if not text or not text.strip():
            results.append({"lang": "unknown", "confidence": 0.0, "cyrillic_ratio": 0.0, "keep": False})
            continue
        
        # Очищаем текст
        cleaned = clean_text_for_fasttext(text.strip())
        if not cleaned:
            results.append({"lang": "unknown", "confidence": 0.0, "cyrillic_ratio": 0.0, "keep": False})
            continue
        
        try:
            lang, conf = model.predict(cleaned, k=1)
            lang = lang[0].replace("__label__", "")
            conf = float(conf[0])  # ← конвертируем в Python float
            
            cyrillic_ratio = get_cyrillic_ratio(cleaned)
            keep = (lang == "ru" and conf > 0.5) or cyrillic_ratio > 0.5
            
            results.append({
                "lang": lang,
                "confidence": round(conf, 4),
                "cyrillic_ratio": round(cyrillic_ratio, 4),
                "keep": bool(keep)  # ← конвертируем в Python bool
            })
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            results.append({"lang": "unknown", "confidence": 0.0, "cyrillic_ratio": 0.0, "keep": False})
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Batch detected {len(texts)} texts in {elapsed:.2f} ms")
    
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092, debug=False)
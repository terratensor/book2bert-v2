"""
HTTP-сервис сегментации текста на предложения.
Использует библиотеку razdel для русского языка.
"""

import json
import logging
import time
from typing import List

from flask import Flask, request, jsonify
from razdel import sentenize

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Проверка работоспособности сервиса."""
    return jsonify({"status": "ok"})


@app.route("/segment", methods=["POST"])
def segment():
    """
    Сегментация одного текста.
    Ожидает JSON: {"text": "текст..."}
    Возвращает: {"sentences": ["предложение1", "предложение2"]}
    """
    start_time = time.time()

    # Получаем и валидируем входные данные
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Text is empty or invalid"}), 400

    try:
        # Сегментация
        sentences = [s.text for s in sentenize(text)]

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "Segmented %d sentences from %d chars in %.2f ms",
            len(sentences), len(text), elapsed
        )

        return jsonify({"sentences": sentences})

    except Exception as e:
        logger.exception("Segmentation failed")
        return jsonify({"error": str(e)}), 500


@app.route("/segment_batch", methods=["POST"])
def segment_batch():
    """
    Пакетная сегментация.
    Ожидает JSON: {"texts": ["текст1", "текст2", ...]}
    Возвращает: {"results": [["предложение1a", ...], ["предложение2a", ...]]}
    """
    start_time = time.time()

    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400

    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400

    results = []
    for text in texts:
        if isinstance(text, str) and text.strip():
            try:
                sentences = [s.text for s in sentenize(text)]
                results.append(sentences)
            except Exception as e:
                logger.error("Batch item failed: %s", e)
                results.append([])
        else:
            results.append([])

    elapsed = (time.time() - start_time) * 1000
    logger.info(
        "Batch segmented %d texts in %.2f ms",
        len(texts), elapsed
    )

    return jsonify({"results": results})


if __name__ == "__main__":
    # Для локальной отладки
    app.run(host="0.0.0.0", port=8090, debug=False)
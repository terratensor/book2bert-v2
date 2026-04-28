#!/usr/bin/env python3
"""
HTTP-сервис вычисления перплексии через BERT-tiny.
"""

import os
import logging
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer
from flask import Flask, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Загружаем BERT-tiny (русский)
MODEL_NAME = os.environ.get("PERPLEXITY_MODEL", "cointegrated/rubert-tiny2")
logger.info(f"Loading model {MODEL_NAME}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Порог для garbage (можно настроить)
GARBAGE_THRESHOLD = float(os.environ.get("GARBAGE_THRESHOLD", "500"))

logger.info(f"Model loaded on {device}, garbage threshold: {GARBAGE_THRESHOLD}")


def compute_perplexity(text: str) -> dict:
    """Вычисляет перплексию и дополнительную статистику."""
    if not text or len(text) < 5:
        return {"perplexity": 0.0, "is_garbage": True, "text_length": len(text)}
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs.input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id
        )
        
        perplexity = torch.exp(loss).item()
    
    # Дополнительная эвристика: текст с перплексией > порога ИЛИ с очень высокой (>2000) — точно мусор
    is_garbage = perplexity > GARBAGE_THRESHOLD or perplexity > 2000
    
    return {
        "perplexity": round(perplexity, 2),
        "is_garbage": is_garbage,
        "text_length": len(text)
    }


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(device),
        "garbage_threshold": GARBAGE_THRESHOLD
    })


@app.route("/perplexity", methods=["POST"])
def perplexity():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    
    text = data["text"].strip()
    if not text:
        return jsonify({"perplexity": 0.0, "is_garbage": True, "text_length": 0})
    
    result = compute_perplexity(text)
    return jsonify(result)


@app.route("/perplexity_batch", methods=["POST"])
def perplexity_batch():
    start_time = time.time()
    
    data = request.get_json()
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400
    
    texts = data["texts"]
    if not isinstance(texts, list):
        return jsonify({"error": "'texts' must be a list"}), 400
    
    if len(texts) > 500:
        return jsonify({"error": f"Batch size {len(texts)} exceeds limit 500"}), 400
    
    results = []
    for text in texts:
        if not text or not text.strip():
            results.append({"perplexity": 0.0, "is_garbage": True, "text_length": 0})
            continue
        
        results.append(compute_perplexity(text.strip()))
    
    elapsed = (time.time() - start_time) * 1000
    logger.info(f"Batch processed {len(texts)} texts in {elapsed:.2f} ms")
    
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8093, debug=False)
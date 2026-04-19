# Book2BERT v2 — Geomatrix BERT

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Geomatrix%20BERT-yellow)](https://huggingface.co/terratensor/geomatrix-bert-base)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Go](https://img.shields.io/badge/Go-1.23+-00ADD8.svg)](https://golang.org)

**Geomatrix BERT** — BERT-base модель, обученная с нуля на корпусе русскоязычной научной, технической и исторической литературы.

## 📊 Характеристики

| Параметр | Значение |
|---|---|
| Архитектура | BERT-base (12 слоёв, 768 hidden, 12 голов) |
| Параметры | 110M |
| Словарь | 100,000 токенов (SentencePiece Unigram) |
| Корпус | 176,409 книг, 831M предложений |
| Длительность обучения | ~11 дней на 2×GPU |
| Макс. длина | 512 токенов |

## 🚀 Быстрый старт

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("terratensor/geomatrix-bert-base")
tokenizer = AutoTokenizer.from_pretrained("terratensor/geomatrix-bert-base")

# Токенизация
text = "Москва — столица России"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)

# Эмбеддинги
embeddings = outputs.last_hidden_state
```

## 📚 Документация

- [Полная документация](docs/README.md)
- [Подготовка корпуса](docs/corpus_preparation.md)
- [Обучение токенизатора](docs/tokenizer_training.md)
- [Сборка датасета](docs/dataset_building.md)
- [Обучение модели](docs/model_training.md)
- [Запуск в Docker](docs/docker.md)

## 🛠 Установка

### Через pip

```bash
pip install transformers sentencepiece
```

### Из исходников

```bash
git clone https://github.com/terratensor/book2bert-v2
cd book2bert-v2
pip install -r requirements.txt
```

## 📖 Примеры использования

### 1. Получение эмбеддингов

```python
from transformers import AutoModel, AutoTokenizer
import torch

model = AutoModel.from_pretrained("terratensor/geomatrix-bert-base")
tokenizer = AutoTokenizer.from_pretrained("terratensor/geomatrix-bert-base")

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

text = "Квантовая механика — фундаментальная физическая теория"
embedding = get_embedding(text)
print(embedding.shape)  # (768,)
```

### 2. MLM (Masked Language Modeling)

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="terratensor/geomatrix-bert-base")
result = fill_mask("Москва — [MASK] России")
print(result[0]["token_str"])  # "столица"
```

## 🎯 Бенчмарки

| Задача | Метрика | Значение |
|---|---|---|
| MLM Accuracy | Top-1 | XX% |
| MLM Accuracy | Top-5 | XX% |
| Perplexity | — | XX |

*Бенчмарки на русскоязычных датасетах будут добавлены.*

## 📦 Структура проекта

```
book2bert-v2/
├── cmd/                    # Go-утилиты
├── pkg/                    # Go-пакеты
├── services/               # Python-сервисы
├── scripts/                # Python-скрипты
├── models/                 # Обученные модели
├── data/                   # Данные
├── docs/                   # Документация
├── docker/                 # Docker-файлы
├── docker-compose.yml      # Docker Compose конфигурация
├── requirements.txt        # Python-зависимости
├── go.mod                  # Go-зависимости
└── LICENSE                 # Apache 2.0
```

## 🤝 Благодарности

- Корпус собран из открытых источников (flibusta, militera, geomatrix)
- Сегментация текста — [razdel](https://github.com/natasha/razdel)
- Токенизация — [SentencePiece](https://github.com/google/sentencepiece)

## 📄 Лицензия

Apache 2.0. Подробнее в файле [LICENSE](LICENSE).

## 🔗 Ссылки

- [🤗 HuggingFace Model](https://huggingface.co/terratensor/geomatrix-bert-base)
- [🌐 Geomatrix Project](https://gmtx.ru)
- [📧 Контакты](mailto:...)

## Инструкция по запуску обработки корпуса (v2)

---

### 1. Запуск сервиса сегментации (razdel)

Сервис должен быть запущен ДО начала обработки корпусов.

```bash
cd /mnt/work/audetv/go/src/github.com/terratensor/book2bert-v2/segmenter

# Запуск с 16 воркерами (важно для производительности)
gunicorn -w 16 -b 0.0.0.0:8090 --timeout 120 app:app
```

**Проверка работоспособности:**

```bash
curl http://localhost:8090/health
# Ожидаемый ответ: {"status":"ok"}
```

---

### 2. Обработка корпусов (по очереди)

Все команды выполняются из директории `v2`:

```bash
cd /mnt/work/audetv/go/src/github.com/terratensor/book2bert-v2
```

#### 2.1 Обработка flibusta_2023 (основной корпус, 143k файлов)

```bash
go run cmd/process-corpus/main.go \
    --corpus data/raw/flibusta_2023 \
    --output data/processed \
    --segmenter http://localhost:8090 \
    --workers 16
```

**Ожидаемый результат:**
- Время: ~2-3 часа
- Выход: `data/processed/books_meta.jsonl` + файлы с предложениями

#### 2.2 Обработка flibusta_2025 (дополнение, 21k файлов)

```bash
go run cmd/process-corpus/main.go \
    --corpus data/raw/flibusta_2025 \
    --output data/processed \
    --segmenter http://localhost:8090 \
    --workers 16
```

**Ожидаемый результат:**
- Время: ~30-40 минут

#### 2.3 Обработка militera (военная литература, 11k файлов)

```bash
go run cmd/process-corpus/main.go \
    --corpus data/raw/militera \
    --output data/processed \
    --segmenter http://localhost:8090 \
    --workers 16
```

**Ожидаемый результат:**
- Время: ~15-20 минут

#### 2.4 Обработка geomatrix (география, 191 файл)

```bash
go run cmd/process-corpus/main.go \
    --corpus data/raw/geomatrix \
    --output data/processed \
    --segmenter http://localhost:8090 \
    --workers 16
```

**Ожидаемый результат:**
- Время: ~1-2 минуты

---

### 3. Проверка результатов

После обработки всех корпусов:

```bash
# Общее количество JSONL файлов (предложения)
find data/processed -name "*.jsonl" -not -name "books_meta.jsonl" | wc -l

# Размер выходных данных
du -sh data/processed/

# Просмотр метаданных
head -5 data/processed/books_meta.jsonl | jq .

# Просмотр предложений одной книги
head -5 data/processed/*.jsonl | head -20
```

**Ожидаемые цифры:**
- Количество JSONL файлов ≈ количество обработанных книг (около 176k)
- Размер данных: ~50-70 GB
- Метаданные: `books_meta.jsonl` содержит записи для каждой книги

---

### Запуск анализатора
```bash
cd /mnt/work/audetv/go/src/github.com/terratensor/book2bert-v2

go run cmd/analyze-corpus/main.go \
    --dir data/processed/ \
    --output data/analysis/ \
    --workers 32
```
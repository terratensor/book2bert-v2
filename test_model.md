# По языкам (автоматически ru, en, mixed)
python test_model.py --model ... --mode language --index data/index/book_index.jsonl

# Расширенный контекст (3 предложения)
python test_model.py --model ... --mode extended

# Множественное маскирование
python test_model.py --model ... --mode multi --num-masks 3

# Один язык
python test_model.py --model ... --mode random --language en --num-examples 10

# Всё сразу
python test_model.py --model ... --mode all --index data/index/book_index.jsonl

```bash
python test_model.py     --model models/bert-base-ml-phase1_128/checkpoint-195000     --mode random   --cleaned data/processed/ --num-examples 10

```
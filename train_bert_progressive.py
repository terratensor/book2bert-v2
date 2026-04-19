#!/usr/bin/env python3
"""
Прогрессивное обучение BERT-base на двух GPU.

Архитектура:
- Фаза 1: 128 токенов, 400k шагов
- Фаза 2: 256 токенов, 400k шагов  
- Фаза 3: 512 токенов, 200k шагов

Запуск:
torchrun --nproc_per_node=2 train_bert_progressive.py
"""

import os
import sys
import logging
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    BertConfig,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from simple_sp_tokenizer import SimpleSPTokenizer

# ============================================================================
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ПОТОКОВЫЙ ДАТАСЕТ (не загружает всё в память)
# ============================================================================

class StreamingChunkDataset(IterableDataset):
    """
    Потоковое чтение чанков из текстового файла.
    
    Каждая строка файла = один чанк (несколько предложений, объединённых пробелами).
    При итерации читаем строку, токенизируем на лету, отдаём батч.
    
    ПАМЯТЬ: O(1) — в памяти только текущая строка.
    """
    
    def __init__(self, file_path, tokenizer, max_length, shuffle=False):
        self.file_path = file_path      # путь к .txt файлу
        self.tokenizer = tokenizer      # наш SimpleSPTokenizer
        self.max_length = max_length    # 128, 256 или 512
        self.shuffle = shuffle          # перемешивать ли строки
    
    def __len__(self):
        """Быстрый подсчёт строк через wc -l."""
        if not hasattr(self, '_len'):
            import subprocess
            result = subprocess.run(
                ['wc', '-l', self.file_path],
                capture_output=True, text=True
            )
            self._len = int(result.stdout.split()[0])
        return self._len
    
    def __iter__(self):
        """Главный метод: отдаёт батчи для Trainer."""
        # Открываем файл
        with open(self.file_path, 'r', encoding='utf-8') as f:
            # Читаем построчно
            for line in f:
                text = line.strip()
                if not text:
                    continue
                
                # ТОКЕНИЗАЦИЯ НА ЛЕТУ (занимает <1 мс)
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',  # паддинг до max_length
                    truncation=True,       # обрезаем, если длиннее
                    return_tensors='pt',   # возвращаем PyTorch тензоры
                )
                
                # Отдаём словарь с тензорами
                yield {
                    'input_ids': encoded['input_ids'][0],      # [max_length]
                    'attention_mask': encoded['attention_mask'][0],  # [max_length]
                }


# ============================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ ОДНОЙ ФАЗЫ
# ============================================================================

def train_phase(
    model,                  # модель BERT (одна и та же для всех фаз)
    tokenizer,              # токенизатор
    phase_name,             # "phase1", "phase2", "phase3"
    train_file,             # путь к train_*.txt
    val_file,               # путь к val_*.txt
    max_length,             # 128, 256 или 512
    batch_size,             # per_device_batch_size
    gradient_accumulation,  # сколько батчей накапливать
    max_steps,              # сколько шагов обучать
    learning_rate,          # скорость обучения
    save_steps=2000,        # как часто сохранять чекпоинты
):
    """
    Обучает модель одну фазу и сохраняет результат.
    
    ВАЖНО: модель НЕ пересоздаётся между фазами.
    Веса, полученные в фазе 1, используются как начальные для фазы 2.
    
    При падении обучения — автоматически продолжит с последнего чекпоинта.
    """
    
    # ========================================================================
    # ЛОГИРОВАНИЕ ПАРАМЕТРОВ ФАЗЫ
    # ========================================================================
    logger.info("=" * 80)
    logger.info(f"PHASE: {phase_name}")
    logger.info(f"  Train file: {train_file}")
    logger.info(f"  Val file:   {val_file}")
    logger.info(f"  Max length: {max_length}")
    logger.info(f"  Batch size: {batch_size} × {gradient_accumulation} × 2 GPUs = {batch_size * gradient_accumulation * 2}")
    logger.info(f"  Max steps:  {max_steps:,}")
    logger.info(f"  Save every: {save_steps:,} steps")
    logger.info(f"  LR:         {learning_rate}")
    logger.info("=" * 80)
    
    # ========================================================================
    # СОЗДАЁМ ПОТОКОВЫЕ ДАТАСЕТЫ (не загружают всё в память)
    # ========================================================================
    train_dataset = StreamingChunkDataset(train_file, tokenizer, max_length)
    val_dataset = StreamingChunkDataset(val_file, tokenizer, max_length)
    
    logger.info(f"Train dataset size: {len(train_dataset):,} chunks")
    logger.info(f"Val dataset size:   {len(val_dataset):,} chunks")
    
    # ========================================================================
    # DATA COLLATOR ДЛЯ MASKED LANGUAGE MODELING
    # ========================================================================
    # DataCollator берёт батч текстов, токенизирует их, и случайно маскирует 15% токенов.
    # Например: "Москва — столица России" -> "Москва — [MASK] России"
    # Модель должна предсказать "[MASK]" -> "столица"
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,               # Masked Language Modeling
        mlm_probability=0.15,   # 15% токенов маскируется
    )
    
    # ========================================================================
    # ПРОВЕРЯЕМ НАЛИЧИЕ ЧЕКПОИНТОВ ДЛЯ ВОССТАНОВЛЕНИЯ
    # ========================================================================
    output_dir = f"models/bert-base-ru-{phase_name}"
    resume_from_checkpoint = None
    
    if os.path.exists(output_dir):
        # Ищем все чекпоинты в директории
        import glob
        checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
        if checkpoints:
            # Берём последний (самый свежий) чекпоинт
            resume_from_checkpoint = checkpoints[-1]
            logger.info(f"🔄 Found checkpoint: {resume_from_checkpoint}")
            
            # Читаем состояние тренировки, чтобы узнать сколько шагов уже пройдено
            state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
            if os.path.exists(state_path):
                import json
                with open(state_path) as f:
                    state = json.load(f)
                steps_done = state.get('global_step', 0)
                epochs_done = state.get('epoch', 0)
                best_metric = state.get('best_metric', 'N/A')
                logger.info(f"  Already trained: {steps_done:,} steps ({steps_done/max_steps*100:.1f}%)")
                logger.info(f"  Epochs done: {epochs_done:.2f}")
                logger.info(f"  Best val loss: {best_metric}")
    
    # ========================================================================
    # АРГУМЕНТЫ ОБУЧЕНИЯ ДЛЯ HUGGINGFACE TRAINER
    # ========================================================================
    training_args = TrainingArguments(
        # Куда сохранять чекпоинты и логи
        output_dir=output_dir,
        
        # Восстановление с чекпоинта (если нашли — продолжит, если нет — начнёт с нуля)
        resume_from_checkpoint=resume_from_checkpoint,
        
        # Количество шагов (НЕ эпох!). Один шаг = один батч
        max_steps=max_steps,
        
        # Размер батча на одной GPU
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        
        # Накапливаем градиенты несколько шагов перед обновлением весов
        gradient_accumulation_steps=gradient_accumulation,
        
        # Как часто сохранять чекпоинты (в шагах)
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,      # хранить только 5 последних чекпоинтов
        
        # Как часто выводить логи в консоль и TensorBoard
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        
        # Валидация
        eval_strategy="steps",
        eval_steps=save_steps,
        
        # Оптимизатор и learning rate
        learning_rate=learning_rate,
        warmup_steps=int(max_steps * 0.01),
        weight_decay=0.01,
        
        # Смешанная точность (FP16)
        fp16=True,
        
        # Количество процессов для загрузки данных
        dataloader_num_workers=4,
        
        # Куда отправлять логи
        report_to="tensorboard",
        
        # Настройки для распределённого обучения
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )
    
    # ========================================================================
    # СОЗДАЁМ TRAINER
    # ========================================================================
    trainer = Trainer(
        model=model,                # модель BERT
        args=training_args,         # аргументы обучения
        data_collator=data_collator,  # коллатор для MLM
        train_dataset=train_dataset,  # тренировочный датасет
        eval_dataset=val_dataset,     # валидационный датасет
    )
    
    # ========================================================================
    # ЗАПУСК ОБУЧЕНИЯ
    # ========================================================================
    logger.info(f"🚀 Starting training for {max_steps:,} steps...")
    
    # Trainer.train() возвращает количество пройденных шагов
    # Если был чекпоинт — начнёт с него и дойдёт до max_steps
    train_result = trainer.train()
    
    logger.info(f"✅ Training completed! Total steps: {train_result.global_step:,}")
    logger.info(f"   Final loss: {train_result.training_loss:.4f}")
    
    # ========================================================================
    # СОХРАНЯЕМ ФИНАЛЬНУЮ МОДЕЛЬ ПОСЛЕ ФАЗЫ
    # ========================================================================
    final_output = f"models/bert-base-ru-{phase_name}-final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    
    logger.info(f"💾 Model saved to {final_output}")
    logger.info("=" * 80)
    
    return trainer

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Прогрессивное обучение BERT-base."""
    
    # Проверяем GPU
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # ========================================================================
    # 1. ЗАГРУЖАЕМ ТОКЕНИЗАТОР
    # ========================================================================
    tokenizer_path = "models/tokenizer/final/sp_100k.model"
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, PAD={tokenizer.pad_token_id}")
    
    # ========================================================================
    # 2. СОЗДАЁМ МОДЕЛЬ BERT-base (110M параметров)
    # ========================================================================
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,  # 100,000
        hidden_size=768,                   # размер эмбеддингов
        num_hidden_layers=12,              # количество слоёв Transformer
        num_attention_heads=12,            # голов внимания
        intermediate_size=3072,            # размер FFN слоя (4 × hidden_size)
        max_position_embeddings=512,       # ВАЖНО: СРАЗУ 512 для всех фаз!
        type_vocab_size=2,                 # для token_type_ids (не используется)
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    model = BertForMaskedLM(config)
    
    # Считаем параметры
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {params:,} total, {trainable_params:,} trainable")
    
    # Сохраняем конфиг
    config.save_pretrained("models/bert-base-ru-config")
    logger.info("Config saved to models/bert-base-ru-config")
    
    # ========================================================================
    # 3. ФАЗА 1: 128 ТОКЕНОВ (400k шагов)
    # ========================================================================
    # Используем 98% данных из phase1 (разные книги от phase2 и phase3)
    train_phase(
        model=model,
        tokenizer=tokenizer,
        phase_name="phase1_128",
        train_file="data/bert/phase1_128_train.txt",
        val_file="data/bert/phase1_128_val.txt",
        max_length=128,
        batch_size=32,               # 32 на GPU × 2 GPU = 64
        gradient_accumulation=2,     # эффективный batch = 64 × 2 = 128
        max_steps=400_000,
        learning_rate=5e-4,          # высокий LR для начального обучения
    )
    
    # ========================================================================
    # 4. ФАЗА 2: 256 ТОКЕНОВ (400k шагов)
    # ========================================================================
    # Используем ДРУГИЕ книги (phase2), чтобы модель видела новый текст
    train_phase(
        model=model,
        tokenizer=tokenizer,
        phase_name="phase2_256",
        train_file="data/bert/phase2_256_train.txt",
        val_file="data/bert/phase2_256_val.txt",
        max_length=256,
        batch_size=16,               # уменьшаем батч (длиннее последовательности)
        gradient_accumulation=4,     # эффективный batch = 16 × 4 × 2 = 128
        max_steps=400_000,
        learning_rate=3e-4,          # снижаем LR
    )
    
    # ========================================================================
    # 5. ФАЗА 3: 512 ТОКЕНОВ (200k шагов)
    # ========================================================================
    # Используем третью часть книг (phase3)
    train_phase(
        model=model,
        tokenizer=tokenizer,
        phase_name="phase3_512",
        train_file="data/bert/phase3_512_train.txt",
        val_file="data/bert/phase3_512_val.txt",
        max_length=512,
        batch_size=8,                # ещё меньше
        gradient_accumulation=8,     # эффективный batch = 8 × 8 × 2 = 128
        max_steps=200_000,
        learning_rate=1e-4,          # минимальный LR для тонкой настройки
    )
    
    # ========================================================================
    # 6. СОХРАНЯЕМ ФИНАЛЬНУЮ МОДЕЛЬ
    # ========================================================================
    final_output = "models/bert-base-ru-final"
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info("=" * 80)
    logger.info(f"FINAL MODEL SAVED TO {final_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
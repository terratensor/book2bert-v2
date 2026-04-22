#!/usr/bin/env python3
"""
Прогрессивное обучение BERT-base на двух GPU.

Архитектура:
- Фаза 1: 128 токенов, 400k шагов
- Фаза 2: 256 токенов, 400k шагов  
- Фаза 3: 512 токенов, 200k шагов

Валидация:
- Быстрая: 20k примеров каждые 5000 шагов (~2 мин)
- Полная: весь val-датасет в конце каждой фазы (~2 часа)

Запуск:
torchrun --nproc_per_node=2 train_bert_progressive.py 2>&1 | tee phase_training.log
"""

import os
import sys
import logging
import glob
import json
import time
import torch
from torch.utils.data import IterableDataset
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
    
    Каждая строка файла = один чанк.
    ПАМЯТЬ: O(1) — в памяти только текущая строка.
    """
    
    def __init__(self, file_path, tokenizer, max_length, max_examples=None):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_examples = max_examples
        self._len = None
    
    def __len__(self):
        """Возвращает РЕАЛЬНОЕ количество примеров с учётом max_examples."""
        if self._len is None:
            import subprocess
            result = subprocess.run(
                ['wc', '-l', self.file_path],
                capture_output=True, text=True
            )
            total = int(result.stdout.split()[0])
            if self.max_examples:
                total = min(total, self.max_examples)
            self._len = total
        return self._len
    
    def __iter__(self):
        """Главный метод: отдаёт батчи для Trainer."""
        count = 0
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if self.max_examples and count >= self.max_examples:
                    break
                
                text = line.strip()
                if not text:
                    continue
                
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt',
                )
                
                yield {
                    'input_ids': encoded['input_ids'][0],
                    'attention_mask': encoded['attention_mask'][0],
                }
                count += 1


# ============================================================================
# ПОЛНАЯ ВАЛИДАЦИЯ (выполняется один раз в конце фазы)
# ============================================================================

def full_validation(model, tokenizer, val_file, max_length, phase_name, batch_size=32):
    """
    Полная валидация на ВСЁМ val-датасете (1.1M примеров).
    Занимает ~2 часа.
    """
    logger.info("=" * 80)
    logger.info(f"🔍 FULL VALIDATION for {phase_name}")
    logger.info(f"   Val file: {val_file}")
    logger.info(f"   Max length: {max_length}")
    logger.info("=" * 80)
    
    # Полный датасет без ограничений
    full_val_dataset = StreamingChunkDataset(val_file, tokenizer, max_length)
    logger.info(f"   Full val size: {len(full_val_dataset):,} chunks")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Trainer только для валидации (без логирования, без прогресс-бара)
    val_args = TrainingArguments(
        output_dir=f"models/bert-base-ru-{phase_name}-full-val",
        per_device_eval_batch_size=batch_size,
        fp16=True,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        report_to=[],       # без TensorBoard
        # disable_tqdm=True,  # без прогресс-бара
    )
    
    trainer = Trainer(
        model=model,
        args=val_args,
        data_collator=data_collator,
    )
    
    start_time = time.time()
    metrics = trainer.evaluate(eval_dataset=full_val_dataset)
    elapsed = time.time() - start_time
    
    perplexity = torch.exp(torch.tensor(metrics['eval_loss']))
    
    logger.info(f"   ✅ Full validation done in {elapsed/60:.1f} min")
    logger.info(f"   Loss: {metrics['eval_loss']:.4f}")
    logger.info(f"   Perplexity: {perplexity:.2f}")
    logger.info("=" * 80)
    
    return metrics


# ============================================================================
# ФУНКЦИЯ ОБУЧЕНИЯ ОДНОЙ ФАЗЫ
# ============================================================================

def train_phase(
    model,
    tokenizer,
    phase_name,
    train_file,
    val_file,
    max_length,
    batch_size,
    gradient_accumulation,
    max_steps,
    learning_rate,
    save_steps=5000,
    val_max_examples=20000,
    do_full_validation=True,
):
    """
    Обучает модель одну фазу и сохраняет результат.
    
    ВАЖНО: модель НЕ пересоздаётся между фазами.
    Веса, полученные в фазе 1, используются как начальные для фазы 2.
    """
    
    logger.info("=" * 80)
    logger.info(f"🚀 PHASE: {phase_name}")
    logger.info(f"   Train file: {train_file}")
    logger.info(f"   Val file:   {val_file}")
    logger.info(f"   Max length: {max_length}")
    logger.info(f"   Batch size: {batch_size} × {gradient_accumulation} × 2 GPUs = {batch_size * gradient_accumulation * 2}")
    logger.info(f"   Max steps:  {max_steps:,}")
    logger.info(f"   Save every: {save_steps:,} steps")
    logger.info(f"   Fast val:   {val_max_examples:,} samples")
    logger.info(f"   Full val:   {'Yes' if do_full_validation else 'No'}")
    logger.info(f"   LR:         {learning_rate}")
    logger.info("=" * 80)
    
    # Датасеты
    train_dataset = StreamingChunkDataset(train_file, tokenizer, max_length)
    val_dataset = StreamingChunkDataset(
        val_file, tokenizer, max_length, max_examples=val_max_examples
    )
    
    logger.info(f"Train dataset size: {len(train_dataset):,} chunks")
    logger.info(f"Fast val size:      {len(val_dataset):,} chunks")
    
    # DataCollator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Проверяем чекпоинты
    output_dir = f"models/bert-base-ru-{phase_name}"
    resume_from_checkpoint = None
    
    if os.path.exists(output_dir):
        checkpoints = sorted(glob.glob(f"{output_dir}/checkpoint-*"))
        if checkpoints:
            resume_from_checkpoint = checkpoints[-1]
            logger.info(f"🔄 Found checkpoint: {resume_from_checkpoint}")
            
            state_path = os.path.join(resume_from_checkpoint, "trainer_state.json")
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state = json.load(f)
                steps_done = state.get('global_step', 0)
                logger.info(f"   Already trained: {steps_done:,} steps ({steps_done/max_steps*100:.1f}%)")
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        
        # Количество шагов
        max_steps=max_steps,
        
        # Размер батча
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        
        # Сохранение
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        
        # Логирование
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,        
        eval_strategy="steps",
        eval_steps=save_steps,
        
        max_grad_norm=1.0,  # ← ДОБАВИТЬ
        # lr_scheduler_type="cosine",  # ← ДОБАВИТЬ (вместо linear)
        lr_scheduler_type="constant",  # ← ДОБАВИТЬ (вместо linear)

        # Оптимизатор
        learning_rate=learning_rate,
        warmup_steps=int(max_steps * 0.01),
        weight_decay=0.01,
        
        # Смешанная точность
        fp16=True,
        
        # Данные
        dataloader_num_workers=4,
        
        # Логи
        report_to="tensorboard",
        
        # Прогресс-бар
        disable_tqdm=False,
        
        # Multi-GPU
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    logger.info(f"🎯 Starting training for {max_steps:,} steps...")
    
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    logger.info(f"✅ Training completed! Total steps: {train_result.global_step:,}")
    logger.info(f"   Final training loss: {train_result.training_loss:.4f}")
    
    # ПОЛНАЯ ВАЛИДАЦИЯ В КОНЦЕ ФАЗЫ
    if do_full_validation:
        full_validation(model, tokenizer, val_file, max_length, phase_name, batch_size)
    
    # Сохраняем модель
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
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Загружаем токенизатор
    tokenizer_path = "models/tokenizer/final/sp_100k.model"
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, PAD={tokenizer.pad_token_id}")
    
    # Создаём модель BERT-base (110M параметров)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,  # Сразу 512 для всех фаз!
        type_vocab_size=2,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    model = BertForMaskedLM(config)
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,} total")
    
    config.save_pretrained("models/bert-base-ru-config")
    logger.info("Config saved to models/bert-base-ru-config")
    
    # ========================================================================
    # ФАЗА 1: 128 токенов, 400k шагов
    # ========================================================================
    train_phase(
        model=model,
        tokenizer=tokenizer,
        phase_name="phase1_128",
        train_file="data/bert/phase1_128_train.txt",
        val_file="data/bert/phase1_128_val.txt",
        max_length=128,
        batch_size=32,
        gradient_accumulation=2,
        max_steps=400_000,
        learning_rate=1e-4,
        save_steps=5000,
        val_max_examples=20000,
        do_full_validation=True,
    )
    
    # ========================================================================
    # ФАЗА 2: 256 токенов, 400k шагов
    # ========================================================================
    train_phase(
        model=model,
        tokenizer=tokenizer,
        phase_name="phase2_256",
        train_file="data/bert/phase2_256_train.txt",
        val_file="data/bert/phase2_256_val.txt",
        max_length=256,
        batch_size=16,
        gradient_accumulation=4,
        max_steps=400_000,
        learning_rate=5e-5,
        save_steps=5000,
        val_max_examples=20000,
        do_full_validation=True,
    )
    
    # ========================================================================
    # ФАЗА 3: 512 токенов, 200k шагов
    # ========================================================================
    train_phase(
        model=model,
        tokenizer=tokenizer,
        phase_name="phase3_512",
        train_file="data/bert/phase3_512_train.txt",
        val_file="data/bert/phase3_512_val.txt",
        max_length=512,
        batch_size=8,
        gradient_accumulation=8,
        max_steps=200_000,
        learning_rate=2e-5,
        save_steps=5000,
        val_max_examples=20000,
        do_full_validation=True,
    )
    
    # ========================================================================
    # ФИНАЛЬНОЕ СОХРАНЕНИЕ
    # ========================================================================
    final_output = "models/bert-base-ru-final"
    model.save_pretrained(final_output)
    tokenizer.save_pretrained(final_output)
    logger.info("=" * 80)
    logger.info(f"🎉 FINAL MODEL SAVED TO {final_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
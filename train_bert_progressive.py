#!/usr/bin/env python3
"""
Прогрессивное обучение BERT-base на двух GPU.

Архитектура:
- Фаза 1: 128 токенов, 400k шагов
- Фаза 2: 256 токенов, 400k шагов  
- Фаза 3: 512 токенов, 200k шагов

Запуск:
  # Все фазы подряд (не рекомендуется — долго)
  torchrun --nproc_per_node=2 train_bert_progressive.py

  # Отдельная фаза
  torchrun --nproc_per_node=2 train_bert_progressive.py --phase 1
  torchrun --nproc_per_node=2 train_bert_progressive.py --phase 2
  torchrun --nproc_per_node=2 train_bert_progressive.py --phase 3
"""

import os
import sys
import logging
import glob
import json
import time
import argparse
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
# КОНФИГУРАЦИИ ФАЗ
# ============================================================================

PHASE_CONFIGS = {
    1: {
        "phase_name": "phase1_128",
        "train_file": "data/bert_multilingual/phase1_128_train.txt",
        "val_file": "data/bert_multilingual/phase1_128_val.txt",
        "max_length": 128,
        "batch_size": 128,
        "gradient_accumulation": 1,
        "max_steps": 400_000,
        "learning_rate": 1e-4 ,
        "load_from": None,  # фаза 1 — с нуля
    },
    2: {
        "phase_name": "phase2_256",
        "train_file": "data/bert_multilingual/phase2_256_train.txt",
        "val_file": "data/bert_multilingual/phase2_256_val.txt",
        "max_length": 256,
        "batch_size": 64,
        "gradient_accumulation": 2,
        "max_steps": 400_000,
        "learning_rate": 5e-5,
        "load_from": "models/bert-base-ml-phase1_128-final",  # загружаем веса фазы 1
    },
    3: {
        "phase_name": "phase3_512",
        "train_file": "data/bert_multilingual/phase3_512_train.txt",
        "val_file": "data/bert_multilingual/phase3_512_val.txt",
        "max_length": 512,
        "batch_size": 32,
        "gradient_accumulation": 4,
        "max_steps": 200_000,
        "learning_rate": 2e-5,
        "load_from": "models/bert-base-ml-phase2_256-final",  # загружаем веса фазы 2
    },
}

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
    Полная валидация на ВСЁМ val-датасете.
    Занимает ~2 часа.
    """
    logger.info("=" * 80)
    logger.info(f"🔍 FULL VALIDATION for {phase_name}")
    logger.info(f"   Val file: {val_file}")
    logger.info(f"   Max length: {max_length}")
    logger.info("=" * 80)
    
    full_val_dataset = StreamingChunkDataset(val_file, tokenizer, max_length)
    logger.info(f"   Full val size: {len(full_val_dataset):,} chunks")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    val_args = TrainingArguments(
        output_dir=f"models/bert-base-ml-{phase_name}-full-val",
        per_device_eval_batch_size=batch_size,
        fp16=True,
        ddp_find_unused_parameters=False,
        ddp_backend="nccl",
        report_to=[],
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
    
    # Проверяем чекпоинты для восстановления внутри фазы
    output_dir = f"models/bert-base-ml-{phase_name}"
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

        average_tokens_across_devices=True, 
        
        # Сохранение
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=12,
        
        # Логирование
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,        
        eval_strategy="steps",
        eval_steps=save_steps,
        
        max_grad_norm=1.0,
        lr_scheduler_type="linear",
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Оптимизатор
        learning_rate=learning_rate,
        warmup_steps=int(max_steps * 0.01),
        weight_decay=0.01,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.98,
        
        # Смешанная точность
        fp16=True,
        
        # Данные
        dataloader_num_workers=8,
        
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
    final_output = f"models/bert-base-ml-{phase_name}-final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    
    logger.info(f"💾 Model saved to {final_output}")
    logger.info("=" * 80)
    
    return trainer


# ============================================================================
# ЗАГРУЗКА МОДЕЛИ ИЗ ПРЕДЫДУЩЕЙ ФАЗЫ
# ============================================================================

def load_model_from_phase(load_path, tokenizer):
    """
    Загружает веса модели из финального чекпоинта предыдущей фазы.
    Если путь не существует — создаёт модель с нуля.
    """
    if load_path and os.path.exists(load_path):
        logger.info(f"📥 Loading model weights from {load_path}")
        model = BertForMaskedLM.from_pretrained(load_path)
        logger.info("   Weights loaded successfully")
    else:
        if load_path:
            logger.warning(f"⚠️ Model path not found: {load_path}")
            logger.warning("   Creating model from scratch")
        else:
            logger.info("🆕 Creating model from scratch (Phase 1)")
        
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=512,
            type_vocab_size=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        
        model = BertForMaskedLM(config)
        config.save_pretrained("models/bert-base-ml-config")
        logger.info("Config saved to models/bert-base-ml-config")
    
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,} total")
    
    return model


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Прогрессивное обучение BERT-base")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                       help="Номер фазы для запуска (0 = все фазы подряд)")
    args = parser.parse_args()
    
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Загружаем токенизатор
    tokenizer_path = "models/tokenizer/multilingual/sp_42k.model"
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = SimpleSPTokenizer(tokenizer_path)
    logger.info(f"Vocab size: {tokenizer.vocab_size}")
    logger.info(f"Special tokens: CLS={tokenizer.cls_token_id}, SEP={tokenizer.sep_token_id}, PAD={tokenizer.pad_token_id}")
    
    # Определяем, какие фазы запускать
    if args.phase == 0:
        phases_to_run = [1, 2, 3]
        logger.info("🔄 Running ALL phases sequentially")
    else:
        phases_to_run = [args.phase]
        logger.info(f"🔄 Running only Phase {args.phase}")
    
    # Загружаем модель (с нуля или из предыдущей фазы)
    first_phase = phases_to_run[0]
    load_from = PHASE_CONFIGS[first_phase]["load_from"]
    model = load_model_from_phase(load_from, tokenizer)
    
    # Запускаем фазы
    for phase_num in phases_to_run:
        config = PHASE_CONFIGS[phase_num]
        
        logger.info(f"\n{'=' * 80}")
        logger.info(f"STARTING PHASE {phase_num}")
        logger.info(f"{'=' * 80}")
        
        train_phase(
            model=model,
            tokenizer=tokenizer,
            phase_name=config["phase_name"],
            train_file=config["train_file"],
            val_file=config["val_file"],
            max_length=config["max_length"],
            batch_size=config["batch_size"],
            gradient_accumulation=config["gradient_accumulation"],
            max_steps=config["max_steps"],
            learning_rate=config["learning_rate"],
            save_steps=5000,
            val_max_examples=50000,
            do_full_validation=True,
        )
    
    # Финальное сохранение
    if len(phases_to_run) > 1 or phases_to_run[-1] == 3:
        final_output = "models/bert-base--final"
        model.save_pretrained(final_output)
        tokenizer.save_pretrained(final_output)
        logger.info("=" * 80)
        logger.info(f"🎉 FINAL MODEL SAVED TO {final_output}")
        logger.info("=" * 80)


if __name__ == "__main__":
    main()
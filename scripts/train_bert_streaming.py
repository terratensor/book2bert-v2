# train_bert_streaming.py
import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import (
    BertConfig, BertForMaskedLM, Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from simple_sp_tokenizer import SimpleSPTokenizer
import math

class StreamingChunkDataset(IterableDataset):
    """Потоковое чтение чанков из текстового файла."""
    
    def __init__(self, file_path, tokenizer, max_length=128, 
                 infinite=False, shuffle_buffer=10000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.infinite = infinite
        self.shuffle_buffer = shuffle_buffer
        
        # Считаем количество строк для прогресса
        self._len = None
    
    def __len__(self):
        if self._len is None:
            # Быстрый подсчет строк через wc -l
            import subprocess
            result = subprocess.run(['wc', '-l', self.file_path], 
                                    capture_output=True, text=True)
            self._len = int(result.stdout.split()[0])
        return self._len
    
    def __iter__(self):
        # Буфер для перемешивания
        buffer = []
        
        while True:
            with open(self.file_path, 'r') as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    
                    # Токенизируем на лету
                    encoded = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        padding='max_length',
                        truncation=True,
                    )
                    
                    example = {
                        'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
                        'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
                    }
                    
                    if self.shuffle_buffer > 0:
                        buffer.append(example)
                        if len(buffer) >= self.shuffle_buffer:
                            # Случайный элемент из буфера
                            idx = torch.randint(0, len(buffer), (1,)).item()
                            yield buffer.pop(idx)
                    else:
                        yield example
                
                # Отдаем оставшиеся в буфере
                if self.shuffle_buffer > 0:
                    while buffer:
                        idx = torch.randint(0, len(buffer), (1,)).item()
                        yield buffer.pop(idx)
            
            if not self.infinite:
                break


def main():
    # Токенизатор
    tokenizer = SimpleSPTokenizer("models/tokenizer/final/sp_100k.model")
    
    # Конфиг BERT-base
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,  # Сразу 512 для будущих фаз
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    
    model = BertForMaskedLM(config)
    print(f"Parameters: {model.num_parameters() / 1e6:.1f}M")
    
    # Датасеты
    train_dataset = StreamingChunkDataset(
        "data/bert/temp_chunks_7p3byzgn/train_128.txt",
        tokenizer,
        max_length=128,
        infinite=True,
        shuffle_buffer=10000,
    )
    
    # Для валидации - маленький файл или первые N строк
    val_dataset = StreamingChunkDataset(
        "data/bert/temp_chunks_7p3byzgn/train_128.txt",
        tokenizer,
        max_length=128,
        infinite=False,
        shuffle_buffer=0,
    )
    
    # Коллатор для MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    # Аргументы обучения
    training_args = TrainingArguments(
        output_dir="models/bert-base-ru-phase1",
        overwrite_output_dir=True,
        max_steps=400_000,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        logging_steps=100,
        save_steps=50000,
        eval_steps=10000,
        learning_rate=5e-4,
        warmup_steps=10000,
        weight_decay=0.01,
        fp16=True,
        dataloader_num_workers=4,
        report_to="tensorboard",
        # Для IterableDataset нужно указать max_steps, а не epochs
        max_steps=400_000,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    trainer.save_model("models/bert-base-ru-phase1-final")


if __name__ == "__main__":
    main()
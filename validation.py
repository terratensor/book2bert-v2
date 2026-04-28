import torch
import time
import os
from transformers import BertForMaskedLM, DataCollatorForLanguageModeling
from simple_sp_tokenizer import SimpleSPTokenizer
from train_bert_progressive import StreamingChunkDataset

# Загружаем модель
print("Loading model...")
model = BertForMaskedLM.from_pretrained("models/bert-base-ru-phase1_128/checkpoint-255000")
tokenizer = SimpleSPTokenizer("models/tokenizer/final/32k/sp_32k.model")

# DataParallel на обе карты
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs: DataParallel")
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    params = sum(p.numel() for p in model.module.parameters())
elif torch.cuda.is_available():
    model = model.cuda()
    params = sum(p.numel() for p in model.parameters())
else:
    params = sum(p.numel() for p in model.parameters())

print(f"Model loaded. Parameters: {params:,}")

# Полный val-датасет
val_file = "data/bert_full/phase1_128_val.txt"
print(f"Loading val dataset from {val_file}...")
val_dataset = StreamingChunkDataset(val_file, tokenizer, max_length=128)
print(f"Val dataset size: {len(val_dataset):,} chunks")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Evaluation loop
model.eval()
total_loss = 0
total_steps = 0

print("Starting full validation on 2 GPUs...")
start_time = time.time()

dataloader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256,
    collate_fn=data_collator,
    num_workers=8,
)

with torch.no_grad():
    for batch in dataloader:
        if torch.cuda.is_available():
            batch = {k: v.cuda() for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        if isinstance(loss, torch.Tensor) and loss.numel() > 1:
            loss = loss.mean()
        
        total_loss += loss.item()
        total_steps += 1
        
        if total_steps % 200 == 0:
            elapsed = time.time() - start_time
            print(f"  Step {total_steps}/{len(dataloader)} ({total_steps/len(dataloader)*100:.1f}%), "
                  f"avg loss: {total_loss/total_steps:.4f}, elapsed: {elapsed/60:.1f} min")

elapsed = time.time() - start_time
avg_loss = total_loss / total_steps
perplexity = torch.exp(torch.tensor(avg_loss))

# Вывод и сохранение
result = f"""
{'='*60}
FULL VALIDATION RESULTS (Phase 1, checkpoint 255000)
{'='*60}
GPUs used:    {torch.cuda.device_count()}
Val file:     {val_file}
Val size:     {len(val_dataset):,} chunks
Val loss:     {avg_loss:.4f}
Perplexity:   {perplexity:.2f}
Time:         {elapsed/60:.1f} minutes
{'='*60}
"""

print(result)

with open("phase1_full_validation.txt", "w") as f:
    f.write(result)

print("Results saved to phase1_full_validation.txt")
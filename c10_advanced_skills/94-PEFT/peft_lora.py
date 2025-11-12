# -*- coding: utf-8 -*-
# @Time    : 2025/4/8 17:27
# @Author  : CFuShn
# @Comments: 
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
import deepspeed
import os

# 1. 加载预训练的BERT模型和Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 2. 配置LoRA：在多个层上应用LoRA
lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["attention.self.query", "attention.self.key", "output.dense",
                        "intermediate.dense"],
        bias="none",
        task_type="SEQUENCE_CLASSIFICATION"
)

# 3. 将LoRA集成到BERT模型中
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()  # 输出可训练的参数数量

# 4. 加载IMDB数据集
dataset = load_dataset("imdb")
small_train_dataset = dataset['train'].shuffle(seed=42).select(range(1000))  # 使用1000条数据
small_test_dataset = dataset['test'].shuffle(seed=42).select(range(200))  # 使用200条数据


# 数据预处理函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)


# 预处理数据集
encoded_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
encoded_test_dataset = small_test_dataset.map(tokenize_function, batched=True)

# 将数据集的格式转换为PyTorch张量
encoded_train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
encoded_test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# 5. 数据加载器
from torch.utils.data import DataLoader

train_dataloader = DataLoader(encoded_train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(encoded_test_dataset, batch_size=16)

# 6. DeepSpeed配置
ds_config = {
    "train_batch_size": 16,
    "fp16": {
        "enabled": True  # 启用FP16混合精度训练
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3
        "offload_optimizer": {
            "device": "cpu",  # 将优化器状态卸载到CPU，节省显存
        }
    },
    "checkpoint": {
        "save_dir": "./checkpoints",
        "save_interval": 500  # 每500个step保存一次
    }
}

# 7. 使用DeepSpeed包装模型和优化器
peft_model, optimizer, _, _ = deepspeed.initialize(
        model=peft_model,
        model_parameters=peft_model.parameters(),
        config=ds_config
)

# 8. 使用学习率调度器
num_training_steps = len(train_dataloader) * 3  # 3个epoch
scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=num_training_steps
)

# 9. 训练和验证循环
from tqdm import tqdm

epochs = 3
best_accuracy = 0.0

for epoch in range(epochs):
    # 训练阶段
    peft_model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")):
        input_ids = batch["input_ids"].to(peft_model.device)
        attention_mask = batch["attention_mask"].to(peft_model.device)
        labels = batch["label"].to(peft_model.device)

        outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        peft_model.backward(loss)
        peft_model.step()
        scheduler.step()

        total_loss += loss.item()

        if step % 100 == 0:
            print(f"Step {step} - Loss: {loss.item():.4f}")

    print(f"Epoch {epoch + 1} - Training loss: {total_loss / len(train_dataloader):.4f}")

    # 验证阶段
    peft_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluation"):
            input_ids = batch["input_ids"].to(peft_model.device)
            attention_mask = batch["attention_mask"].to(peft_model.device)
            labels = batch["label"].to(peft_model.device)

            outputs = peft_model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Epoch {epoch + 1} - Validation Accuracy: {accuracy:.2f}%")

    # 保存最佳模型
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        os.makedirs("./checkpoints", exist_ok=True)
        peft_model.save_pretrained("./checkpoints/best_model")
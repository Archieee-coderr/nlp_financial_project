import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def run_bert_finetuning(sentences, labels, model_name="ProsusAI/finbert"):

    # 1️⃣ 转 HuggingFace Dataset
    dataset = Dataset.from_dict({
        "text": sentences,
        "label": labels
    })

    dataset = dataset.train_test_split(test_size=0.2)

    # 2️⃣ tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(example):
        return tokenizer(
            example["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    dataset = dataset.map(tokenize, batched=True)

    # 3️⃣ 加载模型（关键！）
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3   # negative, neutral, positive
    )

    # 4️⃣ 训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        logging_dir="./logs",
        load_best_model_at_end=True,
    )

    # 5️⃣ Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 6️⃣ 开始训练
    trainer.train()

    # 7️⃣ 评估
    results = trainer.evaluate()

    print("\n===== BERT Fine-tuning Results =====")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

    return results
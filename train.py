"""
Fine-tune DistilBERT for multi-label news article topic classification.

Usage:
    python train.py --dataset ContextNews/articles --split 2026_02_15
    python train.py --dataset ContextNews/articles --split 2026_02_15 --epochs 5 --batch-size 16
    python train.py --dataset ContextNews/articles --split 2026_02_15 --push-to myorg/my-model

Environment variables:
    HF_TOKEN - HuggingFace token with write access (for pushing model)
"""

import argparse

import numpy as np
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

load_dotenv()

BASE_MODEL = "distilbert-base-uncased"

TOPICS = [
    "politics",
    "geopolitics",
    "conflict",
    "crime",
    "law",
    "business",
    "economy",
    "markets",
    "technology",
    "science",
    "health",
    "environment",
    "society",
    "education",
    "sports",
    "entertainment",
]


def build_input_text(row: dict) -> str:
    title = row.get("title") or ""
    summary = row.get("summary") or ""
    text = row.get("text") or ""
    # Take first 300 words of text to match classification input
    text_words = text.split()[:300]
    text_excerpt = " ".join(text_words)

    parts = []
    if title:
        parts.append(title)
    if summary:
        parts.append(summary)
    if text_excerpt:
        parts.append(text_excerpt)

    return " ".join(parts)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="micro", zero_division=0),
        "recall": recall_score(labels, preds, average="micro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for multi-label topic classification"
    )
    parser.add_argument(
        "--dataset", required=True, help="HuggingFace dataset ID (e.g. ContextNews/articles)"
    )
    parser.add_argument(
        "--split", required=True, help="Dataset split to use (e.g. 2026_02_15)"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)"
    )
    parser.add_argument(
        "--val-split", default=None,
        help="Validation split name (e.g. 'validation'). If not provided, "
             "10%% of the training split is held out automatically.",
    )
    parser.add_argument(
        "--output-dir", default="./model_output",
        help="Local directory for checkpoints (default: ./model_output)",
    )
    parser.add_argument(
        "--push-to", default=None,
        help="HuggingFace repo to push the trained model to (e.g. myorg/my-model)",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset {args.dataset} (split={args.split})...")
    ds = load_dataset(args.dataset, split=args.split)

    # Filter to only rows that have been classified (at least one topic is not null)
    def is_labeled(row):
        return any(row[t] is not None for t in TOPICS)

    ds = ds.filter(is_labeled)
    print(f"Found {len(ds)} labeled rows.")

    if len(ds) == 0:
        print("No labeled data found. Run classify.py first.")
        return

    # Build labels and input text
    def preprocess(row):
        row["input_text"] = build_input_text(row)
        row["labels"] = [float(row[t] or 0) for t in TOPICS]
        return row

    ds = ds.map(preprocess)

    # Train/val split
    if args.val_split:
        print(f"Loading validation split '{args.val_split}'...")
        val_ds = load_dataset(args.dataset, split=args.val_split)
        val_ds = val_ds.filter(is_labeled)
        val_ds = val_ds.map(preprocess)
        train_ds = ds
    else:
        split = ds.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        val_ds = split["test"]
    print(f"Train: {len(train_ds)} rows, Val: {len(val_ds)} rows.")

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(batch):
        encoding = tokenizer(
            batch["input_text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        encoding["labels"] = batch["labels"]
        return encoding

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=val_ds.column_names)

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(TOPICS),
        problem_type="multi_label_classification",
        id2label={i: t for i, t in enumerate(TOPICS)},
        label2id={t: i for i, t in enumerate(TOPICS)},
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        logging_steps=50,
        push_to_hub=args.push_to is not None,
        hub_model_id=args.push_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Evaluate
    print("\nFinal evaluation:")
    metrics = trainer.evaluate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save locally
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")

    # Push to hub
    if args.push_to:
        print(f"Pushing model to {args.push_to}...")
        trainer.push_to_hub()
        tokenizer.push_to_hub(args.push_to)
        print("Done.")


if __name__ == "__main__":
    main()

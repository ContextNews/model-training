# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-label news article topic classification pipeline for the **ContextNews** project. Articles are classified into 16 topics (politics, geopolitics, conflict, crime, law, business, economy, markets, technology, science, health, environment, society, education, sports, entertainment). Articles can belong to multiple topics simultaneously.

## Pipeline

The workflow is a three-stage pipeline, run in order:

1. **classify.py** — Labels raw articles from `ContextNews/articles` on HuggingFace using the OpenAI API (gpt-5 with structured output). Classifies async with concurrency control, then pushes updated dataset back to HuggingFace.
2. **create_labelled_dataset.py** — Filters labelled rows, shuffles (seed=42), splits 80/10/10 into train/validation/test, and pushes to `ContextNews/labelled_articles`.
3. **train.py** — Fine-tunes `distilbert-base-uncased` for multi-label classification using HuggingFace Transformers `Trainer`. Saves to `./model_output/` and optionally pushes to HuggingFace Hub.

`train_notebook.ipynb` is a Colab-oriented version of stage 3 that trains against `ContextNews/labelled_articles` and pushes to `ContextNews/news-classifier`. Requires a T4 GPU runtime.

## Commands

```bash
# Install dependencies (uses Poetry)
poetry install

# Stage 1: Classify articles with OpenAI
python classify.py                  # default: 1000 rows, concurrency 10
python classify.py -n 500 -c 20    # custom row limit and concurrency

# Stage 2: Create labelled dataset splits
python create_labelled_dataset.py

# Stage 3: Train model
python train.py --dataset ContextNews/articles --split 2026_02_15
python train.py --dataset ContextNews/articles --split 2026_02_15 --epochs 5 --batch-size 16 --push-to ContextNews/news-classifier
```

## Environment Variables

Requires a `.env` file (gitignored) with:
- `OPENAI_API_KEY` — for classify.py
- `HF_TOKEN` — HuggingFace token with write access (for pushing datasets/models)

## Key Design Decisions

- Input text is built from article title + summary + first 300 words of body text (truncated) — consistent across both classification and training.
- Classification uses OpenAI structured output (`json_schema` response format) with strict mode to guarantee valid topic scores (0 or 1 per topic).
- Training uses `problem_type="multi_label_classification"` with sigmoid thresholding at 0.5. Metrics: f1_micro (primary for model selection), f1_macro, precision, recall.
- The TOPICS list (16 items) is duplicated across all scripts — changes must be kept in sync.

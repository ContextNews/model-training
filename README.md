# model-training

Training pipeline for the [ContextNews](https://huggingface.co/ContextNews) multi-label news topic classifier. Articles are classified into 16 topics using GPT-5, then a DistilBERT model is fine-tuned on the labelled data.

## Links

- **Model:** [ContextNews/news-classifier](https://huggingface.co/ContextNews/news-classifier)
- **Labelled dataset:** [ContextNews/labelled_articles](https://huggingface.co/datasets/ContextNews/labelled_articles)
- **Raw dataset:** [ContextNews/articles](https://huggingface.co/datasets/ContextNews/articles)

## Structure

```
scripts/
  classify.py                 # Label articles via OpenAI API
  create_labelled_dataset.py  # Split labelled data into train/val/test
  train.py                    # Fine-tune DistilBERT locally

notebooks/
  train_notebook.ipynb        # Colab training notebook (GPU)
  threshold_tuning.ipynb      # Per-class threshold optimisation
```

## Setup

```bash
poetry install
```

Requires a `.env` file with `OPENAI_API_KEY` and `HF_TOKEN`.

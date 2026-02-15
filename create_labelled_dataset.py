"""
Create a labelled dataset from classified articles and upload to HuggingFace.

Filters labelled rows from ContextNews/articles, shuffles them, splits into
train/val/test (80/10/10), and pushes to ContextNews/labelled_articles.
"""

from datasets import DatasetDict, concatenate_datasets, get_dataset_split_names, load_dataset
from dotenv import load_dotenv

load_dotenv()

SOURCE_DATASET = "ContextNews/articles"
TARGET_DATASET = "ContextNews/labelled_articles"

TOPICS = [
    "politics", "geopolitics", "conflict", "crime", "law", "business",
    "economy", "markets", "technology", "science", "health", "environment",
    "society", "education", "sports", "entertainment",
]

SEED = 42


def main():
    # Load all splits
    splits = get_dataset_split_names(SOURCE_DATASET)
    print(f"Loading {SOURCE_DATASET} (splits: {splits})...")
    datasets = [load_dataset(SOURCE_DATASET, split=s) for s in splits]
    ds = concatenate_datasets(datasets)
    print(f"Total rows: {len(ds)}")

    # Filter to labelled rows
    ds = ds.filter(lambda row: any(row[t] is not None for t in TOPICS))
    print(f"Labelled rows: {len(ds)}")

    if len(ds) == 0:
        print("No labelled rows found.")
        return

    # Shuffle
    ds = ds.shuffle(seed=SEED)

    # Split 80/10/10
    split1 = ds.train_test_split(test_size=0.2, seed=SEED)
    train_ds = split1["train"]
    remaining = split1["test"]

    split2 = remaining.train_test_split(test_size=0.5, seed=SEED)
    val_ds = split2["train"]
    test_ds = split2["test"]

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    dataset_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    print(f"Pushing to {TARGET_DATASET}...")
    dataset_dict.push_to_hub(TARGET_DATASET)
    print("Done.")


if __name__ == "__main__":
    main()

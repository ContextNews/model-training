"""
Classify HuggingFace dataset articles into topics using OpenAI async API.

Usage:
    python classify.py                  # classify up to 1000 unclassified rows
    python classify.py -n 500           # classify up to 500 rows
    python classify.py --concurrency 20 # limit concurrent requests

Environment variables:
    OPENAI_API_KEY  - Your OpenAI API key
    HF_TOKEN        - HuggingFace token with write access (for pushing results)
"""

import argparse
import asyncio
import json

from dotenv import load_dotenv
from datasets import concatenate_datasets, get_dataset_split_names, load_dataset
from openai import AsyncOpenAI

load_dotenv()

DATASET_ID = "ContextNews/articles"
MODEL = "gpt-5"

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

SYSTEM_PROMPT = """You are a news article classifier. Given the title, summary, and opening text of a news article, determine which topics it belongs to.

For each topic, return 1 if the article belongs to that topic, or 0 if it does not.
An article can belong to multiple topics.

Topics:
- politics: Domestic politics, elections, government policy, legislation, political parties
- geopolitics: International relations, diplomacy, foreign policy, global governance
- conflict: Wars, armed conflicts, military operations, terrorism, insurgencies
- crime: Criminal activity, fraud, theft, organized crime, cybercrime
- law: Legal proceedings, court cases, regulations, judicial decisions, legal reform
- business: Companies, corporate news, mergers, startups, industry developments
- economy: Economic indicators, employment, trade, inflation, fiscal policy
- markets: Stock markets, commodities, currencies, cryptocurrency, investment
- technology: Tech industry, software, hardware, AI, digital innovation
- science: Scientific research, discoveries, space, physics, biology
- health: Public health, medicine, diseases, healthcare systems, pharmaceuticals
- environment: Climate change, pollution, conservation, natural disasters
- society: Social issues, demographics, culture, immigration, religion
- education: Schools, universities, academic research, education policy
- sports: Athletic competitions, leagues, athletes, sporting events
- entertainment: Film, music, TV, celebrities, arts, gaming"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "topic_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {t: {"type": "integer", "enum": [0, 1]} for t in TOPICS},
            "required": TOPICS,
            "additionalProperties": False,
        },
    },
}


def load_all_splits():
    """Load all splits from the dataset and concatenate them."""
    splits = get_dataset_split_names(DATASET_ID)
    print(f"Found splits: {splits}")
    datasets = [load_dataset(DATASET_ID, split=s) for s in splits]
    ds = concatenate_datasets(datasets)
    print(f"Loaded {len(ds)} total rows across {len(splits)} split(s).")
    return ds, splits


def truncate_text(text: str | None, max_words: int = 300) -> str:
    if not text:
        return ""
    words = text.split()
    return " ".join(words[:max_words])


def build_user_message(row: dict) -> str:
    title = row.get("title") or ""
    summary = row.get("summary") or ""
    text_excerpt = truncate_text(row.get("text"), max_words=300)

    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if text_excerpt:
        parts.append(f"Text: {text_excerpt}")

    return "\n\n".join(parts)


async def classify_row(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    row: dict,
    progress: dict,
) -> tuple[str, dict | None]:
    """Classify a single row. Returns (article_id, scores) or (article_id, None) on failure."""
    article_id = str(row["id"])
    user_msg = build_user_message(row)
    if not user_msg.strip():
        progress["skipped"] += 1
        return article_id, None

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                response_format=RESPONSE_SCHEMA,
            )
            content = response.choices[0].message.content
            scores = json.loads(content)
            progress["done"] += 1
            total = progress["total"]
            done = progress["done"]
            failed = progress["failed"]
            print(f"\r  Classified {done}/{total} ({failed} failed)", end="", flush=True)
            return article_id, scores
        except Exception as e:
            progress["failed"] += 1
            print(f"\n  Error classifying {article_id}: {e}")
            return article_id, None


async def run(args):
    client = AsyncOpenAI()

    print(f"Loading dataset {DATASET_ID}...")
    ds, splits = load_all_splits()

    # Filter to rows that haven't been classified yet
    def needs_classification(row):
        return all(row[t] is None for t in TOPICS)

    unclassified = ds.filter(needs_classification)
    total_unclassified = len(unclassified)

    if total_unclassified == 0:
        print("All rows are already classified. Nothing to do.")
        return

    limit = min(args.num_rows, total_unclassified)
    print(f"Found {total_unclassified} unclassified rows. Will classify {limit}.")

    # Select the subset to classify
    rows = [unclassified[i] for i in range(limit)]

    semaphore = asyncio.Semaphore(args.concurrency)
    progress = {"done": 0, "failed": 0, "skipped": 0, "total": limit}

    print(f"Classifying with {args.concurrency} concurrent requests...")
    tasks = [classify_row(client, semaphore, row, progress) for row in rows]
    results = await asyncio.gather(*tasks)
    print()  # newline after progress

    # Build lookup
    classifications = {aid: scores for aid, scores in results if scores is not None}
    print(
        f"Completed: {progress['done']} succeeded, "
        f"{progress['failed']} failed, {progress['skipped']} skipped."
    )

    if not classifications:
        print("No successful classifications to apply.")
        return

    # Apply classifications to the full dataset
    def apply_classification(row):
        article_id = str(row["id"])
        if article_id in classifications:
            scores = classifications[article_id]
            for topic in TOPICS:
                row[topic] = scores.get(topic)
        return row

    ds = ds.map(apply_classification)

    classified_count = sum(1 for row in ds if any(row[t] is not None for t in TOPICS))
    print(f"Dataset now has {classified_count}/{len(ds)} classified rows.")

    # Push to HuggingFace
    if len(splits) == 1:
        print(f"Pushing updated dataset to {DATASET_ID} (split={splits[0]})...")
        ds.push_to_hub(DATASET_ID, split=splits[0])
    else:
        print(f"Pushing updated dataset to {DATASET_ID} (split=train)...")
        ds.push_to_hub(DATASET_ID, split="train")
    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Classify articles into topics using OpenAI"
    )
    parser.add_argument(
        "-n", "--num-rows", type=int, default=1000,
        help="Maximum number of rows to classify (default: 1000)",
    )
    parser.add_argument(
        "-c", "--concurrency", type=int, default=10,
        help="Maximum concurrent API requests (default: 10)",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

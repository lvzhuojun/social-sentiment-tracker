"""
src/data_loader.py — Data loading, cleaning, and splitting utilities.

Supports the Sentiment140 dataset (twitter_training.csv) and falls back to
auto-generated mock data when the real dataset is unavailable.
"""

import re
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    MOCK_DATA_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    VAL_SIZE,
    get_logger,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sentiment140 loader
# ---------------------------------------------------------------------------

def load_sentiment140(filepath: Path | str) -> pd.DataFrame:
    """Load and normalise the Sentiment140 CSV dataset.

    The raw file has no header; columns are assigned and label 4 is mapped to 1
    so the result is a binary classification problem (0 = negative, 1 = positive).

    Args:
        filepath: Path to the CSV file (``twitter_training.csv``).

    Returns:
        DataFrame with columns ``['id', 'label', 'user', 'date', 'query', 'text']``
        and labels in {0, 1}.

    Raises:
        FileNotFoundError: If *filepath* does not exist.

    Example:
        >>> df = load_sentiment140("data/raw/twitter_training.csv")
        >>> df['label'].value_counts()
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    logger.info("Loading Sentiment140 from %s", filepath)
    df = pd.read_csv(
        filepath,
        encoding="latin-1",
        header=None,
        names=["label", "id", "date", "query", "user", "text"],
    )

    # Map 4 → 1 for binary classification
    df["label"] = df["label"].map({0: 0, 4: 1})
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    logger.info("Loaded %d rows, label distribution: %s", len(df), df["label"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

# Pre-compiled patterns for performance
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")          # keep the word, drop the #
_NON_ALPHA_RE = re.compile(r"[^a-z0-9\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a raw social-media string.

    Steps applied in order:
    1. Lower-case
    2. Remove URLs (http/https/www)
    3. Remove @mentions
    4. Strip ``#`` from hashtags (keep the word)
    5. Remove remaining non-alphanumeric characters
    6. Collapse multiple whitespace to a single space

    Args:
        text: Raw input string.

    Returns:
        Cleaned string (may be empty if all tokens were noise).

    Example:
        >>> clean_text("Hello @user! Check https://example.com #NLP :)")
        'hello check nlp'
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)   # #nlp → " nlp "
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# DataFrame preprocessing
# ---------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`clean_text` to the ``text`` column and remove noise rows.

    Args:
        df: DataFrame containing at least a ``text`` column.

    Returns:
        Cleaned DataFrame with:
        * ``clean_text`` column added
        * Rows with empty clean text removed
        * Duplicate rows removed
        * Index reset

    Example:
        >>> df = pd.DataFrame({'text': ['Hello world!', '', None], 'label': [1, 0, 1]})
        >>> preprocess_dataframe(df)
    """
    logger.info("Preprocessing %d rows …", len(df))
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    before = len(df)
    df = df[df["clean_text"].str.strip().astype(bool)]   # drop empty
    df = df.drop_duplicates(subset=["clean_text"])
    df = df.reset_index(drop=True)

    logger.info("Preprocessing complete: %d → %d rows", before, len(df))
    return df


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train, validation, and test sets.

    Args:
        df: Full cleaned DataFrame with a ``label`` column.
        test_size: Fraction reserved for the test set (default 0.2).
        val_size: Fraction of the *remaining* data reserved for validation
                  (default 0.1, i.e. ≈ 8 % of total).

    Returns:
        Tuple of ``(train_df, val_df, test_df)``.

    Example:
        >>> train, val, test = split_data(df)
        >>> len(train) + len(val) + len(test) == len(df)
        True
    """
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size / (1 - test_size),
        random_state=RANDOM_SEED,
        stratify=train_val_df["label"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    logger.info(
        "Split sizes — train: %d | val: %d | test: %d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Mock data generator (development fallback)
# ---------------------------------------------------------------------------

_POSITIVE_TEMPLATES = [
    "I absolutely love this product! It works perfectly.",
    "What an amazing day! Feeling grateful and happy.",
    "Great job team, the new feature is fantastic.",
    "Just had the best coffee ever. Highly recommend!",
    "So excited about the upcoming vacation. Can't wait!",
    "The customer service was outstanding. Five stars!",
    "This movie was brilliant. Loved every minute of it.",
    "Feeling motivated and ready to take on the world today.",
    "The weather is beautiful today. Perfect for a walk.",
    "Just finished a great book. Totally recommend it!",
]

_NEGATIVE_TEMPLATES = [
    "Terrible experience. The product broke after one day.",
    "I'm so frustrated with this service. Total waste of money.",
    "The worst customer support I've ever encountered.",
    "Really disappointed with the quality. Not worth it.",
    "Had a horrible commute today. Everything went wrong.",
    "The app keeps crashing. Absolutely unusable right now.",
    "So tired of all this unnecessary drama and stress.",
    "This is the third time my order was delayed. Unacceptable.",
    "The food was cold and tasteless. Never coming back.",
    "Feeling really down today. Nothing seems to go right.",
]

_NEUTRAL_TEMPLATES = [
    "Just got back from the grocery store.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "I updated the software to the latest version.",
    "The report has been submitted for review.",
    "Currently reading a book about machine learning.",
    "The bus arrives every 20 minutes on weekdays.",
    "Watched a documentary about climate change last night.",
    "The package was delivered this morning.",
    "Attended a webinar on data science trends.",
    "The library is open until 9 PM on Fridays.",
]


def generate_mock_data(n: int = 500, save_path: Path | None = None) -> pd.DataFrame:
    """Generate synthetic sentiment data for development / testing.

    Produces a balanced dataset covering positive (label=1), negative (label=0),
    and neutral (label=2) classes with realistic-looking timestamps.

    Args:
        n: Total number of rows to generate (default 500).
        save_path: If provided, save the CSV to this path.

    Returns:
        DataFrame with columns ``['id', 'label', 'date', 'text']``.

    Example:
        >>> df = generate_mock_data(300)
        >>> df['label'].value_counts()
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    today = datetime.today()
    records = []
    per_class = n // 3
    remainder = n - per_class * 3

    templates = [
        (1, _POSITIVE_TEMPLATES),
        (0, _NEGATIVE_TEMPLATES),
        (2, _NEUTRAL_TEMPLATES),
    ]

    idx = 0
    for i, (label, pool) in enumerate(templates):
        count = per_class + (1 if i < remainder else 0)
        for _ in range(count):
            days_ago = random.randint(0, 90)
            date_str = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            text = random.choice(pool)
            # Add light variation to avoid exact duplicates
            if random.random() < 0.4:
                suffix = random.choice([" really", " honestly", " definitely", ""])
                text = text.rstrip("!.") + suffix + "."
            records.append({"id": idx, "label": label, "date": date_str, "text": text})
            idx += 1

    df = pd.DataFrame(records).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info("Mock data saved to %s", save_path)

    logger.info("Generated %d mock records (label dist: %s)", len(df), df["label"].value_counts().to_dict())
    return df


# ---------------------------------------------------------------------------
# Convenience loader — auto-selects real data or mock fallback
# ---------------------------------------------------------------------------

def load_data(real_path: Path | None = None) -> pd.DataFrame:
    """Load the best available dataset.

    Tries the Sentiment140 CSV first; falls back to mock data if unavailable.

    Args:
        real_path: Optional explicit path to the Sentiment140 CSV.
                   Defaults to ``config.SENTIMENT140_PATH``.

    Returns:
        Preprocessed DataFrame ready for modelling.
    """
    from config import SENTIMENT140_PATH  # local import to avoid circular

    path = Path(real_path) if real_path else SENTIMENT140_PATH

    try:
        df = load_sentiment140(path)
        # Sentiment140 has no neutral class — treat as binary
        df = df[df["label"].isin([0, 1])].copy()
    except FileNotFoundError:
        logger.warning("Sentiment140 not found at %s — generating mock data.", path)
        df = generate_mock_data(save_path=MOCK_DATA_PATH)

    df = preprocess_dataframe(df)
    return df


# ---------------------------------------------------------------------------
# CLI smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    print(df.head())
    print("Shape:", df.shape)
    print("Labels:", df["label"].value_counts().to_dict())

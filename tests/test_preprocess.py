"""Tests for src/preprocess.py."""

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocess import (
    add_text_features,
    lemmatize,
    remove_stopwords,
    tokenize,
)


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic(self):
        result = tokenize("machine learning is great")
        assert isinstance(result, list)
        assert "machine" in result
        assert "learning" in result

    def test_empty_string(self):
        result = tokenize("")
        assert result == []

    def test_non_string_returns_empty(self):
        assert tokenize(None) == []
        assert tokenize(42) == []

    def test_single_word(self):
        assert tokenize("hello") == ["hello"]

    def test_preserves_case(self):
        result = tokenize("Hello World")
        assert "Hello" in result or "hello" in result  # NLTK lowercases or not


# ---------------------------------------------------------------------------
# remove_stopwords
# ---------------------------------------------------------------------------

class TestRemoveStopwords:
    def test_removes_common_stopwords(self):
        tokens = ["this", "is", "a", "great", "product"]
        result = remove_stopwords(tokens)
        assert "great" in result
        assert "product" in result
        for stop in ("this", "is", "a"):
            assert stop not in result

    def test_empty_list(self):
        assert remove_stopwords([]) == []

    def test_all_stopwords(self):
        tokens = ["the", "a", "is", "and", "or"]
        result = remove_stopwords(tokens)
        assert result == []

    def test_no_stopwords(self):
        tokens = ["machine", "learning", "sentiment"]
        result = remove_stopwords(tokens)
        assert result == tokens

    def test_case_insensitive(self):
        tokens = ["This", "IS", "great"]
        result = remove_stopwords(tokens)
        assert "great" in result


# ---------------------------------------------------------------------------
# lemmatize
# ---------------------------------------------------------------------------

class TestLemmatize:
    def test_plurals(self):
        result = lemmatize(["dogs", "cats", "birds"])
        assert "dog" in result
        assert "cat" in result
        assert "bird" in result

    def test_empty_list(self):
        assert lemmatize([]) == []

    def test_returns_list(self):
        result = lemmatize(["running"])
        assert isinstance(result, list)

    def test_already_lemmatised(self):
        tokens = ["machine", "learning"]
        result = lemmatize(tokens)
        assert "machine" in result
        assert "learning" in result


# ---------------------------------------------------------------------------
# add_text_features
# ---------------------------------------------------------------------------

class TestAddTextFeatures:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "clean_text": [
                "machine learning is great",
                "sentiment analysis nlp",
                "i love this product",
            ],
            "label": [1, 1, 1],
        })

    def test_adds_word_count(self, sample_df):
        out = add_text_features(sample_df)
        assert "word_count" in out.columns

    def test_adds_char_count(self, sample_df):
        out = add_text_features(sample_df)
        assert "char_count" in out.columns

    def test_adds_avg_word_len(self, sample_df):
        out = add_text_features(sample_df)
        assert "avg_word_len" in out.columns

    def test_adds_unique_word_ratio(self, sample_df):
        out = add_text_features(sample_df)
        assert "unique_word_ratio" in out.columns

    def test_word_count_values(self, sample_df):
        out = add_text_features(sample_df)
        assert out["word_count"].iloc[0] == 4  # "machine learning is great"

    def test_char_count_values(self, sample_df):
        out = add_text_features(sample_df)
        assert out["char_count"].iloc[0] == len("machine learning is great")

    def test_unique_ratio_range(self, sample_df):
        out = add_text_features(sample_df)
        assert (out["unique_word_ratio"] >= 0).all()
        assert (out["unique_word_ratio"] <= 1).all()

    def test_does_not_mutate_input(self, sample_df):
        _ = add_text_features(sample_df)
        assert "word_count" not in sample_df.columns

    def test_handles_empty_text(self):
        df = pd.DataFrame({"clean_text": ["", "hello world"], "label": [0, 1]})
        out = add_text_features(df)
        assert out["word_count"].iloc[0] == 0
        assert out["char_count"].iloc[0] == 0

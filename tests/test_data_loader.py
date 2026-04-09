"""Tests for src/data_loader.py."""

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import (
    clean_text,
    generate_mock_data,
    preprocess_dataframe,
    split_data,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_url_https(self):
        result = clean_text("visit https://example.com today")
        assert "https" not in result
        assert "example" not in result

    def test_removes_url_www(self):
        result = clean_text("go to www.example.com")
        assert "www" not in result

    def test_removes_mention(self):
        result = clean_text("hello @user how are you")
        assert "@user" not in result
        assert "user" not in result

    def test_strips_hashtag_symbol_keeps_word(self):
        result = clean_text("#NLP is great")
        assert "nlp" in result
        assert "#" not in result

    def test_removes_punctuation(self):
        result = clean_text("great! really? yes.")
        assert "!" not in result
        assert "?" not in result

    def test_collapses_whitespace(self):
        result = clean_text("too   many    spaces")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_non_string_returns_empty(self):
        assert clean_text(None) == ""
        assert clean_text(123) == ""

    def test_all_noise(self):
        result = clean_text("@user https://t.co/abc #!")
        assert result == ""

    def test_mixed(self):
        result = clean_text("Hello @user! Check https://example.com #NLP :)")
        assert "hello" in result
        assert "nlp" in result
        assert "@" not in result
        assert "http" not in result


# ---------------------------------------------------------------------------
# generate_mock_data
# ---------------------------------------------------------------------------

class TestGenerateMockData:
    def test_returns_dataframe(self):
        df = generate_mock_data(n=30)
        assert isinstance(df, pd.DataFrame)

    def test_correct_columns(self):
        df = generate_mock_data(n=30)
        for col in ("id", "label", "date", "text"):
            assert col in df.columns

    def test_correct_row_count(self):
        df = generate_mock_data(n=30)
        assert len(df) == 30

    def test_three_classes(self):
        df = generate_mock_data(n=300)
        assert set(df["label"].unique()) == {0, 1, 2}

    def test_roughly_balanced(self):
        df = generate_mock_data(n=300)
        counts = df["label"].value_counts()
        for c in counts:
            assert 80 <= c <= 120, f"Class count {c} is too unbalanced"

    def test_reproducible_with_same_seed(self):
        df1 = generate_mock_data(n=50)
        df2 = generate_mock_data(n=50)
        assert df1["text"].tolist() == df2["text"].tolist()

    def test_save_path(self, tmp_path):
        path = tmp_path / "mock.csv"
        df = generate_mock_data(n=30, save_path=path)
        assert path.exists()
        df_loaded = pd.read_csv(path)
        assert len(df_loaded) == 30


# ---------------------------------------------------------------------------
# preprocess_dataframe
# ---------------------------------------------------------------------------

class TestPreprocessDataframe:
    def _make_df(self, texts, labels=None):
        if labels is None:
            labels = [1] * len(texts)
        return pd.DataFrame({"text": texts, "label": labels})

    def test_adds_clean_text_column(self):
        df = self._make_df(["Hello world", "Good morning"])
        out = preprocess_dataframe(df)
        assert "clean_text" in out.columns

    def test_drops_empty_after_cleaning(self):
        df = self._make_df(["@user https://t.co", "valid text here"])
        out = preprocess_dataframe(df)
        assert len(out) == 1
        assert "valid" in out["clean_text"].iloc[0]

    def test_drops_none_text(self):
        df = self._make_df([None, "good text"])
        out = preprocess_dataframe(df)
        assert len(out) == 1

    def test_drops_duplicate_clean_text(self):
        df = self._make_df(["hello world", "HELLO WORLD", "different text"])
        out = preprocess_dataframe(df)
        assert len(out) == 2

    def test_resets_index(self):
        df = self._make_df(["text one", "text two", "text three"])
        out = preprocess_dataframe(df)
        assert list(out.index) == list(range(len(out)))

    def test_does_not_mutate_input(self):
        df = self._make_df(["hello"])
        _ = preprocess_dataframe(df)
        assert "clean_text" not in df.columns


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    @pytest.fixture
    def mock_df(self):
        df = generate_mock_data(n=300)
        return preprocess_dataframe(df)

    def test_returns_three_dataframes(self, mock_df, tmp_path):
        result = split_data(mock_df, save_dir=tmp_path)
        assert len(result) == 3
        for part in result:
            assert isinstance(part, pd.DataFrame)

    def test_total_size_preserved(self, mock_df, tmp_path):
        train, val, test = split_data(mock_df, save_dir=tmp_path)
        assert len(train) + len(val) + len(test) == len(mock_df)

    def test_test_fraction(self, mock_df, tmp_path):
        train, val, test = split_data(mock_df, test_size=0.2, save_dir=tmp_path)
        assert abs(len(test) / len(mock_df) - 0.2) < 0.05

    def test_no_overlap(self, mock_df, tmp_path):
        # Use the 'id' column (not .index) because split_data calls reset_index
        train, val, test = split_data(mock_df, save_dir=tmp_path)
        train_ids = set(train["id"])
        val_ids = set(val["id"])
        test_ids = set(test["id"])
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)

    def test_saves_csv_files(self, mock_df, tmp_path):
        split_data(mock_df, save_dir=tmp_path)
        for name in ("train.csv", "val.csv", "test.csv"):
            assert (tmp_path / name).exists()

    def test_stratified_labels(self, mock_df, tmp_path):
        train, val, test = split_data(mock_df, save_dir=tmp_path)
        for part in (train, val, test):
            assert set(part["label"].unique()) == set(mock_df["label"].unique())

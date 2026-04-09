"""Tests for src/baseline_model.py."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.baseline_model import build_pipeline, load_baseline_model, predict, train_baseline
from src.data_loader import generate_mock_data, preprocess_dataframe, split_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_split(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    df = preprocess_dataframe(generate_mock_data(n=150))
    train, val, test = split_data(df, save_dir=tmp)
    return train, val, test


# ---------------------------------------------------------------------------
# build_pipeline
# ---------------------------------------------------------------------------

class TestBuildPipeline:
    def test_returns_pipeline(self):
        pipe = build_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_has_tfidf_step(self):
        pipe = build_pipeline()
        assert "tfidf" in pipe.named_steps

    def test_has_clf_step(self):
        pipe = build_pipeline()
        assert "clf" in pipe.named_steps

    def test_pipeline_is_unfitted(self):
        from sklearn.exceptions import NotFittedError
        pipe = build_pipeline()
        with pytest.raises(NotFittedError):
            pipe.predict(["test"])


# ---------------------------------------------------------------------------
# train_baseline + predict
# ---------------------------------------------------------------------------

class TestTrainBaseline:
    def test_returns_pipeline(self, small_split, tmp_path):
        train, val, _ = small_split
        pipe = train_baseline(train, val)
        assert isinstance(pipe, Pipeline)

    def test_saves_model_file(self, small_split, tmp_path):
        from config import BASELINE_MODEL_PATH
        train, val, _ = small_split
        train_baseline(train, val)
        assert BASELINE_MODEL_PATH.exists()


class TestPredict:
    @pytest.fixture
    def trained_pipeline(self, small_split):
        train, val, _ = small_split
        return train_baseline(train, val)

    def test_returns_two_arrays(self, trained_pipeline):
        labels, probs = predict(trained_pipeline, ["great product", "terrible service"])
        assert isinstance(labels, np.ndarray)
        assert isinstance(probs, np.ndarray)

    def test_label_shape(self, trained_pipeline):
        texts = ["good", "bad", "okay"]
        labels, _ = predict(trained_pipeline, texts)
        assert labels.shape == (3,)

    def test_probs_shape(self, trained_pipeline):
        texts = ["good", "bad", "okay"]
        _, probs = predict(trained_pipeline, texts)
        assert probs.ndim == 2
        assert probs.shape[0] == 3

    def test_probs_sum_to_one(self, trained_pipeline):
        _, probs = predict(trained_pipeline, ["any text here"])
        assert abs(probs[0].sum() - 1.0) < 1e-6

    def test_labels_are_valid_classes(self, trained_pipeline, small_split):
        train, _, _ = small_split
        valid_labels = set(train["label"].unique())
        labels, _ = predict(trained_pipeline, ["some text"])
        assert int(labels[0]) in valid_labels


# ---------------------------------------------------------------------------
# load_baseline_model
# ---------------------------------------------------------------------------

class TestLoadBaselineModel:
    def test_raises_if_not_found(self, tmp_path):
        fake_path = tmp_path / "nonexistent.pkl"
        with pytest.raises(FileNotFoundError):
            load_baseline_model(path=fake_path)

    def test_loads_saved_model(self, small_split, tmp_path):
        train, val, _ = small_split
        pipe = train_baseline(train, val)
        from config import BASELINE_MODEL_PATH
        loaded = load_baseline_model(path=BASELINE_MODEL_PATH)
        assert isinstance(loaded, Pipeline)

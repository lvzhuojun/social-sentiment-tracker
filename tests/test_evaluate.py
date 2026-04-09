"""Tests for src/evaluate.py."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import compare_models, evaluate_model, plot_confusion_matrix


# ---------------------------------------------------------------------------
# evaluate_model
# ---------------------------------------------------------------------------

class TestEvaluateModel:
    @pytest.fixture
    def binary_data(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 1])
        return y_true, y_pred

    def test_returns_dict(self, binary_data):
        y_true, y_pred = binary_data
        result = evaluate_model(y_true, y_pred)
        assert isinstance(result, dict)

    def test_has_required_keys(self, binary_data):
        y_true, y_pred = binary_data
        result = evaluate_model(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            assert key in result

    def test_perfect_prediction(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_model(y_true, y_pred)
        assert result["accuracy"] == 1.0
        assert result["f1"] == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        result = evaluate_model(y_true, y_pred)
        assert result["accuracy"] == 0.0

    def test_roc_auc_with_scores(self, binary_data):
        y_true, y_pred = binary_data
        y_scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4],
                              [0.9, 0.1], [0.2, 0.8], [0.4, 0.6],
                              [0.1, 0.9], [0.3, 0.7]])
        result = evaluate_model(y_true, y_pred, y_scores=y_scores)
        assert not math.isnan(result["roc_auc"])
        assert 0.0 <= result["roc_auc"] <= 1.0

    def test_roc_auc_nan_without_scores(self, binary_data):
        y_true, y_pred = binary_data
        result = evaluate_model(y_true, y_pred)
        assert math.isnan(result["roc_auc"])

    def test_values_in_range(self, binary_data):
        y_true, y_pred = binary_data
        result = evaluate_model(y_true, y_pred)
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0

    def test_three_class(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 0, 2])
        result = evaluate_model(y_true, y_pred, model_name="3-class")
        assert isinstance(result, dict)
        assert 0.0 <= result["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# plot_confusion_matrix
# ---------------------------------------------------------------------------

class TestPlotConfusionMatrix:
    def test_returns_path(self, tmp_path, monkeypatch):
        import src.evaluate as ev
        monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = plot_confusion_matrix(y_true, y_pred, model_name="Test")
        assert isinstance(result, Path)

    def test_saves_png(self, tmp_path, monkeypatch):
        import src.evaluate as ev
        monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0])
        result = plot_confusion_matrix(y_true, y_pred, model_name="SaveTest")
        assert result.exists()
        assert result.suffix == ".png"


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    @pytest.fixture
    def sample_results(self):
        baseline = {"accuracy": 0.85, "precision": 0.86, "recall": 0.85, "f1": 0.85, "roc_auc": 0.92}
        bert = {"accuracy": 0.90, "precision": 0.91, "recall": 0.90, "f1": 0.90, "roc_auc": 0.96}
        return baseline, bert

    def test_returns_dataframe(self, sample_results, tmp_path, monkeypatch):
        import src.evaluate as ev
        monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
        baseline, bert = sample_results
        result = compare_models(baseline, bert)
        assert isinstance(result, pd.DataFrame)

    def test_dataframe_has_two_rows(self, sample_results, tmp_path, monkeypatch):
        import src.evaluate as ev
        monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
        baseline, bert = sample_results
        result = compare_models(baseline, bert)
        assert len(result) == 2

    def test_dataframe_columns(self, sample_results, tmp_path, monkeypatch):
        import src.evaluate as ev
        monkeypatch.setattr(ev, "FIGURES_DIR", tmp_path)
        baseline, bert = sample_results
        result = compare_models(baseline, bert)
        for col in ("Accuracy", "Precision", "Recall", "F1"):
            assert col in result.columns

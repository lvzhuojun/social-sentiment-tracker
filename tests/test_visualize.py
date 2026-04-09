"""Tests for src/visualize.py."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.visualize import (
    plot_confidence_gauge,
    plot_sentiment_distribution,
    plot_sentiment_over_time,
    plot_text_length_distribution,
    plot_top_keywords,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    from src.data_loader import generate_mock_data, preprocess_dataframe
    return preprocess_dataframe(generate_mock_data(n=120))


# ---------------------------------------------------------------------------
# plot_sentiment_distribution
# ---------------------------------------------------------------------------

class TestPlotSentimentDistribution:
    def test_returns_figure(self, sample_df):
        fig = plot_sentiment_distribution(sample_df)
        assert isinstance(fig, go.Figure)

    def test_figure_has_data(self, sample_df):
        fig = plot_sentiment_distribution(sample_df)
        assert len(fig.data) > 0

    def test_empty_df_does_not_crash(self):
        df = pd.DataFrame({"label": []})
        fig = plot_sentiment_distribution(df)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_text_length_distribution
# ---------------------------------------------------------------------------

class TestPlotTextLengthDistribution:
    def test_returns_figure(self, sample_df):
        fig = plot_text_length_distribution(sample_df)
        assert isinstance(fig, go.Figure)

    def test_figure_has_data(self, sample_df):
        fig = plot_text_length_distribution(sample_df)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# plot_top_keywords
# ---------------------------------------------------------------------------

class TestPlotTopKeywords:
    def test_returns_figure(self, sample_df):
        fig = plot_top_keywords(sample_df, n=10)
        assert isinstance(fig, go.Figure)

    def test_returns_figure_with_sentiment_filter(self, sample_df):
        fig = plot_top_keywords(sample_df, n=10, sentiment=1)
        assert isinstance(fig, go.Figure)

    def test_empty_subset_returns_empty_figure(self, sample_df):
        fig = plot_top_keywords(sample_df, n=10, sentiment=99)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_sentiment_over_time
# ---------------------------------------------------------------------------

class TestPlotSentimentOverTime:
    def test_returns_figure(self, sample_df):
        fig = plot_sentiment_over_time(sample_df)
        assert isinstance(fig, go.Figure)

    def test_missing_date_column_returns_empty(self, sample_df):
        df = sample_df.drop(columns=["date"], errors="ignore")
        # should return a Figure even if date col is absent
        fig = plot_sentiment_over_time(df, date_col="nonexistent_col")
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_confidence_gauge
# ---------------------------------------------------------------------------

class TestPlotConfidenceGauge:
    def test_returns_figure(self):
        fig = plot_confidence_gauge(0.87, "Positive")
        assert isinstance(fig, go.Figure)

    def test_positive_color(self):
        fig = plot_confidence_gauge(0.9, "Positive")
        bar_color = fig.data[0].gauge.bar.color
        assert bar_color == "#00CC96"

    def test_negative_color(self):
        fig = plot_confidence_gauge(0.7, "Negative")
        bar_color = fig.data[0].gauge.bar.color
        assert bar_color == "#EF553B"

    def test_neutral_color(self):
        fig = plot_confidence_gauge(0.6, "Neutral")
        bar_color = fig.data[0].gauge.bar.color
        assert bar_color == "#636EFA"

    def test_value_displayed(self):
        fig = plot_confidence_gauge(0.75, "Positive")
        assert fig.data[0].value == 75.0

    @pytest.mark.parametrize("conf", [0.0, 0.5, 1.0])
    def test_boundary_confidences(self, conf):
        fig = plot_confidence_gauge(conf, "Positive")
        assert isinstance(fig, go.Figure)

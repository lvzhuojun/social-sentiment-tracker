"""Tests for config.py — paths, seed, and logger."""

import logging
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

def test_root_dir_is_path():
    assert isinstance(config.ROOT_DIR, Path)


def test_data_dirs_are_paths():
    assert isinstance(config.RAW_DATA_DIR, Path)
    assert isinstance(config.PROCESSED_DATA_DIR, Path)
    assert isinstance(config.MODELS_DIR, Path)
    assert isinstance(config.FIGURES_DIR, Path)


def test_paths_are_under_root():
    assert config.RAW_DATA_DIR.is_relative_to(config.ROOT_DIR)
    assert config.MODELS_DIR.is_relative_to(config.ROOT_DIR)
    assert config.FIGURES_DIR.is_relative_to(config.ROOT_DIR)


# ---------------------------------------------------------------------------
# Hyperparameter types
# ---------------------------------------------------------------------------

def test_hyperparameter_types():
    assert isinstance(config.RANDOM_SEED, int)
    assert isinstance(config.BATCH_SIZE, int)
    assert isinstance(config.EPOCHS, int)
    assert isinstance(config.LEARNING_RATE, float)
    assert isinstance(config.MAX_LENGTH, int)
    assert isinstance(config.TFIDF_MAX_FEATURES, int)


def test_hyperparameter_values_sensible():
    assert 0 < config.LEARNING_RATE < 1
    assert config.BATCH_SIZE > 0
    assert config.EPOCHS > 0
    assert config.MAX_LENGTH > 0
    assert config.TFIDF_MAX_FEATURES > 0
    assert 0.0 < config.TEST_SIZE < 1.0
    assert 0.0 < config.VAL_SIZE < 1.0


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------

def test_set_seed_runs_without_error():
    config.set_seed(0)
    config.set_seed(42)
    config.set_seed(99)


def test_set_seed_fixes_random():
    import random
    config.set_seed(42)
    val1 = random.random()
    config.set_seed(42)
    val2 = random.random()
    assert val1 == val2


def test_set_seed_fixes_numpy():
    import numpy as np
    config.set_seed(42)
    arr1 = np.random.rand(5)
    config.set_seed(42)
    arr2 = np.random.rand(5)
    assert (arr1 == arr2).all()


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------

def test_get_logger_returns_logger():
    logger = config.get_logger("test_module")
    assert isinstance(logger, logging.Logger)


def test_get_logger_name():
    logger = config.get_logger("my.module")
    assert logger.name == "my.module"


def test_get_logger_idempotent():
    l1 = config.get_logger("same")
    l2 = config.get_logger("same")
    assert l1 is l2

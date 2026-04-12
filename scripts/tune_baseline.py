"""
scripts/tune_baseline.py — Hyperparameter grid search for the TF-IDF + LR baseline.

Runs a 3-fold stratified cross-validation grid search over the most impactful
TF-IDF and LogisticRegression parameters, then saves a heatmap and prints
the best configuration found.

Usage:
    python scripts/tune_baseline.py
    python scripts/tune_baseline.py --output reports/tuning_results.json

Requirements:
    - data/raw/tweet_eval_sentiment.csv  (run scripts/download_data.py first)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import FIGURES_DIR, TWEET_EVAL_PATH, get_logger, set_seed

logger = get_logger(__name__)
set_seed()

# ---------------------------------------------------------------------------
# Parameter grid
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "tfidf__max_features": [10_000, 30_000, 50_000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "clf__C": [0.1, 1.0, 10.0],
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(output_path: Path) -> None:
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from src.baseline_model import build_pipeline
    from src.data_loader import preprocess_dataframe

    logger.info("Loading TweetEval data from %s", TWEET_EVAL_PATH)
    df = pd.read_csv(TWEET_EVAL_PATH)
    df = df.dropna(subset=["label", "text"])
    df["label"] = df["label"].astype(int)

    # Use only train + validation splits for tuning (keep test blind)
    tune_df = df[df["split"].isin(["train", "validation"])].copy().reset_index(drop=True)
    tune_df = preprocess_dataframe(tune_df)

    X = tune_df["clean_text"].tolist()
    y = tune_df["label"].values
    logger.info("Tuning on %d samples, %d classes", len(X), len(np.unique(y)))

    pipeline = build_pipeline()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    logger.info("Starting GridSearchCV — %d parameter combinations × 3 folds",
                3 * 3 * 3)  # len of each list in PARAM_GRID
    t0 = time.time()

    grid_search = GridSearchCV(
        pipeline,
        PARAM_GRID,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    grid_search.fit(X, y)

    elapsed = time.time() - t0
    logger.info("Grid search completed in %.1f min", elapsed / 60)

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    logger.info("Best weighted F1: %.4f", best_score)
    logger.info("Best params: %s", best_params)

    print("\n" + "=" * 60)
    print("GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"Best weighted F1 (3-fold CV): {best_score:.4f}")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=" * 60 + "\n")

    # Compare to config defaults
    from config import LR_C, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE
    print("Config defaults for comparison:")
    print(f"  tfidf__max_features : {TFIDF_MAX_FEATURES}")
    print(f"  tfidf__ngram_range  : {TFIDF_NGRAM_RANGE}")
    print(f"  clf__C              : {LR_C}")
    print()

    # ---------------------------------------------------------------------------
    # Save results to JSON
    # ---------------------------------------------------------------------------
    cv_results = grid_search.cv_results_
    records = []
    for i in range(len(cv_results["params"])):
        records.append({
            "params": {k: str(v) for k, v in cv_results["params"][i].items()},
            "mean_f1": round(float(cv_results["mean_test_score"][i]), 4),
            "std_f1": round(float(cv_results["std_test_score"][i]), 4),
            "rank": int(cv_results["rank_test_score"][i]),
        })
    records.sort(key=lambda r: r["rank"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": {k: str(v) for k, v in best_params.items()},
        "best_f1_cv": round(best_score, 4),
        "all_results": records,
    }
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Results saved → %s", output_path)

    # ---------------------------------------------------------------------------
    # Heatmap: ngram_range vs max_features (at best C)
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        best_C = best_params["clf__C"]
        mask = np.array([
            r["params"]["clf__C"] == str(best_C) for r in records
        ])
        sub = [r for r, m in zip(records, mask) if m]

        ngrams = sorted({r["params"]["tfidf__ngram_range"] for r in sub})
        feats = sorted({r["params"]["tfidf__max_features"] for r in sub})
        matrix = np.zeros((len(ngrams), len(feats)))
        for r in sub:
            i = ngrams.index(r["params"]["tfidf__ngram_range"])
            j = feats.index(r["params"]["tfidf__max_features"])
            matrix[i, j] = r["mean_f1"]

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.heatmap(
            matrix,
            annot=True, fmt=".4f",
            xticklabels=[f"{f}" for f in feats],
            yticklabels=ngrams,
            cmap="YlGnBu",
            ax=ax,
            vmin=matrix.min() - 0.01,
            vmax=matrix.max() + 0.005,
        )
        ax.set_title(f"Weighted F1 by TF-IDF params  (C={best_C})", fontsize=12)
        ax.set_xlabel("max_features")
        ax.set_ylabel("ngram_range")
        fig.tight_layout()

        heatmap_path = FIGURES_DIR / "tuning_heatmap.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(heatmap_path, dpi=150)
        plt.close(fig)
        logger.info("Heatmap saved → %s", heatmap_path)
        print(f"Heatmap → {heatmap_path}")

    except Exception as exc:
        logger.warning("Could not generate heatmap: %s", exc)

    print(f"Full results → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF + LR hyperparameter grid search")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tuning_results.json"),
        help="Path to save JSON results (default: reports/tuning_results.json)",
    )
    args = parser.parse_args()
    main(args.output)

"""run_tuning.py — Bayesian hyperparameter search for all classifier models.

Workflow
--------
1. Load dataset and split into train / validation / test.
2. For each model run ``model.tune(X_train, y_train, X_val, y_val)``
   using Optuna (TPE sampler).  Only the validation set is used to
   score trials — the test set is never touched during search.
3. After tuning, retrain with best params on train + validation combined.
4. Evaluate on the held-out test set and report.
5. Save the tuned model weights alongside a tuning summary CSV.

Notebook usage
--------------
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
    from ml.run_tuning import run_tuning
    results = run_tuning('Combined_Dataset.xlsx', n_trials=50)
    display(results)

CLI usage
---------
    cd prashant
    python -m ml.run_tuning --data Combined_Dataset.xlsx --n-trials 50
"""
from __future__ import annotations

import argparse
import time
import traceback
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ml.data_pipeline import DataPipeline


# ── Model registry ────────────────────────────────────────────────────────── #

def _build_registry() -> dict:
    from ml.AdaBoost import AdaBoostModel
    from ml.DecisionTree import DecisionTreeModel
    from ml.ExtraTrees import ExtraTreesModel
    from ml.GradientBoosting import GradientBoostingModel
    from ml.KNN import KNNModel
    from ml.LogisticRegression import LogisticRegressionModel
    from ml.MLP import MLPModel
    from ml.NaiveBayes import NaiveBayesModel
    from ml.RandomForest import RandomForestModel
    from ml.SVM import SVMModel

    registry = {
        "DecisionTree": DecisionTreeModel,
        "RandomForest": RandomForestModel,
        "GradientBoosting": GradientBoostingModel,
        "ExtraTrees": ExtraTreesModel,
        "SVM": SVMModel,
        "KNN": KNNModel,
        "LogisticRegression": LogisticRegressionModel,
        "NaiveBayes": NaiveBayesModel,
        "AdaBoost": AdaBoostModel,
        "MLP": MLPModel,
    }

    try:
        from ml.XGBoost import XGBoostModel
        registry["XGBoost"] = XGBoostModel
    except ImportError:
        warnings.warn("xgboost not installed — XGBoost skipped. pip install xgboost")

    return registry


# ── Main orchestrator ─────────────────────────────────────────────────────── #

def run_tuning(
    data_path: str | Path,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    save_dir: Optional[str | Path] = None,
    columns_to_drop: Optional[list[str]] = None,
    log_transform: bool = True,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Tune every registered model and return a comparison DataFrame.

    Parameters
    ----------
    data_path   : Path to the Excel dataset.
    n_trials    : Optuna trials per model (more = better search, slower).
    timeout     : Optional per-model wall-clock limit in seconds.
    save_dir    : Where to write ``{ModelName}_tuned.joblib`` files.
                  Defaults to ``<ml_package>/saved_models/``.
    columns_to_drop : Feature columns to exclude (missing ones ignored).
    log_transform   : Apply log1p to energy features.
    val_size / test_size / random_state : Passed to DataPipeline.
    verbose     : Print per-model progress to stdout.

    Returns
    -------
    pd.DataFrame with one row per model and columns:
        Best-Val-F1, Test-Accuracy, Test-Macro-F1, Test-Weighted-F1,
        Tune-Time(s), Train-Time(s), Best-Params, Status
    """
    if save_dir is None:
        save_dir = Path(__file__).parent / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────── #
    pipeline = DataPipeline(
        log_transform=log_transform,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        columns_to_drop=columns_to_drop,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.load_and_prepare(data_path)
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    label_names = pipeline.get_label_names()

    pipeline.save(save_dir / "pipeline_tuned.joblib")

    if verbose:
        print(f"Dataset   : {Path(data_path).name}")
        print(f"Features  : {len(pipeline.get_feature_names())}")
        print(f"Classes   : {label_names}")
        print(f"Train / Val / Test : {len(y_train)} / {len(y_val)} / {len(y_test)}")
        print(f"Optuna trials/model: {n_trials}")
        print(f"Save dir  : {save_dir}\n")
        print(
            f"{'Model':<22} {'ValF1':>7} {'TestAcc':>8} {'TestF1':>8}"
            f" {'Tune(s)':>8} {'Train(s)':>9} {'Status'}"
        )
        print("-" * 80)

    registry = _build_registry()
    rows = []

    for name, ModelCls in registry.items():
        row: dict = {"Model": name}
        try:
            model = ModelCls()

            # ── 1. Bayesian search on train → scored on val ────────────── #
            t_tune = time.perf_counter()
            tune_result = model.tune(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials,
                timeout=timeout,
            )
            tune_time = time.perf_counter() - t_tune

            # ── 2. Retrain with best params on train + val combined ─────── #
            t_train = time.perf_counter()
            model.train(X_trainval, y_trainval)
            train_time = time.perf_counter() - t_train

            # ── 3. Evaluate on held-out test set ────────────────────────── #
            metrics = model.evaluate(X_test, y_test)

            model.save(save_dir / f"{name}_tuned.joblib")

            row.update({
                "Best-Val-F1": round(tune_result["best_val_f1"], 4),
                "Test-Accuracy": round(metrics["accuracy"], 4),
                "Test-Macro-F1": round(metrics["macro_f1"], 4),
                "Test-Weighted-F1": round(metrics["weighted_f1"], 4),
                "Test-Macro-Precision": round(metrics["macro_precision"], 4),
                "Test-Macro-Recall": round(metrics["macro_recall"], 4),
                "Tune-Time(s)": round(tune_time, 2),
                "Train-Time(s)": round(train_time, 3),
                "Best-Params": tune_result["best_params"],
                "Status": "OK",
            })

            if verbose:
                print(
                    f"{name:<22} {row['Best-Val-F1']:>7.4f}"
                    f" {row['Test-Accuracy']:>8.4f} {row['Test-Macro-F1']:>8.4f}"
                    f" {row['Tune-Time(s)']:>8.1f} {row['Train-Time(s)']:>9.3f}   OK"
                )

        except Exception as exc:
            row.update({
                "Best-Val-F1": None, "Test-Accuracy": None,
                "Test-Macro-F1": None, "Test-Weighted-F1": None,
                "Test-Macro-Precision": None, "Test-Macro-Recall": None,
                "Tune-Time(s)": None, "Train-Time(s)": None,
                "Best-Params": None,
                "Status": f"ERROR: {exc}",
            })
            if verbose:
                print(f"{name:<22} {'—':>7} {'—':>8} {'—':>8} {'—':>8} {'—':>9}   ERROR: {exc}")
                traceback.print_exc()

        rows.append(row)

    results = pd.DataFrame(rows).set_index("Model")

    # Save summary CSV (without the study objects)
    csv_cols = [c for c in results.columns if c != "Best-Params"]
    results[csv_cols].to_csv(save_dir / "tuning_summary.csv")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian hyperparameter tuning for all models.")
    parser.add_argument("--data", required=True, help="Path to Combined_Dataset.xlsx")
    parser.add_argument("--n-trials", type=int, default=50, help="Optuna trials per model")
    parser.add_argument("--timeout", type=int, default=None, help="Per-model timeout in seconds")
    parser.add_argument("--save-dir", default=None, help="Directory for tuned .joblib files")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--no-log", action="store_true", help="Disable log1p transform")
    args = parser.parse_args()

    df = run_tuning(
        data_path=args.data,
        n_trials=args.n_trials,
        timeout=args.timeout,
        save_dir=args.save_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        log_transform=not args.no_log,
    )
    print("\n── Tuning Summary ──")
    display_cols = ["Best-Val-F1", "Test-Accuracy", "Test-Macro-F1", "Tune-Time(s)", "Status"]
    print(df[display_cols].to_string())

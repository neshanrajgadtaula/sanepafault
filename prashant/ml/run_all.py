"""run_all.py — Train and evaluate all classifier models on the fault dataset.

Notebook usage
--------------
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))   # add prashant/ to path
    from ml.run_all import run_all
    results = run_all('Combined_Dataset.xlsx')
    display(results)

CLI usage
---------
    cd prashant
    python -m ml.run_all --data Combined_Dataset.xlsx
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import traceback
import warnings
from pathlib import Path
from typing import Optional

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

    # XGBoost is optional — skip gracefully if not installed
    try:
        from ml.XGBoost import XGBoostModel
        registry["XGBoost"] = XGBoostModel
    except ImportError:
        warnings.warn("xgboost not installed — XGBoost model skipped. pip install xgboost")

    return registry


# ── Main orchestrator ─────────────────────────────────────────────────────── #

def run_all(
    data_path: str | Path,
    save_dir: Optional[str | Path] = None,
    cv: int = 5,
    columns_to_drop: Optional[list[str]] = None,
    log_transform: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every registered model on ``data_path`` and return a metrics table.

    Parameters
    ----------
    data_path:
        Path to the Excel dataset (e.g. ``'Combined_Dataset.xlsx'``).
    save_dir:
        Directory to write ``.joblib`` files.  Defaults to
        ``<ml_package_dir>/saved_models/``.
    cv:
        Number of stratified k-fold cross-validation folds.
    columns_to_drop:
        Optional list of feature columns to exclude before training.
        Missing columns are silently ignored.
    log_transform:
        Apply log1p to energy features (default True).
    test_size / random_state:
        Passed directly to :class:`~ml.data_pipeline.DataPipeline`.
    verbose:
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame
        One row per model with columns:
        Accuracy, Macro-F1, Weighted-F1, Macro-Precision, Macro-Recall,
        CV-Mean-Accuracy, CV-Std-Accuracy, CV-Mean-F1, CV-Std-F1,
        Train-Time(s), Status
    """
    # ── resolve save directory ──────────────────────────────────────────── #
    if save_dir is None:
        save_dir = Path(__file__).parent / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── data ────────────────────────────────────────────────────────────── #
    pipeline = DataPipeline(
        log_transform=log_transform,
        test_size=test_size,
        random_state=random_state,
        columns_to_drop=columns_to_drop,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.load_and_prepare(data_path)
    # For baseline (no tuning), train on train+val combined
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    label_names = pipeline.get_label_names()

    pipeline.save(save_dir / "pipeline.joblib")
    if verbose:
        print(f"Dataset      : {Path(data_path).name}")
        print(f"Features     : {len(pipeline.get_feature_names())}")
        print(f"Classes      : {label_names}")
        print(f"Train+Val/Test: {len(y_trainval)} / {len(y_test)}")
        print(f"Save dir     : {save_dir}\n")
        print(f"{'Model':<22} {'Acc':>6} {'MacF1':>7} {'WF1':>7} {'CVAcc':>7} {'Train(s)':>9} {'Status'}")
        print("-" * 72)

    registry = _build_registry()
    rows = []

    for name, ModelCls in registry.items():
        row: dict = {"Model": name}
        try:
            model = ModelCls()

            t0 = time.perf_counter()
            model.train(X_trainval, y_trainval)
            train_time = time.perf_counter() - t0

            metrics = model.evaluate(X_test, y_test)
            cv_result = model.cross_validate(X_trainval, y_trainval, cv=cv)

            model.save(save_dir / f"{name}.joblib")

            row.update({
                "Accuracy": round(metrics["accuracy"], 4),
                "Macro-F1": round(metrics["macro_f1"], 4),
                "Weighted-F1": round(metrics["weighted_f1"], 4),
                "Macro-Precision": round(metrics["macro_precision"], 4),
                "Macro-Recall": round(metrics["macro_recall"], 4),
                "CV-Mean-Accuracy": round(cv_result["mean_accuracy"], 4),
                "CV-Std-Accuracy": round(cv_result["std_accuracy"], 4),
                "CV-Mean-F1": round(cv_result["mean_f1"], 4),
                "CV-Std-F1": round(cv_result["std_f1"], 4),
                "Train-Time(s)": round(train_time, 3),
                "Status": "OK",
            })

            if verbose:
                print(
                    f"{name:<22} {row['Accuracy']:>6.4f} {row['Macro-F1']:>7.4f}"
                    f" {row['Weighted-F1']:>7.4f} {row['CV-Mean-Accuracy']:>7.4f}"
                    f" {row['Train-Time(s)']:>9.3f}   OK"
                )

        except Exception as exc:
            row.update({
                "Accuracy": None, "Macro-F1": None, "Weighted-F1": None,
                "Macro-Precision": None, "Macro-Recall": None,
                "CV-Mean-Accuracy": None, "CV-Std-Accuracy": None,
                "CV-Mean-F1": None, "CV-Std-F1": None,
                "Train-Time(s)": None,
                "Status": f"ERROR: {exc}",
            })
            if verbose:
                print(f"{name:<22} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {'—':>9}   ERROR: {exc}")
                traceback.print_exc()

        rows.append(row)

    results = pd.DataFrame(rows).set_index("Model")
    return results


# ── CLI entry-point ───────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all fault-classification models.")
    parser.add_argument("--data", required=True, help="Path to Combined_Dataset.xlsx")
    parser.add_argument("--save-dir", default=None, help="Directory for .joblib files")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--no-log", action="store_true", help="Disable log1p transform")
    args = parser.parse_args()

    df = run_all(
        data_path=args.data,
        save_dir=args.save_dir,
        cv=args.cv,
        log_transform=not args.no_log,
    )
    print("\n── Summary ──")
    print(df.to_string())

"""run_all.py — Train and evaluate all classifier models on the fault dataset.

Notebook usage
--------------
    import sys, os
    sys.path.insert(0, os.path.abspath('..'))
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


# ── Plotting helpers (called only when save_graphs=True) ──────────────────── #

def _eda_graphs(data_path, pipeline, graph_dirs: dict) -> None:
    from ml.plots import EDAPlots
    feat = pipeline.get_feature_names()
    print("\n[graphs] Generating EDA plots …")
    EDAPlots.class_distribution(data_path, feat, graph_dirs["eda"])
    EDAPlots.wavelet_energy_heatmap(data_path, feat, graph_dirs["eda"])
    EDAPlots.feature_correlation_heatmap(data_path, feat, graph_dirs["eda"])


def _model_graphs(name, model, y_test, X_test, label_names,
                  feature_names: list, graph_dirs: dict) -> None:
    from ml.plots import EvaluationPlots, FeaturePlots

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    EvaluationPlots.confusion_matrix(y_test, y_pred, label_names, name,
                                     graph_dirs["evaluation"])
    EvaluationPlots.roc_curves(y_test, y_proba, label_names, name,
                               graph_dirs["evaluation"])
    EvaluationPlots.precision_recall_curves(y_test, y_proba, label_names, name,
                                            graph_dirs["evaluation"])
    FeaturePlots.feature_importance(model, feature_names, name,
                                    graph_dirs["features"])
    FeaturePlots.shap_summary(model, X_test, feature_names, name,
                              graph_dirs["features"])


# ── Main orchestrator ─────────────────────────────────────────────────────── #

def run_all(
    data_path: str | Path,
    save_dir: Optional[str | Path] = None,
    cv: int = 5,
    columns_to_drop: Optional[list[str]] = None,
    log_transform: bool = True,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    save_graphs: bool = True,
    graphs_dir: Optional[str | Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every registered model and return a metrics table.

    Parameters
    ----------
    data_path     : Path to the Excel dataset.
    save_dir      : Directory for .joblib model files.
    cv            : Stratified k-fold cross-validation folds.
    columns_to_drop : Feature columns to exclude (missing ones ignored).
    log_transform : Apply log1p to energy features (default True).
    val_size / test_size / random_state : Passed to DataPipeline.
    save_graphs   : Generate and save all IEEE-ready graphs (default True).
    graphs_dir    : Root directory for graphs. Defaults to ``<ml_pkg>/graphs/``.
    verbose       : Print progress to stdout.

    Returns
    -------
    pd.DataFrame — one row per model, columns: Accuracy, Macro-F1,
    Weighted-F1, Macro-Precision, Macro-Recall, CV-Mean-Accuracy,
    CV-Std-Accuracy, CV-Mean-F1, CV-Std-F1, Train-Time(s), Status.
    """
    # ── directories ────────────────────────────────────────────────────── #
    if save_dir is None:
        save_dir = Path(__file__).parent / "saved_models"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if graphs_dir is None:
        graphs_dir = Path(__file__).parent / "graphs"

    # ── data ────────────────────────────────────────────────────────────── #
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

    pipeline.save(save_dir / "pipeline.joblib")

    if verbose:
        print(f"Dataset       : {Path(data_path).name}")
        print(f"Features      : {len(pipeline.get_feature_names())}")
        print(f"Classes       : {label_names}")
        print(f"Train+Val/Test: {len(y_trainval)} / {len(y_test)}")
        print(f"Save dir      : {save_dir}\n")
        print(f"{'Model':<22} {'Acc':>6} {'MacF1':>7} {'WF1':>7} {'CVAcc':>7} {'Train(s)':>9}  Status")
        print("-" * 74)

    # ── EDA graphs (once) ──────────────────────────────────────────────── #
    if save_graphs:
        from ml.plots import make_graph_dirs
        graph_dirs = make_graph_dirs(graphs_dir)
        _eda_graphs(data_path, pipeline, graph_dirs)
        print()

    registry      = _build_registry()
    feature_names = pipeline.get_feature_names()
    rows          = []

    for name, ModelCls in registry.items():
        row: dict  = {"Model": name}
        model      = None

        # ── training (isolated — graph failures must not corrupt this) ── #
        try:
            model = ModelCls()

            t0         = time.perf_counter()
            model.train(X_trainval, y_trainval)
            train_time = time.perf_counter() - t0

            metrics   = model.evaluate(X_test, y_test)
            cv_result = model.cross_validate(X_trainval, y_trainval, cv=cv)

            model.save(save_dir / f"{name}.joblib")

            row.update({
                "Accuracy":         round(metrics["accuracy"],         4),
                "Macro-F1":         round(metrics["macro_f1"],         4),
                "Weighted-F1":      round(metrics["weighted_f1"],      4),
                "Macro-Precision":  round(metrics["macro_precision"],  4),
                "Macro-Recall":     round(metrics["macro_recall"],     4),
                "CV-Mean-Accuracy": round(cv_result["mean_accuracy"],  4),
                "CV-Std-Accuracy":  round(cv_result["std_accuracy"],   4),
                "CV-Mean-F1":       round(cv_result["mean_f1"],        4),
                "CV-Std-F1":        round(cv_result["std_f1"],         4),
                "Train-Time(s)":    round(train_time,                  3),
                "Status":           "OK",
            })

            if verbose:
                print(
                    f"{name:<22} {row['Accuracy']:>6.4f} {row['Macro-F1']:>7.4f}"
                    f" {row['Weighted-F1']:>7.4f} {row['CV-Mean-Accuracy']:>7.4f}"
                    f" {row['Train-Time(s)']:>9.3f}  OK"
                )

        except Exception as exc:
            row.update({k: None for k in [
                "Accuracy", "Macro-F1", "Weighted-F1", "Macro-Precision",
                "Macro-Recall", "CV-Mean-Accuracy", "CV-Std-Accuracy",
                "CV-Mean-F1", "CV-Std-F1", "Train-Time(s)",
            ]})
            row["Status"] = f"ERROR: {exc}"
            if verbose:
                print(f"{name:<22} {'—':>6} {'—':>7} {'—':>7} {'—':>7} {'—':>9}  ERROR: {exc}")
                traceback.print_exc()

        rows.append(row)

        # ── per-model graphs (separate try so failures don't affect results) #
        if save_graphs and model is not None and row["Status"] == "OK":
            try:
                _model_graphs(name, model, y_test, X_test,
                              label_names, feature_names, graph_dirs)
            except Exception as ge:
                warnings.warn(f"[graphs] {name}: {ge}")

    results = pd.DataFrame(rows).set_index("Model")

    # ── model comparison graph (needs all rows) ────────────────────────── #
    if save_graphs:
        from ml.plots import EvaluationPlots
        EvaluationPlots.model_comparison(results, graph_dirs["evaluation"])

    return results


# ── CLI ───────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all fault-classification models.")
    parser.add_argument("--data",       required=True,         help="Path to Combined_Dataset.xlsx")
    parser.add_argument("--save-dir",   default=None,          help="Directory for .joblib files")
    parser.add_argument("--graphs-dir", default=None,          help="Root directory for graphs")
    parser.add_argument("--cv",         type=int, default=5,   help="Cross-validation folds")
    parser.add_argument("--no-graphs",  action="store_true",   help="Skip graph generation")
    parser.add_argument("--no-log",     action="store_true",   help="Disable log1p transform")
    args = parser.parse_args()

    df = run_all(
        data_path=args.data,
        save_dir=args.save_dir,
        graphs_dir=args.graphs_dir,
        cv=args.cv,
        save_graphs=not args.no_graphs,
        log_transform=not args.no_log,
    )
    print("\n── Summary ──")
    print(df.to_string())

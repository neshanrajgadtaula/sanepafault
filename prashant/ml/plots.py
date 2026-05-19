"""plots.py — IEEE-ready visualisations for fault classification papers.

Organised into four stage classes:
    EDAPlots        → eda/       01–03
    EvaluationPlots → evaluation/ 04–07
    TuningPlots     → tuning/    08–09
    FeaturePlots    → features/  10–11
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── IEEE style ─────────────────────────────────────────────────────────── #

_IEEE_RC: dict = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.framealpha": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
}

SINGLE = 3.5    # single-column width (inches)
DOUBLE = 7.16   # double-column width (inches)

# Colorblind-safe, prints well in grayscale
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

_TARGET = "Fault_Label"
_META   = ("Window_No", "Start_SNo", "End_SNo")


# ── Shared helpers ─────────────────────────────────────────────────────── #

def _ctx():
    return plt.rc_context(_IEEE_RC)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [graph] {path.parent.name}/{path.name}")


def _load_raw(
    filepath: Union[str, Path],
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Load Excel, drop metadata, return (X_df with feature_names cols, y_raw str labels)."""
    df = pd.read_excel(filepath)
    meta = [c for c in _META if c in df.columns]
    df = df.drop(columns=meta)
    present = [c for c in feature_names if c in df.columns]
    return df[present], df[_TARGET]


def _color(i: int) -> str:
    return PALETTE[i % len(PALETTE)]


# ══════════════════════════════════════════════════════════════════════════ #
#  Stage 1 — EDA                                                             #
# ══════════════════════════════════════════════════════════════════════════ #

class EDAPlots:
    """Graphs for the dataset / EDA section of the paper."""

    @staticmethod
    def class_distribution(
        filepath: Union[str, Path],
        feature_names: list[str],
        save_dir: Union[str, Path],
    ) -> None:
        """Bar chart of sample count per fault class (01_class_distribution.png)."""
        _, y = _load_raw(filepath, feature_names)
        counts = y.value_counts().sort_index()
        labels = counts.index.tolist()

        with _ctx():
            fig, ax = plt.subplots(figsize=(DOUBLE, 2.8))
            colors = [_color(i) for i in range(len(labels))]
            bars = ax.bar(labels, counts.values, color=colors,
                          edgecolor="white", linewidth=0.5, width=0.65)

            ymax = counts.max()
            for bar, cnt in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + ymax * 0.015,
                        str(cnt), ha="center", va="bottom",
                        fontsize=7, fontweight="bold")

            ax.set_xlabel("Fault Class")
            ax.set_ylabel("Sample Count")
            ax.set_title("Class Distribution")
            ax.set_ylim(0, ymax * 1.13)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            _save(fig, Path(save_dir) / "01_class_distribution.png")

    @staticmethod
    def wavelet_energy_heatmap(
        filepath: Union[str, Path],
        feature_names: list[str],
        save_dir: Union[str, Path],
    ) -> None:
        """Log-mean wavelet energy per fault class × feature heatmap (02_...)."""
        X, y = _load_raw(filepath, feature_names)
        fault_classes = sorted(y.unique(), key=str)

        log_X = np.log1p(X.values.astype(float))
        tmp = pd.DataFrame(log_X, columns=X.columns)
        tmp[_TARGET] = y.values
        per_class = (tmp.groupby(_TARGET)[X.columns.tolist()]
                       .mean()
                       .reindex(fault_classes))

        n_feat = len(feature_names)
        n_cls  = len(fault_classes)

        with _ctx():
            fig, ax = plt.subplots(figsize=(DOUBLE, max(2.5, n_cls * 0.42)))
            im = ax.imshow(per_class.values, aspect="auto", cmap="YlOrRd")

            ax.set_xticks(range(n_feat))
            ax.set_xticklabels(X.columns.tolist(), rotation=90, fontsize=6.0)
            ax.set_yticks(range(n_cls))
            ax.set_yticklabels(fault_classes, fontsize=8)

            cbar = fig.colorbar(im, ax=ax, pad=0.01)
            cbar.set_label("log(1+Energy)", fontsize=8)
            ax.set_title("Mean Wavelet Energy per Fault Class (log scale)")
            plt.tight_layout()
            _save(fig, Path(save_dir) / "02_wavelet_energy_heatmap.png")

    @staticmethod
    def feature_correlation_heatmap(
        filepath: Union[str, Path],
        feature_names: list[str],
        save_dir: Union[str, Path],
    ) -> None:
        """Pearson correlation heatmap of the retained feature set (03_...)."""
        X, _ = _load_raw(filepath, feature_names)
        corr = X.corr()
        n = len(feature_names)

        with _ctx():
            sz = max(4.5, n * 0.22)
            fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
            im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(feature_names, rotation=90, fontsize=5.5)
            ax.set_yticklabels(feature_names, fontsize=5.5)

            cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Pearson r", fontsize=8)
            ax.set_title(f"Feature Correlation Matrix ({n} features)")
            plt.tight_layout()
            _save(fig, Path(save_dir) / "03_feature_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════ #
#  Stage 2 — Evaluation                                                      #
# ══════════════════════════════════════════════════════════════════════════ #

class EvaluationPlots:
    """Per-model and cross-model evaluation graphs."""

    @staticmethod
    def confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        label_names: list[str],
        model_name: str,
        save_dir: Union[str, Path],
    ) -> None:
        """Normalised confusion matrix (row = true class) (04_...)."""
        from sklearn.metrics import confusion_matrix as _cm
        cm = _cm(y_true, y_pred)
        # Row-normalise to fractions (handles class-imbalance visually)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_norm = np.where(cm.sum(axis=1, keepdims=True) == 0, 0,
                               cm / cm.sum(axis=1, keepdims=True))
        n = cm.shape[0]
        shown = label_names[:n] if len(label_names) >= n else [str(i) for i in range(n)]

        with _ctx():
            sz = max(3.5, n * 0.48)
            fig, ax = plt.subplots(figsize=(sz, sz * 0.92))
            im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(shown, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(shown, fontsize=7)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title(f"Confusion Matrix — {model_name}")

            thresh = 0.5
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=6,
                            color="white" if cm_norm[i, j] > thresh else "black")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Fraction", fontsize=8)
            plt.tight_layout()
            _save(fig, Path(save_dir) / f"04_confusion_matrix_{model_name}.png")

    @staticmethod
    def model_comparison(
        results_df: pd.DataFrame,
        save_dir: Union[str, Path],
    ) -> None:
        """Grouped bar chart comparing all models on key metrics (05_...)."""
        # Accept column names from both run_all and run_tuning
        _candidates = [
            ("Accuracy",         "Test-Accuracy"),
            ("Macro-F1",         "Test-Macro-F1"),
            ("Weighted-F1",      "Test-Weighted-F1"),
        ]
        metric_cols = []
        for a, b in _candidates:
            if a in results_df.columns:
                metric_cols.append(a)
            elif b in results_df.columns:
                metric_cols.append(b)

        if not metric_cols:
            warnings.warn("model_comparison: no recognised metric columns found.")
            return

        df = results_df[metric_cols].dropna().astype(float)
        if df.empty:
            warnings.warn("model_comparison: no successful model results to plot.")
            return
        models   = df.index.tolist()
        n_m      = len(models)
        n_met    = len(metric_cols)
        x        = np.arange(n_m)
        width    = 0.75 / n_met

        with _ctx():
            fig, ax = plt.subplots(figsize=(DOUBLE, 3.4))
            for i, col in enumerate(metric_cols):
                offset = (i - n_met / 2 + 0.5) * width
                vals   = df[col].values
                ax.bar(x + offset, vals, width * 0.92,
                       label=col.replace("Test-", ""),
                       color=_color(i), edgecolor="white", linewidth=0.4)

            lo = max(0, float(df.values.min()) - 0.03)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7.5)
            ax.set_ylabel("Score")
            ax.set_ylim(lo, 1.005)
            ax.set_title("Model Comparison")
            ax.legend(loc="lower right")
            plt.tight_layout()
            _save(fig, Path(save_dir) / "05_model_comparison.png")

    @staticmethod
    def roc_curves(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        label_names: list[str],
        model_name: str,
        save_dir: Union[str, Path],
    ) -> None:
        """One-vs-Rest ROC curves for every fault class (06_...)."""
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        classes  = np.arange(len(label_names))
        y_bin    = label_binarize(y_true, classes=classes)
        if len(label_names) == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])

        with _ctx():
            fig, ax = plt.subplots(figsize=(SINGLE + 0.6, SINGLE + 0.6))
            present = np.unique(y_true)
            for i in present:
                if i >= y_proba.shape[1]:
                    continue
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc     = auc(fpr, tpr)
                lbl = label_names[i] if i < len(label_names) else str(i)
                ax.plot(fpr, tpr, color=_color(int(i)), linewidth=1.0,
                        label=f"{lbl} ({roc_auc:.3f})")

            ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curves (OvR) — {model_name}")
            ax.legend(fontsize=6.5, loc="lower right",
                      ncol=max(1, len(present) // 8))
            plt.tight_layout()
            _save(fig, Path(save_dir) / f"06_roc_curves_{model_name}.png")

    @staticmethod
    def precision_recall_curves(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        label_names: list[str],
        model_name: str,
        save_dir: Union[str, Path],
    ) -> None:
        """Per-class Precision-Recall curves (07_...)."""
        from sklearn.metrics import average_precision_score, precision_recall_curve
        from sklearn.preprocessing import label_binarize

        classes = np.arange(len(label_names))
        y_bin   = label_binarize(y_true, classes=classes)
        if len(label_names) == 2:
            y_bin = np.hstack([1 - y_bin, y_bin])

        with _ctx():
            fig, ax = plt.subplots(figsize=(SINGLE + 0.6, SINGLE + 0.6))
            present = np.unique(y_true)
            for i in present:
                if i >= y_proba.shape[1]:
                    continue
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                ap  = average_precision_score(y_bin[:, i], y_proba[:, i])
                lbl = label_names[i] if i < len(label_names) else str(i)
                ax.plot(rec, prec, color=_color(int(i)), linewidth=1.0,
                        label=f"{lbl} (AP={ap:.3f})")

            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curves — {model_name}")
            ax.legend(fontsize=6.5, loc="lower left",
                      ncol=max(1, len(present) // 8))
            plt.tight_layout()
            _save(fig, Path(save_dir) / f"07_precision_recall_{model_name}.png")


# ══════════════════════════════════════════════════════════════════════════ #
#  Stage 3 — Tuning                                                          #
# ══════════════════════════════════════════════════════════════════════════ #

class TuningPlots:
    """Optuna study visualisations for the hyperparameter tuning section."""

    @staticmethod
    def optimization_history(
        study,
        model_name: str,
        save_dir: Union[str, Path],
    ) -> None:
        """Best val macro-F1 vs. trial number (08_...)."""
        try:
            from optuna.visualization.matplotlib import plot_optimization_history
        except ImportError:
            warnings.warn("optuna matplotlib visualisation not available.")
            return

        with _ctx():
            ax = plot_optimization_history(study)
            ax.set_title(f"Optimisation History — {model_name}")
            ax.set_xlabel("Trial Number")
            ax.set_ylabel("Best Val Macro-F1")
            fig = ax.figure
            fig.set_size_inches(SINGLE + 1.0, 3.0)
            _save(fig, Path(save_dir) / f"08_optimization_history_{model_name}.png")

    @staticmethod
    def param_importance(
        study,
        model_name: str,
        save_dir: Union[str, Path],
    ) -> None:
        """Optuna hyperparameter importance bar chart (09_...)."""
        try:
            from optuna.visualization.matplotlib import plot_param_importances
        except ImportError:
            warnings.warn("optuna matplotlib visualisation not available.")
            return

        # Need at least a few completed trials for importance calculation
        completed = [t for t in study.trials if t.state.name == "COMPLETE"]
        if len(completed) < 4:
            warnings.warn(f"{model_name}: too few trials for param importance — skipping.")
            return

        with _ctx():
            ax = plot_param_importances(study)
            ax.set_title(f"Hyperparameter Importance — {model_name}")
            fig = ax.figure
            fig.set_size_inches(SINGLE + 1.0, 3.0)
            _save(fig, Path(save_dir) / f"09_param_importance_{model_name}.png")


# ══════════════════════════════════════════════════════════════════════════ #
#  Stage 4 — Feature Analysis                                                #
# ══════════════════════════════════════════════════════════════════════════ #

class FeaturePlots:
    """Feature importance and SHAP visualisations."""

    @staticmethod
    def feature_importance(
        model,
        feature_names: list[str],
        model_name: str,
        save_dir: Union[str, Path],
        top_n: int = 20,
    ) -> None:
        """Horizontal bar chart of built-in feature importances (10_...)."""
        est = model.model
        if hasattr(est, "named_steps"):
            est = est.named_steps["clf"]

        importances: Optional[np.ndarray] = None
        if hasattr(est, "feature_importances_"):
            importances = np.array(est.feature_importances_)
        elif hasattr(est, "coef_"):
            importances = np.abs(np.array(est.coef_)).mean(axis=0)

        if importances is None:
            warnings.warn(f"{model_name}: no feature_importances_ or coef_ — skipping.")
            return

        n      = min(top_n, len(feature_names))
        idx    = np.argsort(importances)[::-1][:n]
        vals   = importances[idx]
        names  = [feature_names[i] if i < len(feature_names) else f"f{i}" for i in idx]

        with _ctx():
            fig, ax = plt.subplots(figsize=(SINGLE + 0.5, max(2.8, n * 0.28)))
            y_pos = np.arange(n)
            ax.barh(y_pos, vals,
                    color=[_color(i) for i in range(n)],
                    edgecolor="white", height=0.72)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlabel("Importance Score")
            ax.set_title(f"Feature Importance (Top {n}) — {model_name}")
            plt.tight_layout()
            _save(fig, Path(save_dir) / f"10_feature_importance_{model_name}.png")

    @staticmethod
    def shap_summary(
        model,
        X: np.ndarray,
        feature_names: list[str],
        model_name: str,
        save_dir: Union[str, Path],
        max_samples: int = 300,
    ) -> None:
        """SHAP mean-|value| bar chart for tree-based models (11_...)."""
        try:
            import shap
        except ImportError:
            warnings.warn("shap not installed — skipping. pip install shap")
            return

        est    = model.model
        scaler = None
        if hasattr(est, "named_steps"):
            scaler = est.named_steps.get("scaler")
            est    = est.named_steps["clf"]

        if not hasattr(est, "feature_importances_"):
            warnings.warn(
                f"{model_name}: SHAP TreeExplainer requires a tree-based model — skipping."
            )
            return

        X_samp = X[:max_samples]
        if scaler is not None:
            X_samp = scaler.transform(X_samp)

        try:
            explainer  = shap.TreeExplainer(est)
            shap_vals  = explainer.shap_values(X_samp)

            # Multi-class: shap_vals is a list of (n_samples, n_features) arrays
            # Average absolute SHAP across classes
            if isinstance(shap_vals, list):
                sv = np.mean(np.abs(np.array(shap_vals)), axis=0)
            else:
                sv = shap_vals

            with _ctx():
                shap.summary_plot(
                    sv, X_samp,
                    feature_names=feature_names,
                    plot_type="bar",
                    show=False,
                    max_display=min(20, len(feature_names)),
                )
                fig = plt.gcf()
                fig.set_size_inches(SINGLE + 0.5, max(3.0, min(20, len(feature_names)) * 0.28))
                ax  = fig.axes[0]
                ax.set_title(f"SHAP Feature Importance — {model_name}",
                             fontsize=10, fontweight="bold")
                _save(fig, Path(save_dir) / f"11_shap_summary_{model_name}.png")

        except Exception as exc:
            warnings.warn(f"SHAP failed for {model_name}: {exc}")


# ══════════════════════════════════════════════════════════════════════════ #
#  Convenience: create all graph subdirectories                              #
# ══════════════════════════════════════════════════════════════════════════ #

def make_graph_dirs(base: Union[str, Path]) -> dict[str, Path]:
    """Create and return the four stage subdirectories under *base*."""
    base = Path(base)
    dirs = {
        "eda":        base / "eda",
        "evaluation": base / "evaluation",
        "tuning":     base / "tuning",
        "features":   base / "features",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

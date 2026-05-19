from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from ml.base_model import BaseModel


class Evaluator:
    """Shared evaluation utilities used by all model wrappers."""

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        labels: Optional[list[str]] = None,
    ) -> dict:
        """Compute a standard set of classification metrics.

        Returns
        -------
        dict with keys:
            accuracy, macro_precision, macro_recall, macro_f1,
            weighted_f1, confusion_matrix, classification_report
        """
        target_names = labels if labels else None
        report = classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            "classification_report": report,
        }

    # ------------------------------------------------------------------ #
    #  Confusion matrix plot                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred)
        tick_labels = labels if labels else list(range(cm.shape[0]))

        fig, ax = plt.subplots(figsize=(max(6, len(tick_labels)), max(5, len(tick_labels) - 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=tick_labels,
            yticklabels=tick_labels,
            title=title,
            ylabel="True label",
            xlabel="Predicted label",
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=9)

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()

    # ------------------------------------------------------------------ #
    #  Cross-validation                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def cross_validate(
        model: "BaseModel",
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> dict:
        """Stratified k-fold cross-validation.

        Re-instantiates a fresh copy of the model each fold using
        ``type(model)(**model.get_params())`` so the original trained
        model is not mutated.

        Returns
        -------
        dict with keys: cv_accuracy_scores, cv_f1_scores,
                        mean_accuracy, std_accuracy,
                        mean_f1, std_f1
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        acc_scores, f1_scores = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = type(model)(**model.get_params())
            fold_model.train(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)

            acc_scores.append(accuracy_score(y_val, y_pred))
            f1_scores.append(f1_score(y_val, y_pred, average="macro", zero_division=0))

        return {
            "cv_accuracy_scores": acc_scores,
            "cv_f1_scores": f1_scores,
            "mean_accuracy": float(np.mean(acc_scores)),
            "std_accuracy": float(np.std(acc_scores)),
            "mean_f1": float(np.mean(f1_scores)),
            "std_f1": float(np.std(f1_scores)),
        }

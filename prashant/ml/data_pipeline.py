from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataPipeline:
    """Load, preprocess, and split the fault-classification dataset.

    Feature list is derived dynamically from the loaded file so the
    pipeline never breaks when columns are added or removed between
    dataset versions.  Any ``columns_to_drop`` that are absent are
    silently skipped.
    """

    _META_COLS = ("Window_No", "Start_SNo", "End_SNo")
    _TARGET_COL = "Fault_Label"

    def __init__(
        self,
        log_transform: bool = True,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42,
        columns_to_drop: Optional[list[str]] = None,
    ) -> None:
        self.log_transform = log_transform
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.columns_to_drop: list[str] = columns_to_drop or []

        self._label_encoder = LabelEncoder()
        self.feature_names_: list[str] = []   # set after load_and_prepare

    # ------------------------------------------------------------------ #

    def load_and_prepare(
        self, filepath: str | Path
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load the Excel file and return (X_train, X_val, X_test, y_train, y_val, y_test).

        Three-way stratified split: train / validation / test.
        - Train  : used to fit the model each optuna trial and final training.
        - Val    : used by the tuning objective (never seen by the test evaluator).
        - Test   : held out completely; only touched for final reporting.

        All arrays are numpy; labels are integer-encoded.
        """
        df = pd.read_excel(filepath)

        # ── 1. Drop metadata columns that exist ──────────────────────────
        meta_to_drop = [c for c in self._META_COLS if c in df.columns]
        df = df.drop(columns=meta_to_drop)

        # ── 2. Isolate energy features ────────────────────────────────────
        feature_cols = [c for c in df.columns if c.endswith("_Energy")]

        # ── 3. Drop redundant columns (intersection only) ─────────────────
        drop_present = [c for c in self.columns_to_drop if c in feature_cols]
        if drop_present:
            feature_cols = [c for c in feature_cols if c not in drop_present]

        self.feature_names_ = feature_cols

        X = df[feature_cols].copy()
        y_raw = df[self._TARGET_COL].values

        # ── 4. log1p transform ────────────────────────────────────────────
        if self.log_transform:
            X = np.log1p(X.values.astype(float))
        else:
            X = X.values.astype(float)

        # ── 5. Encode labels ──────────────────────────────────────────────
        y = self._label_encoder.fit_transform(y_raw)

        # ── 6. First split: carve out test set ───────────────────────────
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # ── 7. Second split: carve validation from remaining ─────────────
        # val_size is expressed as fraction of the *original* dataset, so
        # we adjust to make it a fraction of the trainval portion.
        val_fraction_of_trainval = self.val_size / (1.0 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction_of_trainval,
            random_state=self.random_state,
            stratify=y_trainval,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ------------------------------------------------------------------ #

    def align_features(self, df: pd.DataFrame) -> np.ndarray:
        """Align a new DataFrame to the feature set seen during training.

        Missing columns are filled with 0; extra columns are dropped.
        Applies log1p if enabled.  Useful for inference on new data.
        """
        if not self.feature_names_:
            raise RuntimeError("Call load_and_prepare first.")
        aligned = pd.DataFrame(index=df.index)
        for col in self.feature_names_:
            if col in df.columns:
                aligned[col] = df[col]
            else:
                warnings.warn(f"Column '{col}' missing from input — filled with 0.")
                aligned[col] = 0.0
        X = aligned.values.astype(float)
        if self.log_transform:
            X = np.log1p(X)
        return X

    # ------------------------------------------------------------------ #

    def get_label_names(self) -> list[str]:
        """Return class names in the order used by the label encoder."""
        return list(self._label_encoder.classes_)

    def get_feature_names(self) -> list[str]:
        """Return the list of feature columns actually used."""
        return list(self.feature_names_)

    def decode_labels(self, y: np.ndarray) -> np.ndarray:
        """Convert integer labels back to original string labels."""
        return self._label_encoder.inverse_transform(y)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "DataPipeline":
        return joblib.load(path)

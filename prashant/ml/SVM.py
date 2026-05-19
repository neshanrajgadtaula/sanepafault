from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ml.base_model import BaseModel
from ml.evaluator import Evaluator


class SVMModel(BaseModel):
    """Wrapper around sklearn SVC with internal StandardScaler."""

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: str = "scale",
        degree: int = 3,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self._params = dict(
            C=C,
            kernel=kernel,
            gamma=gamma,
            degree=degree,
            random_state=random_state,
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, **self._params)),
        ])

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        return Evaluator.compute_metrics(y, self.predict(X), self.predict_proba(X))

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SVMModel":
        return joblib.load(path)

    def get_params(self) -> dict:
        return dict(self._params)

    def set_params(self, **params) -> None:
        self._params.update(params)
        self.model.named_steps["clf"].set_params(**params)

    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> dict:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "C": trial.suggest_float("C", 0.01, 100.0, log=True),
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "linear"]),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, **params, random_state=self._params["random_state"])),
            ])
            pipe.fit(X_train, y_train)
            from sklearn.metrics import f1_score
            return f1_score(y_val, pipe.predict(X_val), average="macro", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best["random_state"] = self._params["random_state"]
        self.set_params(**best)
        return {"best_params": best, "best_val_f1": study.best_value, "study": study}

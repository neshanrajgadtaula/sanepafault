from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.base_model import BaseModel
from ml.evaluator import Evaluator


class KNNModel(BaseModel):
    """Wrapper around sklearn KNeighborsClassifier with internal StandardScaler."""

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        self._params = dict(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
            n_jobs=n_jobs,
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(**self._params)),
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
    def load(cls, path: str) -> "KNNModel":
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
                "n_neighbors": trial.suggest_int("n_neighbors", 1, 20),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
                "metric": trial.suggest_categorical("metric", ["minkowski", "euclidean", "manhattan"]),
                "p": trial.suggest_int("p", 1, 2),
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(**params, n_jobs=self._params["n_jobs"])),
            ])
            pipe.fit(X_train, y_train)
            from sklearn.metrics import f1_score
            return f1_score(y_val, pipe.predict(X_val), average="macro", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best["n_jobs"] = self._params["n_jobs"]
        self.set_params(**best)
        return {"best_params": best, "best_val_f1": study.best_value, "study": study}

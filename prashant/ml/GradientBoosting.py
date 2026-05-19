from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from ml.base_model import BaseModel
from ml.evaluator import Evaluator


class GradientBoostingModel(BaseModel):
    """Wrapper around sklearn GradientBoostingClassifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
        )
        self.model = GradientBoostingClassifier(**self._params)

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
    def load(cls, path: str) -> "GradientBoostingModel":
        return joblib.load(path)

    def get_params(self) -> dict:
        return dict(self._params)

    def set_params(self, **params) -> None:
        self._params.update(params)
        self.model.set_params(**params)

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
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            m = GradientBoostingClassifier(**params, random_state=self._params["random_state"])
            m.fit(X_train, y_train)
            from sklearn.metrics import f1_score
            return f1_score(y_val, m.predict(X_val), average="macro", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best["random_state"] = self._params["random_state"]
        self.set_params(**best)
        return {"best_params": best, "best_val_f1": study.best_value, "study": study}

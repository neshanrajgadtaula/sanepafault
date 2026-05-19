from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.base_model import BaseModel
from ml.evaluator import Evaluator


class LogisticRegressionModel(BaseModel):
    """Wrapper around sklearn LogisticRegression with internal StandardScaler."""

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self._params = dict(
            C=C,
            max_iter=max_iter,
            solver=solver,
            random_state=random_state,
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**self._params)),
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
    def load(cls, path: str) -> "LogisticRegressionModel":
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
                "C": trial.suggest_float("C", 1e-3, 100.0, log=True),
                "solver": trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                "max_iter": trial.suggest_int("max_iter", 200, 2000, step=200),
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(**params, random_state=self._params["random_state"])),
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

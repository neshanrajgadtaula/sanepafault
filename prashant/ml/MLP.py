from __future__ import annotations

from typing import Optional

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.base_model import BaseModel
from ml.evaluator import Evaluator


class MLPModel(BaseModel):
    """Wrapper around sklearn MLPClassifier with internal StandardScaler."""

    def __init__(
        self,
        hidden_layer_sizes: tuple = (100, 50),
        activation: str = "relu",
        solver: str = "adam",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        random_state: int = 42,
        **kwargs,
    ) -> None:
        self._params = dict(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**self._params)),
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
    def load(cls, path: str) -> "MLPModel":
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

        # Predefined architectures to search over
        _architectures = [
            (64,), (128,), (256,),
            (64, 32), (128, 64), (256, 128),
            (128, 64, 32), (256, 128, 64),
        ]

        def objective(trial):
            arch = _architectures[trial.suggest_int("arch_idx", 0, len(_architectures) - 1)]
            params = {
                "hidden_layer_sizes": arch,
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "max_iter": trial.suggest_int("max_iter", 200, 800, step=100),
            }
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(**params, random_state=self._params["random_state"])),
            ])
            pipe.fit(X_train, y_train)
            from sklearn.metrics import f1_score
            return f1_score(y_val, pipe.predict(X_val), average="macro", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best_raw = study.best_params
        arch_idx = best_raw.pop("arch_idx")
        best = {**best_raw, "hidden_layer_sizes": _architectures[arch_idx]}
        best["random_state"] = self._params["random_state"]
        self.set_params(**best)
        return {"best_params": best, "best_val_f1": study.best_value, "study": study}

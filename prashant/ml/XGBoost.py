from __future__ import annotations

from typing import Optional

import joblib
import numpy as np

from ml.base_model import BaseModel
from ml.evaluator import Evaluator


class XGBoostModel(BaseModel):
    """Wrapper around xgboost XGBClassifier.

    Requires: pip install xgboost
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        eval_metric: str = "mlogloss",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs,
    ) -> None:
        self._params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._import_xgb()
        self.model = self._XGBClassifier(**self._params)

    @staticmethod
    def _import_xgb():
        try:
            from xgboost import XGBClassifier  # noqa: F401
            XGBoostModel._XGBClassifier = XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Run: pip install xgboost"
            ) from exc

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
    def load(cls, path: str) -> "XGBoostModel":
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
        XGBClassifier = self._XGBClassifier

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True),
            }
            m = XGBClassifier(
                **params,
                eval_metric=self._params["eval_metric"],
                random_state=self._params["random_state"],
                n_jobs=self._params["n_jobs"],
            )
            m.fit(X_train, y_train)
            from sklearn.metrics import f1_score
            return f1_score(y_val, m.predict(X_val), average="macro", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

        best = study.best_params
        best["eval_metric"] = self._params["eval_metric"]
        best["random_state"] = self._params["random_state"]
        best["n_jobs"] = self._params["n_jobs"]
        self.set_params(**best)
        return {"best_params": best, "best_val_f1": study.best_value, "study": study}

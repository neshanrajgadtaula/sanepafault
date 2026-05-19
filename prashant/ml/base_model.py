from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class BaseModel(ABC):
    """Abstract base class that every classifier wrapper must implement."""

    # ------------------------------------------------------------------ #
    #  Core training & inference                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class-index predictions for X."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class-probability matrix for X (shape: n_samples × n_classes)."""

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Compute and return evaluation metrics on (X, y).

        Returned dict must contain at minimum:
            accuracy, macro_f1, weighted_f1, confusion_matrix,
            classification_report
        """

    @abstractmethod
    def tune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 50,
        timeout: Optional[int] = None,
    ) -> dict:
        """Run Bayesian hyperparameter search (optuna) using the validation set.

        Each trial trains on ``X_train`` / ``y_train`` and scores on
        ``X_val`` / ``y_val`` (macro-F1).  The best params are applied
        via :meth:`set_params` so a subsequent :meth:`train` call uses them.

        Parameters
        ----------
        X_train, y_train : training split (fit each trial here)
        X_val,   y_val   : validation split (score each trial here — never test)
        n_trials         : number of optuna trials
        timeout          : optional wall-clock limit in seconds

        Returns
        -------
        dict with keys: best_params, best_val_f1, study
        """

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
    ) -> dict:
        """Run stratified k-fold cross-validation.

        Delegates to :class:`~ml.evaluator.Evaluator` so the logic lives
        in one place.  Concrete subclasses inherit this for free.
        """
        from ml.evaluator import Evaluator
        return Evaluator.cross_validate(self, X, y, cv=cv)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def save(self, path: str) -> None:
        """Serialise the trained model to *path* (joblib format)."""

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Deserialise a model from *path* and return a ready instance."""

    # ------------------------------------------------------------------ #
    #  Hyper-parameter access                                              #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_params(self) -> dict:
        """Return the model's current hyper-parameters as a dict."""

    @abstractmethod
    def set_params(self, **params) -> None:
        """Update hyper-parameters.  Must be called before :meth:`train`."""

    # ------------------------------------------------------------------ #
    #  Dunder helpers                                                      #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_params()})"

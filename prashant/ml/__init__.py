"""ml — Classical ML classifiers for power system fault detection.

Convenience imports so notebook code stays short:

    from ml import RandomForestModel, DataPipeline, Evaluator

Each model class implements :class:`~ml.base_model.BaseModel` (ABC) with
identical public methods: train, predict, predict_proba, evaluate,
cross_validate, save, load, get_params, set_params.
"""
from ml.base_model import BaseModel
from ml.data_pipeline import DataPipeline
from ml.evaluator import Evaluator
from ml.plots import EDAPlots, EvaluationPlots, FeaturePlots, TuningPlots, make_graph_dirs

from ml.AdaBoost import AdaBoostModel
from ml.DecisionTree import DecisionTreeModel
from ml.ExtraTrees import ExtraTreesModel
from ml.GradientBoosting import GradientBoostingModel
from ml.KNN import KNNModel
from ml.LogisticRegression import LogisticRegressionModel
from ml.MLP import MLPModel
from ml.NaiveBayes import NaiveBayesModel
from ml.RandomForest import RandomForestModel
from ml.SVM import SVMModel

try:
    from ml.XGBoost import XGBoostModel
except ImportError:
    pass  # xgboost not installed

__all__ = [
    "BaseModel",
    "DataPipeline",
    "Evaluator",
    "AdaBoostModel",
    "DecisionTreeModel",
    "ExtraTreesModel",
    "GradientBoostingModel",
    "KNNModel",
    "LogisticRegressionModel",
    "MLPModel",
    "NaiveBayesModel",
    "RandomForestModel",
    "SVMModel",
    "XGBoostModel",
]

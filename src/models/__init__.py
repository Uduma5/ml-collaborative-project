"""
Models Package
Contains all machine learning model implementations.
"""

# Verah's Models
from .decision_tree import DecisionTreeModel
from .kmeans import KMeansModel
from .svm import SVMModel

# Sandra's Models
from .linear_regression import LinearRegressionModel
from .logistic_regression import LogisticRegressionModel
from .knn import KNNModel

# Uduma's Models
from .random_forest import RandomForestModel

__all__ = [
    # Verah's Models
    'DecisionTreeModel',
    'KMeansModel',
    'SVMModel',
    # Sandra's Models
    'LinearRegressionModel',
    'LogisticRegressionModel',
    'KNNModel',
    # Uduma's Models
    'RandomForestModel'
]
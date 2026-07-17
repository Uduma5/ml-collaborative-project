"""
Training Pipeline
Trains all ML models on their respective datasets and saves results.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import pickle
import numpy as np
from datetime import datetime

from src.data_preprocessing import (
    load_iris_data,
    load_breast_cancer_data,
    load_wine_data,
    load_california_housing_data,
    split_and_scale_data
)

from src.models import (
    LinearRegressionModel,
    LogisticRegressionModel,
    KNNModel,
    DecisionTreeModel,
    SVMModel,
    RandomForestModel,
    KMeansModel
)


def train_all_models():
    """
    Train all models and save results.
    """
    print("="*60)
    print("🧠 TRAINING ALL MODELS")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    trained_models = {}
    
    # ============================================
    # 1. LINEAR REGRESSION - California Housing
    # ============================================
    print("\n" + "="*60)
    print("📊 1. Linear Regression - California Housing")
    print("="*60)
    
    X, y, feature_names = load_california_housing_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    model_lr = LinearRegressionModel()
    model_lr.train(X_train, y_train)
    results_lr = model_lr.evaluate(X_test, y_test)
    
    results['linear_regression'] = {
        'dataset': 'California Housing',
        'metrics': results_lr,
        'coefficients': model_lr.get_coefficients(feature_names)
    }
    trained_models['linear_regression'] = model_lr
    
    # ============================================
    # 2. LOGISTIC REGRESSION - Iris
    # ============================================
    print("\n" + "="*60)
    print("📊 2. Logistic Regression - Iris")
    print("="*60)
    
    X, y, feature_names, target_names = load_iris_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    model_logreg = LogisticRegressionModel()
    model_logreg.train(X_train, y_train)
    results_logreg = model_logreg.evaluate(X_test, y_test)
    
    results['logistic_regression'] = {
        'dataset': 'Iris',
        'metrics': results_logreg
    }
    trained_models['logistic_regression'] = model_logreg
    
    # ============================================
    # 3. KNN - Iris
    # ============================================
    print("\n" + "="*60)
    print("📊 3. K-Nearest Neighbors - Iris")
    print("="*60)
    
    X, y, feature_names, target_names = load_iris_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    model_knn = KNNModel(n_neighbors=5)
    model_knn.train(X_train, y_train)
    results_knn = model_knn.evaluate(X_test, y_test)
    
    results['knn'] = {
        'dataset': 'Iris',
        'metrics': results_knn
    }
    trained_models['knn'] = model_knn
    
    # ============================================
    # 4. DECISION TREE - Wine
    # ============================================
    print("\n" + "="*60)
    print("📊 4. Decision Tree - Wine")
    print("="*60)
    
    X, y, feature_names, target_names = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    model_dt = DecisionTreeModel(max_depth=5)
    model_dt.train(X_train, y_train)
    results_dt = model_dt.evaluate(X_test, y_test)
    
    results['decision_tree'] = {
        'dataset': 'Wine',
        'metrics': results_dt,
        'feature_importance': model_dt.get_feature_importance(feature_names)
    }
    trained_models['decision_tree'] = model_dt
    
    # ============================================
    # 5. SVM - Breast Cancer
    # ============================================
    print("\n" + "="*60)
    print("📊 5. Support Vector Machine - Breast Cancer")
    print("="*60)
    
    X, y, feature_names, target_names = load_breast_cancer_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    model_svm = SVMModel(kernel='rbf', C=1.0)
    model_svm.train(X_train, y_train)
    results_svm = model_svm.evaluate(X_test, y_test)
    
    results['svm'] = {
        'dataset': 'Breast Cancer',
        'metrics': results_svm
    }
    trained_models['svm'] = model_svm
    
    # ============================================
    # 6. RANDOM FOREST - Wine
    # ============================================
    print("\n" + "="*60)
    print("📊 6. Random Forest - Wine")
    print("="*60)
    
    X, y, feature_names, target_names = load_wine_data()
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
    
    model_rf = RandomForestModel(n_estimators=100)
    model_rf.train(X_train, y_train)
    results_rf = model_rf.evaluate(X_test, y_test)
    
    results['random_forest'] = {
        'dataset': 'Wine',
        'metrics': results_rf,
        'feature_importance': model_rf.get_feature_importance(feature_names)
    }
    trained_models['random_forest'] = model_rf
    
    # ============================================
    # 7. K-MEANS - Iris
    # ============================================
    print("\n" + "="*60)
    print("📊 7. K-Means Clustering - Iris")
    print("="*60)
    
    X, y, feature_names, target_names = load_iris_data()
    
    model_kmeans = KMeansModel(n_clusters=3)
    model_kmeans.train(X)
    results_kmeans = model_kmeans.evaluate(X)
    
    results['kmeans'] = {
        'dataset': 'Iris',
        'metrics': results_kmeans
    }
    trained_models['kmeans'] = model_kmeans
    
    # ============================================
    # SAVE RESULTS
    # ============================================
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    # Save results as JSON
    # Clean results for JSON serialization
    results_clean = {}
    for model_name, model_results in results.items():
        results_clean[model_name] = {
            'dataset': model_results['dataset'],
            'metrics': {}
        }
        for metric, value in model_results['metrics'].items():
            if isinstance(value, (np.ndarray, list)):
                if metric == 'confusion_matrix':
                    results_clean[model_name]['metrics'][metric] = value.tolist() if hasattr(value, 'tolist') else value
                elif metric == 'predictions':
                    results_clean[model_name]['metrics'][metric] = value.tolist() if hasattr(value, 'tolist') else value
                elif metric == 'classification_report':
                    # Convert classification report to serializable format
                    report = {}
                    for key, val in value.items():
                        if isinstance(val, dict):
                            report[key] = {k: float(v) if isinstance(v, (np.float64, float)) else v for k, v in val.items()}
                        else:
                            report[key] = float(val) if isinstance(val, (np.float64, float)) else val
                    results_clean[model_name]['metrics'][metric] = report
                else:
                    results_clean[model_name]['metrics'][metric] = value.tolist() if hasattr(value, 'tolist') else value
            elif isinstance(value, (np.float64, float)):
                results_clean[model_name]['metrics'][metric] = float(value)
            else:
                results_clean[model_name]['metrics'][metric] = value
        
        # Add additional info if available
        if 'coefficients' in model_results:
            results_clean[model_name]['coefficients'] = model_results['coefficients']
        if 'feature_importance' in model_results:
            results_clean[model_name]['feature_importance'] = model_results['feature_importance']
    
    # Save JSON
    with open('models/training_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    print("✅ Results saved to models/training_results.json")
    
    # Save models
    for name, model in trained_models.items():
        model_path = f'models/{name}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model.model, f)
        print(f"✅ Model saved: {model_path}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return results, trained_models


if __name__ == "__main__":
    train_all_models()
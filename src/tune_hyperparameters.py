"""
Hyperparameter Tuning Module
GridSearchCV and RandomSearchCV implementations for model optimization.
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import json


class HyperparameterTuner:
    """
    Hyperparameter tuning wrapper using GridSearchCV and RandomizedSearchCV.
    """
    
    def __init__(self, model, param_grid, cv=5, scoring='accuracy', n_iter=None):
        """
        Initialize tuner.
        
        Args:
            model: Scikit-learn model
            param_grid (dict): Parameter grid for search
            cv (int): Cross-validation folds
            scoring (str): Scoring metric
            n_iter (int): Number of iterations for RandomizedSearchCV
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.n_iter = n_iter
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.cv_results = None
    
    def grid_search(self, X_train, y_train):
        """
        Perform GridSearchCV.
        """
        print(f"\n🔍 Performing GridSearchCV...")
        print(f"   - Parameters: {len(self.param_grid)} combinations")
        
        grid = GridSearchCV(
            self.model,
            self.param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            verbose=0
        )
        
        grid.fit(X_train, y_train)
        
        self.best_model = grid.best_estimator_
        self.best_params = grid.best_params_
        self.best_score = grid.best_score_
        self.cv_results = grid.cv_results_
        
        print(f"   - Best Score: {self.best_score:.4f}")
        print(f"   - Best Parameters: {self.best_params}")
        
        return self
    
    def random_search(self, X_train, y_train):
        """
        Perform RandomizedSearchCV.
        """
        if self.n_iter is None:
            self.n_iter = 20
        
        print(f"\n🔍 Performing RandomizedSearchCV...")
        print(f"   - Iterations: {self.n_iter}")
        
        random = RandomizedSearchCV(
            self.model,
            self.param_grid,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        random.fit(X_train, y_train)
        
        self.best_model = random.best_estimator_
        self.best_params = random.best_params_
        self.best_score = random.best_score_
        self.cv_results = random.cv_results_
        
        print(f"   - Best Score: {self.best_score:.4f}")
        print(f"   - Best Parameters: {self.best_params}")
        
        return self
    
    def evaluate_best(self, X_test, y_test):
        """
        Evaluate the best model on test data.
        """
        if self.best_model is None:
            raise ValueError("No model tuned yet! Run grid_search() or random_search() first.")
        
        y_pred = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Best Model Test Accuracy: {accuracy:.4f}")
        
        return accuracy


def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Tune Random Forest hyperparameters.
    """
    print("\n" + "="*60)
    print("🌲 Tuning Random Forest")
    print("="*60)
    
    # Parameter grid
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    tuner = HyperparameterTuner(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    # Use RandomizedSearch for speed
    tuner.random_search(X_train, y_train)
    tuner.evaluate_best(X_test, y_test)
    
    # Save results
    results = {
        'model': 'Random Forest',
        'best_params': tuner.best_params,
        'best_score': tuner.best_score,
        'test_accuracy': tuner.evaluate_best(X_test, y_test)
    }
    
    return results


def tune_svm(X_train, y_train, X_test, y_test):
    """
    Tune SVM hyperparameters.
    """
    print("\n" + "="*60)
    print("⚡ Tuning SVM")
    print("="*60)
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }
    
    tuner = HyperparameterTuner(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    # Use RandomizedSearch for speed
    tuner.random_search(X_train, y_train)
    tuner.evaluate_best(X_test, y_test)
    
    results = {
        'model': 'SVM',
        'best_params': tuner.best_params,
        'best_score': tuner.best_score,
        'test_accuracy': tuner.evaluate_best(X_test, y_test)
    }
    
    return results


def tune_decision_tree(X_train, y_train, X_test, y_test):
    """
    Tune Decision Tree hyperparameters.
    """
    print("\n" + "="*60)
    print("🌳 Tuning Decision Tree")
    print("="*60)
    
    param_grid = {
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    tuner = HyperparameterTuner(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring='accuracy'
    )
    
    tuner.random_search(X_train, y_train)
    tuner.evaluate_best(X_test, y_test)
    
    results = {
        'model': 'Decision Tree',
        'best_params': tuner.best_params,
        'best_score': tuner.best_score,
        'test_accuracy': tuner.evaluate_best(X_test, y_test)
    }
    
    return results


def run_all_tuning(X_train, y_train, X_test, y_test):
    """
    Run hyperparameter tuning for all models.
    """
    print("\n" + "="*60)
    print("🧪 HYPERPARAMETER TUNING COMPARISON")
    print("="*60)
    
    results = []
    
    # Tune Random Forest
    rf_results = tune_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_results)
    
    # Tune SVM
    svm_results = tune_svm(X_train, y_train, X_test, y_test)
    results.append(svm_results)
    
    # Tune Decision Tree
    dt_results = tune_decision_tree(X_train, y_train, X_test, y_test)
    results.append(dt_results)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TUNING SUMMARY")
    print("="*60)
    
    for result in results:
        print(f"\n{result['model']}:")
        print(f"   - Best Score (CV): {result['best_score']:.4f}")
        print(f"   - Test Accuracy: {result['test_accuracy']:.4f}")
        print(f"   - Best Params: {result['best_params']}")
    
    # Save results
    with open('models/tuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\n✅ Results saved to models/tuning_results.json")
    
    return results


if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("Testing Hyperparameter Tuning...")
    
    # Load data
    X, y = load_wine(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Run tuning
    results = run_all_tuning(X_train_scaled, y_train, X_test_scaled, y_test)
    
    print("\n✅ Hyperparameter Tuning complete!")
"""
Random Forest Module
Implementation of Random Forest for classification tasks.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


class RandomForestModel:
    """
    Random Forest Classifier wrapper with training and evaluation methods.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            min_samples_split (int): Minimum samples required to split
            random_state (int): Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
        self.name = "Random Forest"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
        print(f"   - Trees: {self.model.n_estimators}")
        print(f"   - Features: {self.model.n_features_in_}")
        return self
    
    def predict(self, X_test):
        """Make predictions on test data."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.predict_proba(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        y_pred = self.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n📊 {self.name} Evaluation Results:")
        print(f"   - Accuracy: {accuracy:.4f}")
        print(f"   - Precision (weighted): {report['weighted avg']['precision']:.4f}")
        print(f"   - Recall (weighted): {report['weighted avg']['recall']:.4f}")
        print(f"   - F1-Score (weighted): {report['weighted avg']['f1-score']:.4f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred
        }
    
    def get_feature_importance(self, feature_names=None):
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        importances = self.model.feature_importances_
        
        if feature_names:
            importance_dict = dict(zip(feature_names, importances))
            importance_dict = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            return importance_dict
        return importances
    
    def get_params(self):
        """Get model parameters."""
        return self.model.get_params()


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import load_wine
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Testing Random Forest Model")
    print("="*60)
    
    # Load Wine dataset
    X, y = load_wine(return_X_y=True)
    print(f"Dataset shape: {X.shape}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    model = RandomForestModel(n_estimators=50)
    model.train(X_train_scaled, y_train)
    results = model.evaluate(X_test_scaled, y_test)
    
    print(f"\n✅ Random Forest test complete!")
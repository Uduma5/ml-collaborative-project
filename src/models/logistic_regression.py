"""
Logistic Regression Module
Implementation of Logistic Regression for classification tasks.
Dataset: Iris
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


class LogisticRegressionModel:
    """
    Logistic Regression wrapper with training and evaluation methods.
    """
    
    def __init__(self, max_iter=200, random_state=42):
        """
        Initialize Logistic Regression model.
        
        Args:
            max_iter (int): Maximum iterations (default: 200)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        self.name = "Logistic Regression"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the Logistic Regression model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
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
        """Evaluate model performance using classification metrics."""
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
    
    def get_coefficients(self, feature_names=None):
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        if feature_names:
            coef_dict = dict(zip(feature_names, self.model.coef_[0]))
            return coef_dict
        return self.model.coef_


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Testing Logistic Regression Model")
    print("="*60)
    
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {set(y)}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    model = LogisticRegressionModel()
    model.train(X_train_scaled, y_train)
    results = model.evaluate(X_test_scaled, y_test)
    
    print(f"\n✅ Logistic Regression test complete!")
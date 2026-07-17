"""
Support Vector Machine Module
Implementation of SVM for classification tasks.
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np


class SVMModel:
    """
    Support Vector Machine wrapper with training and evaluation methods.
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', random_state=42):
        """
        Initialize SVM model.
        
        Args:
            kernel (str): Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C (float): Regularization parameter (default: 1.0)
            gamma (str or float): Kernel coefficient (default: 'scale')
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
        self.name = f"SVM ({kernel} kernel)"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the SVM model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
        print(f"   - Support Vectors: {len(self.model.support_vectors_)}")
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
        print(f"   - Support Vectors: {len(self.model.support_vectors_)}")
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'support_vectors': len(self.model.support_vectors_)
        }
    
    def get_parameter_info(self):
        """Get model parameters information."""
        return {
            'kernel': self.model.kernel,
            'C': self.model.C,
            'gamma': self.model.gamma,
            'support_vectors': len(self.model.support_vectors_)
        }


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    
    print("Testing SVM Model...")
    
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = SVMModel(kernel='rbf', C=1.0)
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    
    print(f"\n✅ SVM test complete!")
"""
K-Nearest Neighbors Module
Implementation of KNN for classification tasks.
Dataset: Iris
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


class KNNModel:
    """
    K-Nearest Neighbors wrapper with training and evaluation methods.
    """
    
    def __init__(self, n_neighbors=5, weights='uniform', metric='euclidean'):
        """
        Initialize KNN model.
        
        Args:
            n_neighbors (int): Number of neighbors (default: 5)
            weights (str): Weight function ('uniform', 'distance')
            metric (str): Distance metric (default: 'euclidean')
        """
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric
        )
        self.name = "K-Nearest Neighbors"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the KNN model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
        print(f"   - k = {self.model.n_neighbors}")
        print(f"   - Weights: {self.model.weights}")
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
    
    def find_best_k(self, X_train, y_train, X_val, y_val, k_range=range(1, 21)):
        """
        Find optimal k value using validation set.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            k_range: Range of k values to test
            
        Returns:
            dict: Results for each k value
        """
        accuracies = []
        
        print(f"\n🔍 Finding optimal k (1 to {max(k_range)})...")
        
        for k in k_range:
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            accuracies.append(acc)
        
        best_k = k_range[np.argmax(accuracies)]
        best_accuracy = max(accuracies)
        
        print(f"   - Best k: {best_k}")
        print(f"   - Best Accuracy: {best_accuracy:.4f}")
        
        # Plot results
        plt.figure(figsize=(8, 5))
        plt.plot(k_range, accuracies, 'bo-')
        plt.xlabel('k Value')
        plt.ylabel('Validation Accuracy')
        plt.title('KNN: Accuracy vs k Value')
        plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best k={best_k}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        return {
            'k_values': list(k_range),
            'accuracies': accuracies,
            'best_k': best_k,
            'best_accuracy': best_accuracy
        }


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Testing KNN Model")
    print("="*60)
    
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    print(f"Dataset shape: {X.shape}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    model = KNNModel(n_neighbors=5)
    model.train(X_train_scaled, y_train)
    results = model.evaluate(X_test_scaled, y_test)
    
    # Find best k using cross-validation
    # Split training data for validation
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42
    )
    model.find_best_k(X_train_sub, y_train_sub, X_val, y_val)
    
    print(f"\n✅ KNN test complete!")
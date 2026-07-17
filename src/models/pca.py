"""
PCA Module
Implementation of Principal Component Analysis for dimensionality reduction.
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt


class PCAModel:
    """
    PCA wrapper with training and visualization methods.
    """
    
    def __init__(self, n_components=2):
        """
        Initialize PCA model.
        
        Args:
            n_components (int): Number of principal components (default: 2)
        """
        self.model = PCA(n_components=n_components)
        self.name = "PCA"
        self.is_trained = False
        self.n_components = n_components
    
    def train(self, X):
        """
        Train the PCA model.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            self: Trained model instance
        """
        # Standardize data first
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        print(f"✅ {self.name} trained successfully!")
        print(f"   - Original dimensions: {X.shape[1]}")
        print(f"   - Reduced dimensions: {self.n_components}")
        print(f"   - Explained variance: {self.model.explained_variance_ratio_.sum():.4f}")
        
        return self
    
    def transform(self, X):
        """
        Transform data to reduced dimensions.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            ndarray: Transformed data
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.transform(X_scaled)
    
    def fit_transform(self, X):
        """
        Fit and transform data in one step.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            ndarray: Transformed data
        """
        self.train(X)
        return self.transform(X)
    
    def evaluate(self, X):
        """
        Evaluate PCA performance.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            dict: Dictionary containing explained variance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        explained_variance = self.model.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        print(f"\n📊 {self.name} Evaluation Results:")
        print(f"   - Explained variance ratios: {explained_variance}")
        print(f"   - Cumulative variance: {cumulative_variance[-1]:.4f}")
        print(f"   - Total explained variance: {cumulative_variance[-1]:.4f}")
        
        return {
            'explained_variance_ratio': explained_variance,
            'cumulative_variance': cumulative_variance,
            'total_explained': cumulative_variance[-1],
            'n_components': self.n_components
        }
    
    def plot_explained_variance(self, max_components=20):
        """
        Plot explained variance for different numbers of components.
        
        Args:
            max_components (int): Maximum components to analyze
        """
        if not hasattr(self, 'scaler'):
            raise ValueError("Model not trained yet! Call train() first.")
        
        # Get the full PCA
        pca_full = PCA()
        pca_full.fit(self.scaler.transform(self.model.components_))
        
        explained_variance = pca_full.explained_variance_ratio_
        
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Individual explained variance
        plt.subplot(1, 2, 1)
        plt.bar(range(1, min(len(explained_variance), max_components) + 1),
                explained_variance[:max_components])
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Individual Explained Variance')
        plt.grid(True, alpha=0.3)
        
        # Subplot 2: Cumulative explained variance
        plt.subplot(1, 2, 2)
        cumulative_variance = np.cumsum(explained_variance[:max_components])
        plt.plot(range(1, min(len(explained_variance), max_components) + 1),
                cumulative_variance, 'bo-', linewidth=2)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return cumulative_variance
    
    def plot_2d_projection(self, X, y=None, title="PCA 2D Projection"):
        """
        Plot 2D projection of data.
        
        Args:
            X (ndarray): Feature matrix
            y (ndarray): Target labels (optional)
            title (str): Plot title
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        if self.n_components != 2:
            print(f"⚠️ Warning: n_components={self.n_components}, but plotting 2D.")
            print("   Retraining with 2 components for visualization...")
            self.n_components = 2
            self.train(X)
        
        X_reduced = self.transform(X)
        
        plt.figure(figsize=(10, 8))
        
        if y is not None:
            scatter = plt.scatter(
                X_reduced[:, 0], X_reduced[:, 1],
                c=y, cmap='viridis',
                alpha=0.7, s=50, edgecolors='black', linewidth=0.5
            )
            plt.colorbar(scatter, label='Target')
        else:
            plt.scatter(
                X_reduced[:, 0], X_reduced[:, 1],
                alpha=0.7, s=50, edgecolors='black', linewidth=0.5
            )
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return X_reduced


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    
    print("="*60)
    print("Testing PCA Model")
    print("="*60)
    
    # Load Iris dataset
    X, y = load_iris(return_X_y=True)
    print(f"Dataset shape: {X.shape}")
    
    # Train PCA
    pca = PCAModel(n_components=2)
    X_reduced = pca.fit_transform(X)
    print(f"Reduced shape: {X_reduced.shape}")
    
    # Evaluate
    results = pca.evaluate(X)
    
    # Plot explained variance
    pca.plot_explained_variance()
    
    # Plot 2D projection
    pca.plot_2d_projection(X, y=y, title="Iris Dataset - PCA 2D Projection")
    
    print(f"\n✅ PCA test complete!")
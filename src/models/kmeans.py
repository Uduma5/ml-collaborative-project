"""
K-Means Clustering Module
Implementation of K-Means for unsupervised learning.
"""

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt


class KMeansModel:
    """
    K-Means Clustering wrapper with training and evaluation methods.
    """
    
    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        """
        Initialize K-Means model.
        
        Args:
            n_clusters (int): Number of clusters (default: 3)
            max_iter (int): Maximum iterations (default: 300)
            random_state (int): Random seed for reproducibility (default: 42)
        """
        self.model = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
            n_init=10
        )
        self.name = "K-Means Clustering"
        self.is_trained = False
    
    def train(self, X):
        """Train the K-Means model on unlabeled data."""
        self.model.fit(X)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
        print(f"   - Clusters: {self.model.n_clusters}")
        print(f"   - Inertia: {self.model.inertia_:.4f}")
        return self
    
    def predict(self, X):
        """Predict cluster labels for data."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.predict(X)
    
    def evaluate(self, X):
        """Evaluate clustering performance."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        labels = self.predict(X)
        
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = -1
        
        print(f"\n📊 {self.name} Evaluation Results:")
        print(f"   - Number of clusters: {self.model.n_clusters}")
        print(f"   - Inertia: {self.model.inertia_:.4f}")
        print(f"   - Silhouette Score: {silhouette:.4f}")
        
        return {
            'labels': labels,
            'inertia': self.model.inertia_,
            'silhouette_score': silhouette,
            'cluster_centers': self.model.cluster_centers_,
            'n_clusters': self.model.n_clusters
        }
    
    def find_optimal_k(self, X, max_k=10):
        """Use elbow method to find optimal number of clusters."""
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        print(f"\n🔍 Finding optimal K (2 to {max_k})...")
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            
            labels = kmeans.labels_
            score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            silhouette_scores.append(score)
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"   - Optimal K: {optimal_k}")
        print(f"   - Best Silhouette Score: {max(silhouette_scores):.4f}")
        
        return {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
    
    def plot_clusters(self, X, title="Clustering Results"):
        """Visualize clustering results (works for 2D data)."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        if X.shape[1] > 2:
            print("⚠️ Warning: Can only plot 2D data. Using first two features.")
            X_plot = X[:, :2]
        else:
            X_plot = X
        
        labels = self.predict(X_plot)
        centers = self.model.cluster_centers_
        
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            X_plot[:, 0], X_plot[:, 1], 
            c=labels, cmap='viridis', 
            alpha=0.6, s=50
        )
        
        if centers.shape[1] > 2:
            centers_plot = centers[:, :2]
        else:
            centers_plot = centers
        
        plt.scatter(
            centers_plot[:, 0], centers_plot[:, 1],
            c='red', marker='X', s=200, 
            edgecolors='black', linewidths=2,
            label='Cluster Centers'
        )
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.colorbar(scatter, label='Cluster')
        plt.tight_layout()
        plt.show()


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    
    print("Testing K-Means Model...")
    
    X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
    
    model = KMeansModel(n_clusters=3)
    model.train(X)
    results = model.evaluate(X)
    model.find_optimal_k(X, max_k=6)
    
    print(f"\n✅ K-Means test complete!")
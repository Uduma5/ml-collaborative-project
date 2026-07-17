"""
Hierarchical Clustering Module
Implementation of Agglomerative Hierarchical Clustering for unsupervised learning.
"""

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist


class HierarchicalClusteringModel:
    """
    Hierarchical Clustering wrapper with training and evaluation methods.
    """
    
    def __init__(self, n_clusters=3, linkage='ward', metric='euclidean'):
        """
        Initialize Hierarchical Clustering model.
        
        Args:
            n_clusters (int): Number of clusters (default: 3)
            linkage (str): Linkage criterion ('ward', 'complete', 'average', 'single')
            metric (str): Distance metric (default: 'euclidean')
        """
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            metric=metric
        )
        self.name = "Hierarchical Clustering"
        self.is_trained = False
        self.linkage = linkage
    
    def train(self, X):
        """
        Train the Hierarchical Clustering model.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            self: Trained model instance
        """
        self.model.fit(X)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
        print(f"   - Clusters: {self.model.n_clusters}")
        print(f"   - Linkage: {self.linkage}")
        print(f"   - Samples: {X.shape[0]}")
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for data.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            ndarray: Cluster labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.fit_predict(X)
    
    def get_labels(self):
        """
        Get cluster labels after training.
        
        Returns:
            ndarray: Cluster labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.labels_
    
    def evaluate(self, X):
        """
        Evaluate clustering performance.
        
        Args:
            X (ndarray): Feature matrix
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        labels = self.get_labels()
        
        # Calculate silhouette score
        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(X, labels)
        else:
            silhouette = -1
        
        # Count samples per cluster
        unique, counts = np.unique(labels, return_counts=True)
        cluster_counts = dict(zip(unique, counts))
        
        print(f"\n📊 {self.name} Evaluation Results:")
        print(f"   - Number of clusters: {self.model.n_clusters}")
        print(f"   - Silhouette Score: {silhouette:.4f}")
        print(f"   - Cluster distribution: {cluster_counts}")
        
        return {
            'labels': labels,
            'n_clusters': self.model.n_clusters,
            'silhouette_score': silhouette,
            'cluster_counts': cluster_counts,
            'n_samples': len(labels)
        }
    
    def find_optimal_clusters(self, X, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score.
        
        Args:
            X (ndarray): Feature matrix
            max_clusters (int): Maximum number of clusters to test
            
        Returns:
            dict: Results for each cluster count
        """
        scores = []
        k_range = range(2, max_clusters + 1)
        
        print(f"\n🔍 Finding optimal clusters (2 to {max_clusters})...")
        
        for k in k_range:
            clustering = AgglomerativeClustering(
                n_clusters=k,
                linkage=self.linkage
            )
            labels = clustering.fit_predict(X)
            score = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else -1
            scores.append(score)
        
        optimal_k = k_range[np.argmax(scores)]
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, scores, 'bo-', linewidth=2)
        plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k={optimal_k}')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title(f'Hierarchical Clustering: Optimal k (Linkage={self.linkage})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"   - Optimal clusters: {optimal_k}")
        print(f"   - Best Silhouette Score: {max(scores):.4f}")
        
        return {
            'k_range': list(k_range),
            'scores': scores,
            'optimal_k': optimal_k,
            'best_score': max(scores)
        }
    
    def plot_dendrogram(self, X, title="Dendrogram"):
        """
        Plot dendrogram for hierarchical clustering.
        
        Args:
            X (ndarray): Feature matrix
            title (str): Plot title
        """
        print("\n📊 Generating dendrogram...")
        
        # Compute linkage matrix
        linkage_matrix = linkage(X, method=self.linkage)
        
        plt.figure(figsize=(12, 8))
        
        # Plot dendrogram
        dendrogram(
            linkage_matrix,
            leaf_rotation=45,
            leaf_font_size=8,
            color_threshold=0.7 * max(linkage_matrix[:, 2])
        )
        
        plt.title(title)
        plt.xlabel('Samples')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()
        
        return linkage_matrix
    
    def plot_clusters(self, X, title="Clustering Results"):
        """
        Visualize clustering results (works for 2D data).
        
        Args:
            X (ndarray): Feature matrix (2D)
            title (str): Plot title
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        if X.shape[1] > 2:
            print("⚠️ Warning: Can only plot 2D data. Using first two features.")
            X_plot = X[:, :2]
        else:
            X_plot = X
        
        labels = self.get_labels()
        
        plt.figure(figsize=(10, 8))
        
        scatter = plt.scatter(
            X_plot[:, 0], X_plot[:, 1],
            c=labels, cmap='viridis',
            alpha=0.7, s=50, edgecolors='black', linewidth=0.5
        )
        
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import make_blobs, load_iris
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Testing Hierarchical Clustering Model")
    print("="*60)
    
    # Generate synthetic data
    X, _ = make_blobs(n_samples=200, centers=3, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train and evaluate
    model = HierarchicalClusteringModel(n_clusters=3, linkage='ward')
    model.train(X_scaled)
    results = model.evaluate(X_scaled)
    
    # Find optimal clusters
    model.find_optimal_clusters(X_scaled, max_clusters=8)
    
    # Plot dendrogram
    model.plot_dendrogram(X_scaled)
    
    # Plot clusters
    model.plot_clusters(X_scaled)
    
    print(f"\n✅ Hierarchical Clustering test complete!")
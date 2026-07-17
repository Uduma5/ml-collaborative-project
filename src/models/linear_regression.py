"""
Linear Regression Module
Implementation of Linear Regression for regression tasks.
Dataset: California Housing
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


class LinearRegressionModel:
    """
    Linear Regression wrapper with training and evaluation methods.
    """
    
    def __init__(self):
        """Initialize Linear Regression model."""
        self.model = LinearRegression()
        self.name = "Linear Regression"
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the Linear Regression model."""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"✅ {self.name} trained successfully!")
        return self
    
    def predict(self, X_test):
        """Make predictions on test data."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance using regression metrics."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        y_pred = self.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n📊 {self.name} Evaluation Results:")
        print(f"   - MSE: {mse:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        print(f"   - R² Score: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
    
    def get_coefficients(self, feature_names=None):
        """Get model coefficients."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        
        if feature_names:
            coef_dict = dict(zip(feature_names, self.model.coef_))
            # Sort by absolute value (most important first)
            coef_dict = dict(sorted(
                coef_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            ))
            return coef_dict
        return self.model.coef_
    
    def get_intercept(self):
        """Get model intercept."""
        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")
        return self.model.intercept_


# Test the class
if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    print("="*60)
    print("Testing Linear Regression Model")
    print("="*60)
    
    # Load California Housing dataset
    X, y = fetch_california_housing(return_X_y=True)
    print(f"Dataset shape: {X.shape}")
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate
    model = LinearRegressionModel()
    model.train(X_train_scaled, y_train)
    results = model.evaluate(X_test_scaled, y_test)
    
    print(f"\n✅ Linear Regression test complete!")
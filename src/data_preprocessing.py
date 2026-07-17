"""
Data preprocessing module for loading and preparing ML datasets.
This module handles loading datasets and preparing them for training.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_iris_data():
    """
    Load and explore the Iris dataset.
    
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target labels
        feature_names (list): Names of features
        target_names (list): Names of target classes
    """
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"\n📊 Iris Dataset Loaded:")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Classes: {target_names}")
    print(f"   - Feature names: {feature_names}")
    
    return X, y, feature_names, target_names


def load_breast_cancer_data():
    """
    Load and explore the Breast Cancer dataset.
    
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target labels
        feature_names (list): Names of features
        target_names (list): Names of target classes
    """
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"\n📊 Breast Cancer Dataset Loaded:")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Classes: {target_names}")
    
    return X, y, feature_names, target_names


def load_wine_data():
    """
    Load and explore the Wine dataset.
    
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target labels
        feature_names (list): Names of features
        target_names (list): Names of target classes
    """
    data = load_wine()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"\n📊 Wine Dataset Loaded:")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Classes: {target_names}")
    
    return X, y, feature_names, target_names


def load_california_housing_data():
    """
    Load and explore the California Housing dataset.
    
    Returns:
        X (ndarray): Feature matrix
        y (ndarray): Target values
        feature_names (list): Names of features
    """
    data = fetch_california_housing()
    X, y = data.data, data.target
    feature_names = data.feature_names
    
    print(f"\n🏠 California Housing Dataset Loaded:")
    print(f"   - Samples: {X.shape[0]}")
    print(f"   - Features: {X.shape[1]}")
    print(f"   - Feature names: {feature_names}")
    
    return X, y, feature_names


def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and apply StandardScaler.
    
    Args:
        X (ndarray): Feature matrix
        y (ndarray): Target values
        test_size (float): Proportion for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📋 Data Split Complete:")
    print(f"   - Training samples: {X_train.shape[0]}")
    print(f"   - Test samples: {X_test.shape[0]}")
    print(f"   - Features scaled: ✅")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def explore_data(X, y, feature_names=None, target_names=None):
    """
    Display basic statistics about the dataset.
    
    Args:
        X (ndarray): Feature matrix
        y (ndarray): Target values
        feature_names (list): Names of features
        target_names (list): Names of target classes
    """
    print("\n" + "="*60)
    print("📊 DATA EXPLORATION")
    print("="*60)
    
    # Basic info
    print(f"\n📌 Dataset Shape: {X.shape}")
    print(f"📌 Target Shape: {y.shape}")
    
    # Summary statistics
    df = pd.DataFrame(X, columns=feature_names)
    print(f"\n📈 Summary Statistics:")
    print(df.describe())
    
    # Check for missing values
    missing = pd.DataFrame(X).isnull().sum().sum()
    print(f"\n🔍 Missing Values: {missing}")
    
    if target_names is not None:
        unique, counts = np.unique(y, return_counts=True)
        print(f"\n🎯 Class Distribution:")
        for i, (cls, count) in enumerate(zip(unique, counts)):
            print(f"   - {target_names[cls]}: {count} samples")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test the functions
    print("Testing data preprocessing module...")
    
    # Test Iris
    X, y, feature_names, target_names = load_iris_data()
    explore_data(X, y, feature_names, target_names)
    
    # Test split
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y)
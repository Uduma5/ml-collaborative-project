"""
Inference Script
Loads a trained model and makes predictions on new data.
"""

import pickle
import sys
import os
sys.path.append(os.path.abspath('.'))

import numpy as np
import pandas as pd
from src.data_preprocessing import (
    load_iris_data,
    load_breast_cancer_data,
    load_wine_data,
    load_california_housing_data
)


def load_model(model_name):
    """
    Load a trained model from the models folder.
    
    Args:
        model_name (str): Name of the model (e.g., 'random_forest')
        
    Returns:
        model: Loaded model
    """
    model_path = f'models/{model_name}_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("Please run python src/train.py first to train the models.")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"✅ Model loaded: {model_name}")
    return model


def get_sample_data(dataset_name, index=0):
    """
    Get a sample data point from a dataset.
    
    Args:
        dataset_name (str): Name of the dataset
        index (int): Index of the sample
        
    Returns:
        tuple: (sample, label, feature_names, label_name)
    """
    # Map display names to function names
    dataset_map = {
        'iris': load_iris_data,
        'breast_cancer': load_breast_cancer_data,
        'wine': load_wine_data,
        'california_housing': load_california_housing_data,
        'Iris': load_iris_data,
        'Breast Cancer': load_breast_cancer_data,
        'Wine': load_wine_data,
        'California Housing': load_california_housing_data
    }
    
    if dataset_name not in dataset_map:
        print(f"❌ Dataset not found: {dataset_name}")
        print(f"   Available: Iris, Breast Cancer, Wine, California Housing")
        return None, None, None, None
    
    # Load the dataset
    data = dataset_map[dataset_name]()
    
    if dataset_name in ['Iris', 'iris']:
        X, y, feature_names, target_names = data
        label_name = target_names[y[index]] if y[index] < len(target_names) else str(y[index])
    elif dataset_name in ['Breast Cancer', 'breast_cancer']:
        X, y, feature_names, target_names = data
        label_name = target_names[y[index]] if y[index] < len(target_names) else str(y[index])
    elif dataset_name in ['Wine', 'wine']:
        X, y, feature_names, target_names = data
        label_name = target_names[y[index]] if y[index] < len(target_names) else str(y[index])
    elif dataset_name in ['California Housing', 'california_housing']:
        X, y, feature_names = data
        label_name = str(y[index])
    else:
        return None, None, None, None
    
    sample = X[index].reshape(1, -1)
    label = y[index]
    
    return sample, label, feature_names, label_name


def predict(model_name, dataset_name, sample_index=0):
    """
    Make a prediction using a trained model.
    
    Args:
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        sample_index (int): Index of the sample
    """
    print("="*60)
    print(f"🔮 Making Prediction with {model_name}")
    print("="*60)
    
    # Load model
    model = load_model(model_name)
    if model is None:
        return
    
    # Get sample data
    X_sample, y_true, feature_names, label_name = get_sample_data(dataset_name, sample_index)
    if X_sample is None:
        return
    
    print(f"\n📊 Dataset: {dataset_name}")
    print(f"📌 Sample Index: {sample_index}")
    print(f"📌 True Label: {y_true} ({label_name if label_name else 'N/A'})")
    
    # Make prediction
    try:
        prediction = model.predict(X_sample)
        
        # Try to get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_sample)
            print(f"📊 Prediction Probability: {proba}")
        
        print(f"\n✅ Prediction: {prediction[0]}")
        
        # Check if prediction matches true label
        if prediction[0] == y_true:
            print("🎯 Prediction CORRECT! ✅")
        else:
            print("⚠️ Prediction INCORRECT! ❌")
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    print("\n" + "="*60)


def interactive_mode():
    """
    Interactive mode for making predictions.
    """
    print("\n" + "="*60)
    print("🤖 ML Model Inference Tool")
    print("="*60)
    
    # List available models
    models = {
        '1': ('linear_regression', 'California Housing', 'regression'),
        '2': ('logistic_regression', 'Iris', 'classification'),
        '3': ('knn', 'Iris', 'classification'),
        '4': ('decision_tree', 'Wine', 'classification'),
        '5': ('svm', 'Breast Cancer', 'classification'),
        '6': ('random_forest', 'Wine', 'classification')
    }
    
    print("\n📋 Available Models:")
    for key, (name, dataset, model_type) in models.items():
        print(f"   {key}. {name} ({model_type}) - Dataset: {dataset}")
    
    # Get user input
    choice = input("\nSelect a model (1-6): ")
    if choice not in models:
        print("❌ Invalid choice")
        return
    
    model_name, dataset_name, model_type = models[choice]
    
    # Get sample index
    try:
        index = int(input("Enter sample index (0-9): "))
    except ValueError:
        index = 0
    
    # Make prediction
    predict(model_name, dataset_name, index)


def predict_single(model_name, dataset_name, sample_index=0):
    """
    Single prediction function for programmatic use.
    
    Args:
        model_name (str): Name of the model
        dataset_name (str): Name of the dataset
        sample_index (int): Index of the sample
    """
    predict(model_name, dataset_name, sample_index)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔮 ML Model Inference Script")
    print("="*60)
    
    # Check if models exist
    if not os.path.exists('models'):
        print("❌ No models found. Please run python src/train.py first.")
        sys.exit(1)
    
    # Run in interactive mode
    interactive_mode()
"""
Evaluation Pipeline
Evaluates all trained models and generates comparison reports.
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


def load_results():
    """Load training results from JSON file."""
    with open('models/training_results.json', 'r') as f:
        return json.load(f)


def create_comparison_table(results):
    """
    Create a comparison table of all models.
    """
    print("\n" + "="*60)
    print("📊 MODEL COMPARISON TABLE")
    print("="*60)
    
    data = []
    
    for model_name, model_results in results.items():
        metrics = model_results['metrics']
        row = {
            'Model': model_name.replace('_', ' ').title(),
            'Dataset': model_results['dataset']
        }
        
        # Classification metrics
        if 'accuracy' in metrics:
            row['Accuracy'] = f"{metrics['accuracy']:.4f}"
        if 'precision' in metrics:
            row['Precision'] = f"{metrics['precision']:.4f}"
        if 'recall' in metrics:
            row['Recall'] = f"{metrics['recall']:.4f}"
        if 'f1_score' in metrics:
            row['F1-Score'] = f"{metrics['f1_score']:.4f}"
        
        # Regression metrics
        if 'rmse' in metrics:
            row['RMSE'] = f"{metrics['rmse']:.4f}"
        if 'r2' in metrics:
            row['R²'] = f"{metrics['r2']:.4f}"
        
        # Clustering metrics
        if 'silhouette_score' in metrics:
            row['Silhouette'] = f"{metrics['silhouette_score']:.4f}"
        
        data.append(row)
    
    df = pd.DataFrame(data)
    print("\n", df.to_string(index=False))
    
    return df


def create_evaluation_report(results):
    """
    Create detailed evaluation report.
    """
    print("\n" + "="*60)
    print("📝 GENERATING EVALUATION REPORT")
    print("="*60)
    
    report_lines = []
    report_lines.append("# Model Evaluation Report\n")
    report_lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    report_lines.append("## Summary\n")
    
    # Summary table
    df = create_comparison_table(results)
    report_lines.append("### Model Performance Comparison\n")
    report_lines.append(df.to_markdown(index=False))
    
    # Detailed results
    report_lines.append("\n## Detailed Results\n")
    
    for model_name, model_results in results.items():
        report_lines.append(f"\n### {model_name.replace('_', ' ').title()}")
        report_lines.append(f"- **Dataset:** {model_results['dataset']}")
        
        metrics = model_results['metrics']
        for metric, value in metrics.items():
            if metric not in ['predictions', 'confusion_matrix', 'classification_report']:
                if isinstance(value, float):
                    report_lines.append(f"- **{metric}:** {value:.4f}")
                else:
                    report_lines.append(f"- **{metric}:** {value}")
        
        # Feature importance
        if 'feature_importance' in model_results:
            report_lines.append("\n#### Feature Importance (Top 5):")
            importance = model_results['feature_importance']
            if isinstance(importance, dict):
                sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                for feature, imp in sorted_imp:
                    report_lines.append(f"- {feature}: {imp:.4f}")
    
    # Save report
    report_path = 'reports/evaluation_report.md'
    with open(report_path, 'w') as f:
        f.writelines(report_lines)
    print(f"\n✅ Evaluation report saved to {report_path}")
    
    return report_lines


def plot_model_comparison(results):
    """
    Create visualization comparing model performance.
    """
    print("\n" + "="*60)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*60)
    
    # Extract accuracies for classification models
    models = []
    accuracies = []
    datasets = []
    
    for model_name, model_results in results.items():
        if 'accuracy' in model_results['metrics']:
            models.append(model_name.replace('_', ' ').title())
            accuracies.append(model_results['metrics']['accuracy'])
            datasets.append(model_results['dataset'])
    
    if models:
        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(models, accuracies, color='steelblue', edgecolor='black')
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        plt.ylim(0, 1.1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('reports/model_comparison.png', dpi=300)
        print("✅ Model comparison plot saved to reports/model_comparison.png")
        plt.show()
    
    # Extract RMSE for regression models
    rmse_models = []
    rmse_values = []
    
    for model_name, model_results in results.items():
        if 'rmse' in model_results['metrics']:
            rmse_models.append(model_name.replace('_', ' ').title())
            rmse_values.append(model_results['metrics']['rmse'])
    
    if rmse_models:
        plt.figure(figsize=(10, 6))
        plt.bar(rmse_models, rmse_values, color='coral', edgecolor='black')
        plt.xlabel('Model')
        plt.ylabel('RMSE')
        plt.title('Regression Model RMSE Comparison')
        for bar, val in zip(plt.gca().patches, rmse_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{val:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('reports/rmse_comparison.png', dpi=300)
        print("✅ RMSE comparison plot saved to reports/rmse_comparison.png")
        plt.show()


if __name__ == "__main__":
    # Load results
    results = load_results()
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Generate report
    create_evaluation_report(results)
    
    # Create visualizations
    plot_model_comparison(results)
    
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE!")
    print("="*60)
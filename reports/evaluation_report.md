# Model Evaluation Report
*Generated: 2026-07-17 12:52:42*
## Summary
### Model Performance Comparison
| Model               | Dataset            |     RMSE |       R² |   Accuracy |   Silhouette |
|:--------------------|:-------------------|---------:|---------:|-----------:|-------------:|
| Linear Regression   | California Housing |   0.7456 |   0.5758 |   nan      |     nan      |
| Logistic Regression | Iris               | nan      | nan      |     1      |     nan      |
| Knn                 | Iris               | nan      | nan      |     1      |     nan      |
| Decision Tree       | Wine               | nan      | nan      |     0.9444 |     nan      |
| Svm                 | Breast Cancer      | nan      | nan      |     0.9825 |     nan      |
| Random Forest       | Wine               | nan      | nan      |     1      |     nan      |
| Kmeans              | Iris               | nan      | nan      |   nan      |       0.5528 |
## Detailed Results

### Linear Regression- **Dataset:** California Housing- **mse:** 0.5559- **rmse:** 0.7456- **r2:** 0.5758
### Logistic Regression- **Dataset:** Iris- **accuracy:** 1.0000
### Knn- **Dataset:** Iris- **accuracy:** 1.0000
### Decision Tree- **Dataset:** Wine- **accuracy:** 0.9444
#### Feature Importance (Top 5):- flavanoids: 0.4111- color_intensity: 0.3849- proline: 0.1641- ash: 0.0209- alcohol: 0.0190
### Svm- **Dataset:** Breast Cancer- **accuracy:** 0.9825- **support_vectors:** 104
### Random Forest- **Dataset:** Wine- **accuracy:** 1.0000
#### Feature Importance (Top 5):- flavanoids: 0.2023- color_intensity: 0.1712- proline: 0.1390- alcohol: 0.1124- od280/od315_of_diluted_wines: 0.1116
### Kmeans- **Dataset:** Iris- **labels:** [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 2, 0, 2, 2, 0]- **inertia:** 78.8514- **silhouette_score:** 0.5528- **cluster_centers:** [[5.901612903225806, 2.7483870967741937, 4.393548387096774, 1.4338709677419355], [5.006, 3.428, 1.4620000000000002, 0.24600000000000055], [6.85, 3.0736842105263156, 5.742105263157894, 2.0710526315789473]]- **n_clusters:** 3
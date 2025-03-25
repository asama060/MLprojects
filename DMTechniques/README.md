
# Breast Cancer Clustering and Anomaly Detection Analysis

This project explores various clustering and anomaly detection techniques applied to the Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository. It aims to uncover hidden patterns, identify potential anomalies, and improve the overall understanding of the dataset.

## Project Structure

1. **Data Loading and Preprocessing:** The project begins by loading the dataset and preprocessing it for subsequent analyses, including feature scaling.
2. **Density-based Clustering (DBSCAN):** A custom DBSCAN implementation is used to cluster the dataset based on neighborhood density. Performance analysis explores how different `eps` and `min_pts` parameters influence the number of clusters formed.
3. **Graph-based Clustering (Spectral Clustering):** A custom Spectral Clustering implementation performs clustering by constructing a similarity graph and utilizing its Laplacian matrix. Performance evaluation involves analyzing silhouette scores for different `k` and `sigma` values.
4. **Anomaly Detection with Isolation Forest:** An Isolation Forest implementation identifies anomalies within the dataset. Subsequent K-means++ clustering is performed after removing anomalies to analyze the impact on performance.
5. **Visualization and Analysis:** The results of the clustering and anomaly detection algorithms are presented visually and analyzed in terms of cluster sizes, anomaly score distributions, and Silhouette scores.

## Key Objectives

- **Clustering:** Identify meaningful clusters within the Breast Cancer dataset using DBSCAN and Spectral Clustering.
- **Anomaly Detection:** Detect potential anomalies or outliers in the dataset using Isolation Forest.
- **Performance Evaluation:** Analyze the performance of the clustering methods using metrics like Silhouette score.
- **Impact of Anomaly Removal:** Investigate how removing anomalies influences the effectiveness of K-means++ clustering.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colaboratory

## Getting Started

1. Make sure you have Python 3 and required libraries installed.
2. Run the notebook in Google Colaboratory. 

## Usage

The notebook provides a comprehensive analysis of the Breast Cancer dataset using different clustering and anomaly detection methods. The project explores the effectiveness of each technique, investigates the impact of removing anomalies, and showcases the results through visualizations and detailed performance evaluations.

## Note

Feel free to experiment with the code, adjust parameters, and modify the analysis to explore different aspects of the dataset. 

## Future Improvements

- Exploration of alternative clustering algorithms.
- Development of more advanced anomaly detection techniques.
- Incorporation of feature engineering and selection methods.

**Happy coding!** 

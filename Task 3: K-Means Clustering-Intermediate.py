#1st Method:-


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs  # For synthetic clustering data
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA  # For 2D visualization
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

try:
    # Step 1: Load the dataset
    # Using synthetic data (make_blobs) to simulate customer segmentation (e.g., age and spending score).
    # Generates 300 samples with 4 features, 4 natural clusters (but we'll determine K via elbow).
    # In a real scenario, replace with pd.read_csv('your_file.csv').
    X, _ = make_blobs(n_samples=300, centers=4, n_features=4, random_state=42, cluster_std=1.0)
    data = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
    print("Synthetic dataset loaded (simulating customer data).")
    print("First 5 rows:")
    print(data.head())
    print(f"Shape: {data.shape}")
    print(f"Data types:\n{data.dtypes}")
    print(f"Missing values:\n{data.isnull().sum()}")

    # Step 2: Preprocess the dataset (scaling)
    # Standardize features (mean=0, std=1) for K-Means, as it's distance-based.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    X_scaled = pd.DataFrame(X_scaled, columns=data.columns)

    print("\nAfter preprocessing (standardization):")
    print(X_scaled.head())

    # Step 3: Determine the optimal number of clusters using the elbow method
    # Compute inertia (within-cluster sum of squares) for K=1 to 10.
    inertia = []
    k_range = range(1, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # n_init for stability
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    # Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True)
    plt.show()

    # From the plot, choose K where inertia starts to decrease slowly (elbow point). Here, K=4 is optimal based on data generation.
    optimal_k = 4  # Adjust based on your elbow plot interpretation
    print(f"\nOptimal K determined via elbow method: {optimal_k}")

    # Optional: Silhouette score for validation (higher is better, closer to 1)
    silhouette_avg = silhouette_score(X_scaled, KMeans(n_clusters=optimal_k, random_state=42, n_init=10).fit_predict(X_scaled))
    print(f"Silhouette Score for K={optimal_k}: {silhouette_avg:.2f} (closer to 1 is better)")

    # Step 4: Apply K-Means clustering with optimal K
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    data['Cluster'] = clusters  # Add cluster labels to original data
    print(f"\nK-Means applied. Cluster centers:\n{kmeans.cluster_centers_}")

    # Step 5: Visualize clusters using 2D scatter plots
    # Use PCA to reduce to 2D for visualization (since data has 4 features).
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    data['PCA1'] = X_pca[:, 0]
    data['PCA2'] = X_pca[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=data, palette='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')  # Plot centroids in PCA space
    plt.title('K-Means Clusters (2D PCA Projection)')
    plt.legend()
    plt.show()

    # Step 6: Interpret the clustering results
    # Analyze cluster sizes and feature means.
    cluster_summary = data.groupby('Cluster').mean()
    print("\nCluster Interpretation (mean values per cluster):")
    print(cluster_summary)
    print("\nCluster Sizes:")
    print(data['Cluster'].value_counts().sort_index())

    # Example Interpretation (based on synthetic data):
    # - Cluster 0: High Feature1, low Feature2 → e.g., "High-spending young customers".
    # - Cluster 1: Low Feature1, high Feature3 → e.g., "Low-spending older customers".
    # - Adjust based on your features (e.g., for real customer data: age, income, spending).
    # Insights: Clusters help segment customers for targeted marketing (e.g., personalized offers).

    print("\nK-Means clustering complete! Use results for segmentation or further analysis.")

except Exception as e:
    print(f"An error occurred: {e}. Check your data or library versions (e.g., sklearn >= 1.0).")









#2nd Method:-



#Step 1: Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



#Step 2: Simulate a Dataset

# Simulated customer data
data = {
    'Annual Income (k$)': [15, 16, 17, 18, 19, 20, 70, 72, 74, 76, 78, 80],
    'Spending Score': [39, 41, 43, 45, 47, 49, 60, 62, 64, 66, 68, 70]
}
df = pd.DataFrame(data)
print("📄 Raw Dataset:\n", df)





#Step 3: Preprocess (Scaling)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)




#Step 4: Elbow Method to Find Optimal K

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.tight_layout()
plt.show()



#Step 5: Apply K-Means Clustering

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
print("\n📊 Clustered Data:\n", df)



#Step 6: Visualize Clusters

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Annual Income (k$)',
    y='Spending Score',
    hue='Cluster',
    data=df,
    palette='Set1',
    s=100
)
plt.title('Customer Segments by K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()




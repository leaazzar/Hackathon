import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the preprocessed data
data_preprocessed = pd.read_csv("preprocessed_data_with_smoking.csv")

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters based on your analysis
clusters = kmeans.fit_predict(data_preprocessed)

# Add cluster labels to the dataset
data_preprocessed['Cluster'] = clusters

# Visualize the Clusters using PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_result = pca.fit_transform(data_preprocessed.drop(columns=["Cluster"]))
data_preprocessed['PCA1'] = pca_result[:, 0]
data_preprocessed['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(data_preprocessed['PCA1'], data_preprocessed['PCA2'], c=data_preprocessed['Cluster'], cmap='viridis')
plt.title("Clusters of Smokers Based on Lifestyle, Personality, and Smoking Factors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Save the clustered data
data_preprocessed.to_csv("2024_PersonalityTraits_SurveyData_with_Smoking_Clusters.csv", index=False)
print("Clustering completed and results saved.")

# Analyze clusters
cluster_analysis = data_preprocessed.groupby("Cluster").mean()

# Save cluster analysis
cluster_analysis.to_csv("Cluster_Analysis_with_Smoking.csv")

# Display cluster analysis
print("Cluster Analysis Summary:")
print(cluster_analysis)

# Calculate mean values for each cluster
cluster_profiles = data_preprocessed.groupby("Cluster").mean()
print("Cluster Profiles:\n", cluster_profiles)

# Save the cluster profiles to a CSV file for further analysis
cluster_profiles.to_csv("Cluster_Profiles.csv")

# Calculate feature-level profiles for all clusters
feature_analysis = data_preprocessed.groupby("Cluster").mean()

# Display feature-specific profiles
print("Feature-Specific Analysis by Cluster:")
print(feature_analysis)

# Save the analysis for external review
feature_analysis.to_csv("Feature_Specific_Analysis.csv")

import seaborn as sns
import matplotlib.pyplot as plt

# Transpose the data for visualization
cluster_profile_transposed = feature_analysis.T

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_profile_transposed, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Comparison Across Clusters")
plt.show()

'''
Cluster 0: Light Smokers with Balanced Lifestyles
Cluster 1: Heavy and Stress-Driven Smokers
Cluster 2: Habitual Smokers with Strong Smoking Patterns

'''


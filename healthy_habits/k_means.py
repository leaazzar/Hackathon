import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load the preprocessed data
data_preprocessed = pd.read_csv(r"C:\Users\Lenovo\Desktop\Hackathon\healthy_habits\preprocessed_data.csv")

# Step 1: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust n_clusters based on your needs
clusters = kmeans.fit_predict(data_preprocessed)

# Add cluster labels to the dataset
data_preprocessed['Cluster'] = clusters

# Step 2: Visualize the Clusters (using PCA for dimensionality reduction)
pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
pca_result = pca.fit_transform(data_preprocessed.drop(columns=["Cluster"]))
data_preprocessed['PCA1'] = pca_result[:, 0]
data_preprocessed['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(8, 6))
plt.scatter(data_preprocessed['PCA1'], data_preprocessed['PCA2'], c=data_preprocessed['Cluster'], cmap='viridis')
plt.title("Clusters of Smokers Based on Lifestyle and Personality")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Save the clustered data for analysis
data_preprocessed.to_csv(r"C:\\Desktop\\2024_PersonalityTraits_SurveyData_clustered.csv", index=False)

print("Clustering completed and results saved.")

# Load the clustered data
data_clustered = pd.read_csv(r"C:\\Desktop\\2024_PersonalityTraits_SurveyData_clustered.csv")

# Compute cluster-wise averages for features
cluster_analysis = data_clustered.groupby("Cluster").mean()

# Save the analysis for review
cluster_analysis.to_csv(r"C:\\Desktop\\Cluster_Analysis.csv")

# Display cluster analysis results
print("Cluster Analysis Summary:")
print(cluster_analysis)

# Define recommendations for each cluster
recommendations = {
    0: "Maintain your current lifestyle with balanced activities.",
    1: "Consider stress management activities such as yoga or mindfulness.",
    2: "Great job staying active! Encourage others in your group to adopt similar habits."
}

# Add recommendations to each smoker in the dataset
data_clustered['Recommendations'] = data_clustered['Cluster'].map(recommendations)

# Save the dataset with recommendations
data_clustered.to_csv("C:\\Desktop\\2024_PersonalityTraits_SurveyData_with_Recommendations.csv", index=False)

print("Recommendations added and saved to the dataset.")


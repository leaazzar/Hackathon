import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import seaborn as sns

# Load the preprocessed data
data_preprocessed = pd.read_csv("preprocessed_data_with_smoking.csv")

# Apply K-Means Clustering
n_clusters = 3  # You can adjust this based on your analysis
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(data_preprocessed)

# Add cluster labels to the dataset
data_preprocessed['Cluster'] = clusters

# Visualize the Clusters using PCA (Dimensionality Reduction to 2D)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(data_preprocessed.drop(columns=["Cluster"]))
data_preprocessed['PCA1'] = pca_result[:, 0]
data_preprocessed['PCA2'] = pca_result[:, 1]

# Plot the clusters in PCA space
plt.figure(figsize=(8, 6))
plt.scatter(data_preprocessed['PCA1'], data_preprocessed['PCA2'], c=data_preprocessed['Cluster'], cmap='viridis')
plt.title("Clusters of Smokers Based on Lifestyle, Personality, and Smoking Factors")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.colorbar(label="Cluster")
plt.show()

# Save the clustered data to CSV for later analysis
data_preprocessed.to_csv("2024_PersonalityTraits_SurveyData_with_Smoking_Clusters.csv", index=False)
print("Clustering completed and results saved.")

# Analyze clusters: Calculate the mean values of features for each cluster
cluster_analysis = data_preprocessed.groupby("Cluster").mean()

# Save cluster analysis for external review
cluster_analysis.to_csv("Cluster_Analysis_with_Smoking.csv")

# Display cluster analysis summary
print("Cluster Analysis Summary:")
print(cluster_analysis)

# Calculate the cluster profiles (mean values for each feature per cluster)
cluster_profiles = data_preprocessed.groupby("Cluster").mean()
print("Cluster Profiles:\n", cluster_profiles)

# Save the cluster profiles to a CSV file for further analysis
cluster_profiles.to_csv("Cluster_Profiles.csv")

# Visualize the feature comparison across clusters using a heatmap
cluster_profile_transposed = cluster_profiles.T
plt.figure(figsize=(10, 8))
sns.heatmap(cluster_profile_transposed, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Comparison Across Clusters")
plt.show()

# Cluster labeling based on analysis (adjust based on your findings)
cluster_labels = {
    0: "Light Smokers with Balanced Lifestyles",
    1: "Heavy and Stress-Driven Smokers",
    2: "Habitual Smokers with Strong Smoking Patterns"
}

# Map the cluster labels to the dataset
data_preprocessed['Cluster_Label'] = data_preprocessed['Cluster'].map(cluster_labels)

# Save the labeled dataset
data_preprocessed.to_csv("Labeled_Cluster_Data.csv", index=False)

# Building a classifier to predict cluster labels
X = data_preprocessed.drop(columns=["Cluster", "Cluster_Label", "PCA1", "PCA2"])
y = data_preprocessed['Cluster']  # Target is the cluster label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rf_clf = RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_clf.predict(X_test)

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the classifier model if needed
import joblib
joblib.dump(rf_clf, "random_forest_classifier_model.pkl")

# Predict the cluster for new data (example)
# new_data = [[...] (new data for prediction)]
# cluster_prediction = rf_clf.predict(new_data)
# print(f"The predicted cluster for the new data is: {cluster_labels[cluster_prediction[0]]}")

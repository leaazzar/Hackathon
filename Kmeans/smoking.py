import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

# Load the dataset
file_path = r"C:\Users\Lenovo\Downloads\2024_PersonalityTraits_SurveyData (1).xls"
df_smoking = pd.read_excel(file_path)

# Define the updated feature set for smoking-related data
updated_smoking_features = [
    "Have you smoked at least one full tobacco cigarette (excluding e-cigarettes) once or more in the past 30 days?",
    "Do you find it difficult to refrain from smoking where it is forbidden (church, library, cinema, plane, etc...)?",
    "How many cigarettes do you smoke each day?",
    "How soon after you wake up do you smoke your first cigarette?",
    "How old were you the first time you smoked a full cigarette (not just a few puffs)?"
]

# Select the relevant columns from the dataset
try:
    selected_smoking_data = df_smoking[updated_smoking_features]
    
    # Handle missing values
    imputer = SimpleImputer(strategy="most_frequent")
    smoking_data_cleaned = pd.DataFrame(
        imputer.fit_transform(selected_smoking_data),
        columns=selected_smoking_data.columns
    )
    
    # Encode categorical variables into numerical values
    encoder = LabelEncoder()
    for col in smoking_data_cleaned.columns:
        smoking_data_cleaned[col] = encoder.fit_transform(smoking_data_cleaned[col])
    
    # Normalize the data for better clustering
    scaler = StandardScaler()
    smoking_data_scaled = scaler.fit_transform(smoking_data_cleaned)
    
    # Save preprocessed data
    pd.DataFrame(smoking_data_scaled, columns=smoking_data_cleaned.columns).to_csv(
        "Updated_Smoking_Specific_Data_Preprocessed2.csv", index=False
    )
    print("Preprocessing completed for updated smoking-specific data.")
    
    # Analyze correlations
    corr_matrix = pd.DataFrame(smoking_data_scaled, columns=smoking_data_cleaned.columns).corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.show()
    
    # Perform clustering
    # Determine the optimal number of clusters using the Elbow Method
    inertia = []
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(smoking_data_scaled)
        inertia.append(kmeans.inertia_)
    
    # Plot the Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 10), inertia, marker='o')
    plt.title("Elbow Method for Optimal Clusters")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()
    
    # Perform clustering with the chosen number of clusters
    optimal_clusters = 4  # Updated to 4 based on the Elbow Method plot
    kmeans_smoking = KMeans(n_clusters=optimal_clusters, random_state=42)
    clusters_smoking = kmeans_smoking.fit_predict(smoking_data_scaled)
    
    # Add cluster labels to the data
    smoking_data_cleaned["Cluster"] = clusters_smoking
    
    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(smoking_data_scaled, clusters_smoking)
    print(f"Silhouette Score (4 Clusters): {silhouette_avg}")
    
    # Visualize clusters using PCA
    pca_smoking = PCA(n_components=2)
    pca_result_smoking = pca_smoking.fit_transform(smoking_data_scaled)
    smoking_data_cleaned["PCA1"] = pca_result_smoking[:, 0]
    smoking_data_cleaned["PCA2"] = pca_result_smoking[:, 1]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(
        smoking_data_cleaned["PCA1"],
        smoking_data_cleaned["PCA2"],
        c=smoking_data_cleaned["Cluster"],
        cmap="viridis"
    )
    plt.title("Clusters Based on Smoking-Specific Data (4 Clusters)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(label="Cluster")
    plt.show()
    
    # Save the clustered data
    smoking_data_cleaned.to_csv("Updated_Smoking_Specific_Data_Clustered2.csv", index=False)
    print("Clustering completed and results saved.")
    
except KeyError as e:
    print(f"Error: {e}. Please ensure all column names exist in the dataset.")

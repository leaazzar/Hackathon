import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the feature importances and encoded dataset
feature_importances_path = "/content/Hackathon/feature_importances.csv"  # Path to the saved CSV
feature_importances = pd.read_csv(feature_importances_path)

encoded_dataset_path = '/content/Hackathon/Encoded_Dataset.csv'  # Path to dataset
encoded_data = pd.read_csv(encoded_dataset_path)

# Filter features by importance threshold
threshold = 0.02  # Adjust threshold as needed
important_features = feature_importances[feature_importances["Importance"] > threshold]["Feature"]

# Subset the dataset to include only the important features
subset_data = encoded_data[important_features]

# Compute the correlation matrix
correlation_matrix = subset_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Heatmap of Important Features", fontsize=16)
plt.show()

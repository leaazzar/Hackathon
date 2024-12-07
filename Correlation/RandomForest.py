import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load the encoded dataset
encoded_dataset_path = 'Encoded_Dataset.csv'  # Replace with your file path
encoded_data = pd.read_csv(encoded_dataset_path)

# Split features and target variable
X = encoded_data.drop(columns=['dependency_label'])
y = encoded_data['dependency_label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Get feature importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Display results
print(f"Model Accuracy: {accuracy}")
print("\nClassification Report:\n", report)
print("\nFeature Importances:\n", feature_importances)

# Optional: Save feature importances to a CSV file
feature_importances.to_csv("feature_importances.csv", index=False)

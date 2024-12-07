import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained classifier model
rf_clf = joblib.load("random_forest_classifier_model.pkl")

# Define the exact features used during training (now in the correct order)
model_features = [
    # Personality and lifestyle factors
    "I see myself as someone who is extraverted, enthusiastic:",
    "I see myself as someone who is critical, quarrelsome:",
    "I see myself as someone who is dependable, self-disciplined:",
    "I see myself as someone who is anxious, easily upset:",
    "I see myself as someone who is open to new experiences:",
    "How often do you exercise?",
    "How often do you feel stressed?",
    # Smoking-related factors
    "Have you smoked at least one full tobacco cigarette (excluding e-cigarettes) once or more in the past 30 days?",
    "Do you find it difficult to refrain from smoking where it is forbidden (church, library, cinema, plane, etc...)?",
    "How many cigarettes do you smoke each day?",
    "Do you smoke if you are so ill that you are in bed most of the day?",
    "How soon after you wake up do you smoke your first cigarette?",
    "How old were you the first time you smoked a full cigarette (not just a few puffs)?"
]

# Recommendation mapping for clusters
recommendations = {
    0: "You are a light smoker with a balanced lifestyle. We recommend staying active, managing stress, and gradually quitting smoking. Try using fitness apps or mindfulness tools for better health.",
    1: "You are a heavy smoker experiencing stress. We suggest a combination of smoking cessation programs and stress-relief techniques. Consider mindfulness exercises, yoga, or relaxation methods to reduce stress.",
    2: "You are a habitual smoker with strong smoking patterns. We recommend looking into long-term smoking cessation programs. Nicotine replacement therapy (NRT), behavioral therapy, and support groups could help in your journey."
}

# Function to recommend based on cluster prediction
def recommend_cluster(user_data):
    # Create a DataFrame with the input data, ensuring it has the same columns as the training data
    user_df = pd.DataFrame([user_data])

    # Add missing columns with default values (e.g., 0 or "None")
    missing_cols = set(model_features) - set(user_df.columns)
    for col in missing_cols:
        user_df[col] = 0  # Or use a neutral value, like "None" for categorical columns

    # Ensure the column order matches the training data
    user_df = user_df[model_features]

    # Predict the cluster for the user
    cluster = rf_clf.predict(user_df)[0]
    
    # Return the recommendation for the predicted cluster
    return recommendations[cluster]

# Example of user data (ensure this matches the feature set used during training)
new_user_data = {
    "I see myself as someone who is extraverted, enthusiastic:": 2,
    "I see myself as someone who is critical, quarrelsome:": 6,
    "I see myself as someone who is dependable, self-disciplined:": 4,
    "I see myself as someone who is anxious, easily upset:": 9,
    "I see myself as someone who is open to new experiences:": 5,
    "How often do you exercise?": 2,  # Number of times per week
    "How often do you feel stressed?": 7,  # Scale from 1 to 10
    "Have you smoked at least one full tobacco cigarette (excluding e-cigarettes) once or more in the past 30 days?": 1,
    "Do you find it difficult to refrain from smoking where it is forbidden (church, library, cinema, plane, etc...)?": 1,
    "How many cigarettes do you smoke each day?": 30,
    "Do you smoke if you are so ill that you are in bed most of the day?": 1,
    "How soon after you wake up do you smoke your first cigarette?": 10,  # Minutes after waking up
    "How old were you the first time you smoked a full cigarette (not just a few puffs)?": 16
}

# Get recommendations for the new user
user_recommendation = recommend_cluster(new_user_data)

# Display the recommendation
print("Recommendation for the user:")
print(user_recommendation)

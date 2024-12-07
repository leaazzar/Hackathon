import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the data
excel_file_path = r"C:\Desktop\2024_PersonalityTraits_SurveyData.xls"
csv_file_path = r"C:\Desktop\2024_PersonalityTraits_SurveyData.csv"

# Load the Excel file
data = pd.read_excel(excel_file_path)

# Save the file as CSV
data.to_csv(csv_file_path, index=False)

print(f"Excel file successfully converted to CSV: {csv_file_path}")

# Select relevant features, including smoking-related factors
features = [
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

# Select only the specified features
data = data[features]

# Handle missing data
imputer = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
encoder = LabelEncoder()
for col in data_imputed.columns:
    data_imputed[col] = encoder.fit_transform(data_imputed[col])

# Normalize numerical data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Convert back to a DataFrame for readability
data_preprocessed = pd.DataFrame(data_scaled, columns=data_imputed.columns)

# Save the preprocessed data for further steps
data_preprocessed.to_csv("preprocessed_data_with_smoking.csv", index=False)

print("Preprocessed data saved for further analysis.")

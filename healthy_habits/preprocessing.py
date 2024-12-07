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

# Select relevant features
# Stress, exercise, and personality traits are assumed as key features.
# Adjust based on your data exploration.
features = [
    "I see myself as someone who is extraverted, enthusiastic:",
    "I see myself as someone who is critical, quarrelsome:",
    "I see myself as someone who is dependable, self-disciplined:",
    "I see myself as someone who is anxious, easily upset:",
    "I see myself as someone who is open to new experiences:",
    "How often do you exercise?",
    "How often do you feel stressed?",
]

data = data[features]

# Handle missing data
# Replace missing values with the most frequent value for categorical variables
imputer = SimpleImputer(strategy="most_frequent")
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical variables
# Convert Agree/Disagree and frequency-based features into numerical values
encoder = LabelEncoder()
for col in data_imputed.columns:
    data_imputed[col] = encoder.fit_transform(data_imputed[col])

# Normalize numerical data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Convert back to a DataFrame for readability
data_preprocessed = pd.DataFrame(data_scaled, columns=data_imputed.columns)

# Save the preprocessed data for further steps
data_preprocessed.to_csv("preprocessed_data.csv", index=False)

# Display the preprocessed data (first few rows)
data_preprocessed.head()

import pandas as pd
import numpy as np

# Load dataset
data = pd.read_excel('/content/hackathon/2024_PersonalityTraits_SurveyData.xls', engine='openpyxl')

# Standardize column names
data.columns = data.columns.str.strip()

# Define the actual column names
cigarettes_col = 'How many cigarettes do you smoke each day?'
time_col = 'How soon after you wake up do you smoke your first cigarette?'

# Normalize and clean string columns
data[cigarettes_col] = data[cigarettes_col].str.strip().str.lower()
data[time_col] = data[time_col].str.strip().str.lower()

# Map categories to numerical values for 'cigarettes_per_day_num'
cigarette_mapping = {
    '10 or less cigarettes/day': 1,
    '11 to 20 cigarettes': 2,
    '21 to 30 cigarettes': 3,
    '31 cigarettes/day or more': 4
}

# Map categories to numerical values for 'time_to_first_cigarette_num'
time_mapping = {
    'within 5 minutes': 3,
    '6 to 30 minutes': 2,
    '31 to 60 minutes': 1,
    'after 60 minutes': 0
}

# Apply mappings
data['cigarettes_per_day_num'] = data[cigarettes_col].map(cigarette_mapping)
data['time_to_first_cigarette_num'] = data[time_col].map(time_mapping)

# Handle unmapped or missing values by filling with 0 or a default value
data['cigarettes_per_day_num'] = data['cigarettes_per_day_num'].fillna(0)
data['time_to_first_cigarette_num'] = data['time_to_first_cigarette_num'].fillna(0)

# Define thresholds for dependency levels
def assign_dependency(row):
    if (row['cigarettes_per_day_num'] >= 3 and row['time_to_first_cigarette_num'] >= 2):
        return '2'  # High dependency
    elif (row['cigarettes_per_day_num'] >= 2 or row['time_to_first_cigarette_num'] >= 1):
        return '1'  # Medium dependency
    else:
        return '0'  # Low dependency

# Apply labeling
data['dependency_label'] = data.apply(assign_dependency, axis=1)

# Drop unnecessary columns
columns_to_disregard = [
    "Last page",
    "Unnamed: 0",
    "Have you smoked at least one full tobacco cigarette (excluding e-cigarettes) once or more in the past 30 days?",
    "What is your favorite or preferred cigarette brand(s) if you were able to access it?",
    "What cigarette brand(s) are you currently using?",
    "Has 2019's revolution or economic crisis caused you to switch away from your favorite or preferred cigarette brand(s) to an  alternative?",
    "What is your current employment status? [Comment]",
    "What is your current marital status? [Comment]",
    "What is your main source of income? [Comment]",
    "What type of income or financial support does your household receive? [Comment]",
    "If you receive payment in Lebanese Lira, what is your current estimated monthly household income? (If income is in US Dollars, then refer to the current black market exchange).",
]

data = data.drop(columns=columns_to_disregard, errors='ignore')

# Save the labeled dataset as a new CSV
output_file = '/content/hackathon/labeled_data.csv'
data.to_csv(output_file, index=False)

print(f"Labeled data saved to {output_file}")

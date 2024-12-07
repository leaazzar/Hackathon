import pandas as pd
import numpy as np

# File path to the CSV file
file_path = "C:/Users/User/OneDrive - American University of Beirut/Desktop/E3/EECE 490/hackathon/2024_PersonalityTraits_SurveyData.csv"

# Load the CSV file
df = pd.read_csv(file_path)

# Drop irrelevant columns
columns_to_disregard = [
    "Last page",
    "Unnamed: 0",
    "Have you smoked at least one full tobacco cigarette (excluding e-cigarettes) once or more in the past 30 days?",
    "What is your favorite or preferred cigarette brand(s) if you were able to access it?",
    "What cigarette brand(s) are you currently using?",
    "Has 2019's revolution or economic crisis caused you to switch away from your favorite or preferred cigarette brand(s) to anÂ  alternative?",
    "What is your current employment status? [Comment]",
    "What is your current marital status? [Comment]",
    "What is your main source of income? [Comment]",
    "What type of income or financial support does your household receive? [Comment]",
    "If you receive payment in Lebanese Lira, what is your current estimated monthly household income? (If income is in US Dollars, then refer to the current black market exchange).",
]
df = df.drop(columns=columns_to_disregard, errors='ignore')

# Map binary columns (Yes/No, Male/Female)
binary_columns = [
    "Do you find it difficult to refrain from smoking where it is forbidden (church, library, cinema, plane, etc...)?",
    "Do you smoke more frequently during the first hours after waking up than during the rest of the day?",
    "Do you smoke if you are so ill that you are in bed most of the day?",
    "Are you currently able to afford your favorite or preferred cigarette brand(s)?",
    "Do you have close friends?",
    "What is your current marital status?",
    "Employment Status",
    "Gender",
]
binary_mapping = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
for col in binary_columns:
    if col in df.columns:
        df[col] = df[col].map(binary_mapping)

# Reverse scale encoding for personality traits
personality_columns = [
    "I see myself as someone who is extraverted, enthusiastic:",
    "I see myself as someone who is critical, quarrelsome:",
    "I see myself as someone who is dependable, self-disciplined:",
    "I see myself as someone who is anxious, easily upset:",
    "I see myself as someone who is open to new experiences:",
    "I see myself as someone who is reserved, quiet:",
    "I see myself as someone who is sympathetic, warm:",
    "I see myself as someone who is disorganized, careless:",
    "I see myself as someone who is calm, emotionally stable:",
    "I see myself as someone who is conventional, uncreative:",
]
reverse_scale_mapping = {
    "Agree strongly": 0,
    "Agree moderately": 1,
    "Agree a little": 2,
    "Neither agree nor disagree": 3,
    "Disagree a little": 4,
    "Disagree moderately": 5,
    "Disagree strongly": 6,
}
for col in personality_columns:
    if col in df.columns:
        df[col] = df[col].map(reverse_scale_mapping)

# Map ordinal columns with specific options
ordinal_columns_mapping = {
    "How soon after you wake up do you smoke your first cigarette?": {
        "5 minutes or less": 1,
        "6 to 30 minutes": 2,
        "31 to 60 minutes": 3,
        "More than 60 minutes": 4,
    },
    "How would you describe your current smoking behavior compared to your smoking behavior before Lebanon's economic crisis and revolution began in 2019?": {
        "Increased": 3,
        "Remained the same": 2,
        "Decreased": 1,
    },
    "How would you describe your current income sufficiency?": {
        "High: completely covers necessities with a few luxury items": 3,
        "Medium: covers all basic needs": 2,
        "Low: doesn't cover basic needs": 1,
    },
    "To what extent were you financially (negatively) affected by the deterioration of the Lebanese economy?": {
        "Very": 5,
        "Moderately": 4,
        "Slightly": 3,
        "Minimally": 2,
        "Not at all": 1,
    },
}
for col, mapping in ordinal_columns_mapping.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

# Handle numerical ranges and invalid values
if "How old were you the first time you smoked a full cigarette (not just a few puffs)?" in df.columns:
    df["How old were you the first time you smoked a full cigarette (not just a few puffs)?"] = pd.to_numeric(
        df["How old were you the first time you smoked a full cigarette (not just a few puffs)?"], errors='coerce'
    )
    valid_range = (df["How old were you the first time you smoked a full cigarette (not just a few puffs)?"] >= 14) & \
                  (df["How old were you the first time you smoked a full cigarette (not just a few puffs)?"] <= 28)
    avg_value = int(df.loc[valid_range, "How old were you the first time you smoked a full cigarette (not just a few puffs)?"].mean())
    df.loc[~valid_range, "How old were you the first time you smoked a full cigarette (not just a few puffs)?"] = avg_value

# One-hot encoding for nominal columns
nominal_columns = [
    "Which governerate do you live in or spend most of your time in?",
    "What is the highest level of education you have attained?",
    "What is your current employment status?",
    "What is your main source of income?",
    "What type of income or financial support does your household receive?",
    "Sector",
]
df = pd.get_dummies(df, columns=nominal_columns, drop_first=True)

# Save the processed dataset
processed_file_path = "processeddata.csv"
df.to_csv(processed_file_path, index=False)

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import os

# Load dataset
df = pd.read_csv("data/UserRewardSim.csv")

# Handle missing columns gracefully
if "interests" in df.columns:
    df["interests"] = df["interests"].fillna("").apply(lambda x: x.split(";"))
else:
    df["interests"] = [[] for _ in range(len(df))]

if "past_rewards" in df.columns:
    df["past_rewards"] = df["past_rewards"].fillna("").apply(lambda x: x.split(";"))
else:
    df["past_rewards"] = [[] for _ in range(len(df))]

# One-hot encode age_group if exists
if "age_group" in df.columns:
    onehot_age = pd.get_dummies(df["age_group"])
else:
    onehot_age = pd.DataFrame()

# One-hot encode location if exists
if "location" in df.columns:
    onehot_location = pd.get_dummies(df["location"])
else:
    onehot_location = pd.DataFrame()

# Multi-hot encode interests
interest_mlb = MultiLabelBinarizer()
onehot_interests = pd.DataFrame(interest_mlb.fit_transform(df["interests"]), columns=interest_mlb.classes_)

# Multi-hot encode past_rewards
reward_mlb = MultiLabelBinarizer()
onehot_rewards = pd.DataFrame(reward_mlb.fit_transform(df["past_rewards"]), columns=reward_mlb.classes_)

# Normalize activity_score if exists
if "activity_score" in df.columns:
    scaler = MinMaxScaler()
    df["activity_score"] = scaler.fit_transform(df[["activity_score"]])
    activity_score = df[["activity_score"]]
else:
    activity_score = pd.DataFrame()

# Final feature set
final_features = pd.concat(
    [onehot_age, onehot_location, onehot_interests, onehot_rewards, activity_score], axis=1
)

# Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Save preprocessed features and reward labels
final_features.to_csv("data/processed_features.csv", index=False)
pd.DataFrame(reward_mlb.classes_).to_csv("models/reward_labels.csv", index=False, header=False)

print("✅ Preprocessing complete! Features saved to 'data/processed_features.csv'")

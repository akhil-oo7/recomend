# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('data/UserRewardSim.csv')

# Limit dataset size for faster testing
df = df.sample(n=1000, random_state=42)

# Feature engineering: Let's assume 'rewards' column is the target
X = df.drop(columns=['rewards'])  # Features
y = df['rewards']  # Target

# Convert categorical features to numeric if necessary
X = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier with reduced settings
rf = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model using pickle
with open('reward_recommender_model.pkl', 'wb') as model_file:
    pickle.dump(rf, model_file)

# Generate recommendations for all users
all_predictions = rf.predict(X)
df['predicted_rewards'] = all_predictions

# Save recommendations to CSV
df[['user_id', 'predicted_rewards']].to_csv('data/recommendations.csv', index=False)

print("Model training complete and saved!")

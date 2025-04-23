# src/train_model.py

import pandas as pd
import pickle
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
import os

# ✅ Load dataset
df = pd.read_csv("data/UserRewardSim.csv")

# ✅ Preprocess 'rewards' column
df['rewards'] = df['rewards'].fillna("").apply(lambda x: x.split(';'))

# ✅ Encode categorical columns
gender_encoder = LabelEncoder()
location_encoder = LabelEncoder()

df['gender'] = gender_encoder.fit_transform(df['gender'])
df['location'] = location_encoder.fit_transform(df['location'])

# ✅ Encode reward labels (Multi-label binarizer)
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(df['rewards'])

# ✅ Features for prediction
feature_cols = ['age', 'gender', 'location', 'activity_score']
X = df[feature_cols]

# ✅ Train model
model = MultiOutputClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X, Y)

# ✅ Save model and encoders
os.makedirs("models", exist_ok=True)
with open("models/reward_classifier.pkl", 'wb') as f:
    pickle.dump({
        'model': model,
        'sklearn_version': sklearn_version,
        'feature_names': feature_cols,
        'mlb': mlb,
        'gender_encoder': gender_encoder,
        'location_encoder': location_encoder
    }, f)
print(f"✅ Model and encoders saved to models/ (scikit-learn v{sklearn_version})")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import pickle

st.title("🎁 Reward Recommendation Dashboard")
st.markdown("Personalized reward recommendations based on user personas")

# Load models and encoders
@st.cache_data
def load_models():
    try:
        with open('models/reward_classifier.pkl', 'rb') as f:
            model_data = pickle.load(f)
            model = model_data['model']
            mlb = model_data['mlb']
            feature_names = model_data['feature_names']
            gender_encoder = model_data['gender_encoder']
            location_encoder = model_data['location_encoder']
        return model, mlb, feature_names, gender_encoder, location_encoder
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.warning("Please retrain the model with the current scikit-learn version")
        return None, None, None

def load_data(uploaded_file=None):
    """Load data from either uploaded file or default path"""
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            required_columns = ['user_id', 'gender', 'age', 'location']
            missing_cols = [col for col in required_columns if col not in uploaded_df.columns]
            if missing_cols:
                st.error(f"Uploaded file missing required columns: {', '.join(missing_cols)}")
                st.info("Required CSV format example:")
                st.code("user_id,gender,age,location\n1,Male,25,New York\n2,Female,30,Chicago")
                return None
            return uploaded_df
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            return None
            
    # Default data loading
    try:
        recs = pd.read_csv("data/recommendations.csv")
        users = pd.read_csv("data/UserRewardSim.csv")
        merged = pd.merge(recs, users, on="user_id")
        return merged
    except Exception as e:
        st.error(f"Error loading default data files: {str(e)}")
        return None

# Main app
def main():
    model, mlb, feature_names, gender_encoder, location_encoder = load_models()
    
    # File upload section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload your user data (CSV)", 
        type=["csv"],
        help="Upload a CSV file containing user data with required columns"
    )
    
    df = load_data(uploaded_file)
    if df is None:
        return
    
    # User selection
    user_id = st.selectbox('Select User', df['user_id'].unique())
    user_data = df[df['user_id'] == user_id]
    
    # Display user persona
    st.subheader("👤 User Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gender", user_data['gender'].values[0])
        st.metric("Age", int(user_data['age'].values[0]))
    with col2:
        st.metric("Location", user_data['location'].values[0])
        if 'tier' in user_data.columns:
            st.metric("Tier", user_data['tier'].values[0])
        else:
            st.metric("Tier", "Not Available")
    
    # Display reward recommendations
    st.subheader("🏆 Recommended Rewards")
    
    # Get predictions and confidence scores
    if model is None or mlb is None or feature_names is None:
        st.error("Failed to load models. Please retrain the model.")
        return
        
    # Encode categorical features using saved encoders
    user_data_encoded = user_data.copy()
    user_data_encoded['gender'] = gender_encoder.transform(user_data['gender'])
    user_data_encoded['location'] = location_encoder.transform(user_data['location'])
    
    preds = np.array(model.predict_proba(user_data_encoded[feature_names]))
    top_n = 5
    
    # Handle prediction array shapes
    try:
        if len(preds.shape) == 1:
            top_indices = np.argsort(-preds)[:top_n]
            top_scores = preds[top_indices]
        elif len(preds.shape) == 2:
            if preds.shape[0] == 1:  # Single sample prediction
                top_indices = np.argsort(-preds[0])[:top_n]
                top_scores = preds[0][top_indices]
            elif preds.shape[1] > 1:  # Multi-output classifier format
                # Average probabilities across all outputs for ranking
                avg_preds = np.mean(preds, axis=1)
                top_indices = np.argsort(-avg_preds)[:top_n]
                top_scores = avg_preds[top_indices]
            else:  # Single output classifier
                top_indices = np.argsort(-preds[:, 0])[:top_n]
                top_scores = preds[top_indices, 0]
        elif len(preds.shape) == 3:  # Multi-output classifier with 3D array
            # Take mean across all outputs for ranking
            avg_preds = np.mean(preds, axis=2)
            top_indices = np.argsort(-avg_preds[0])[:top_n]
            top_scores = avg_preds[0][top_indices]
        else:
            st.error(f"Unexpected prediction array shape: {preds.shape}. Please check the model output.")
            return
    except Exception as e:
        st.error(f"Error processing predictions: {str(e)}")
        st.warning("Please check the model output format")
        return
        
    reward_classes = mlb.classes_[top_indices]
    
    # Display recommendations with confidence
    for reward, score in zip(reward_classes, top_scores):
        confidence = f"{score*100:.1f}%"
        st.progress(score, text=f"{reward} ({confidence} confidence)")
    
    # Tier-based suggestions
    st.subheader("✨ Tier Benefits")
    if 'tier' in user_data.columns:
        tier = user_data['tier'].values[0]
        if tier == 'Gold':
            st.success("🌟 Premium rewards available for Gold members!")
        elif tier == 'Silver':
            st.info("💎 Upgrade to Gold for premium rewards")
        else:
            st.warning("⬆️ Earn more points to unlock better rewards")
    else:
        st.warning("Tier information not available for this user")

if __name__ == "__main__":
    main()

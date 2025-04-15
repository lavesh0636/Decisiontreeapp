import streamlit as st
import pickle
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Mood Activity Suggestion",
    page_icon="😊",
    layout="centered"
)

# Load the trained model and encoders
@st.cache_resource
def load_model():
    try:
        # Check if the model file exists
        if not os.path.exists('mood_model.pkl'):
            st.error("Model file not found. Please run train_model.py first.")
            return None
            
        # Try to import sklearn first
        try:
            import sklearn
        except ImportError:
            st.error("scikit-learn is not installed. Please run: pip install scikit-learn")
            return None
            
        # Load the model and encoders
        with open('mood_model.pkl', 'rb') as f:
            saved_data = pickle.load(f)
            return saved_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the model and encoders
saved_data = load_model()

# If model loading failed, show error and stop
if saved_data is None:
    st.stop()

model = saved_data['model']
mood_encoder = saved_data['mood_encoder']
energy_encoder = saved_data['energy_encoder']
stress_encoder = saved_data['stress_encoder']
activity_encoder = saved_data['activity_encoder']

# Get the class names
mood_classes = mood_encoder.classes_
energy_classes = energy_encoder.classes_
stress_classes = stress_encoder.classes_
activity_classes = activity_encoder.classes_

# App title and description
st.title("Mood Activity Suggestion")
st.markdown("""
    Select your current mood, energy level, stress level, sleep hours, and available time 
    to get a personalized activity suggestion. This recommendation is based on a trained 
    decision tree model.
""")

# Create columns for the form
col1, col2, col3 = st.columns(3)

with col1:
    mood = st.selectbox(
        "Current Mood",
        options=mood_classes,
        index=0
    )

with col2:
    energy = st.selectbox(
        "Energy Level",
        options=energy_classes,
        index=0
    )

with col3:
    stress = st.selectbox(
        "Stress Level",
        options=stress_classes,
        index=0
    )

# Additional features
col4, col5 = st.columns(2)

with col4:
    sleep_hours = st.slider(
        "Sleep Hours (Last Night)",
        min_value=0,
        max_value=12,
        value=8,
        step=1
    )

with col5:
    time_available = st.slider(
        "Time Available (in minutes)",
        min_value=5,
        max_value=120,
        value=30,
        step=5
    )

# Prediction button
if st.button("Get Activity Suggestion"):
    try:
        # Encode the input features
        mood_encoded = mood_encoder.transform([mood])[0]
        energy_encoded = energy_encoder.transform([energy])[0]
        stress_encoded = stress_encoder.transform([stress])[0]

        # Prepare input for prediction
        input_data = np.array([[mood_encoded, energy_encoded, stress_encoded, sleep_hours, time_available]])

        # Make prediction
        prediction_encoded = model.predict(input_data)[0]
        suggested_activity = activity_encoder.inverse_transform([prediction_encoded])[0]

        # Display result with styling
        st.success(f"Suggested Activity: **{suggested_activity}**")
        
        # Add some emojis based on the activity
        activity_emojis = {
            'CBT Worksheet': '📝',
            'Call Friend': '📞',
            'Deep Breathing': '🧘',
            'Gratitude': '🙏',
            'Journaling': '📔',
            'Music Therapy': '🎵',
            'Walk': '🚶'
        }
        
        st.markdown(f"### {activity_emojis[suggested_activity]} {suggested_activity}")
        
        # Add some helpful tips based on the activity
        activity_tips = {
            'CBT Worksheet': "Take 10-15 minutes to complete a CBT worksheet to help process your thoughts and emotions.",
            'Call Friend': "Reach out to a trusted friend or family member for support and connection.",
            'Deep Breathing': "Find a quiet space and practice deep breathing exercises for 5-10 minutes.",
            'Gratitude': "Write down 3 things you're grateful for today.",
            'Journaling': "Spend 10-15 minutes writing about your thoughts and feelings.",
            'Music Therapy': "Listen to calming or uplifting music that matches your current mood.",
            'Walk': "Take a 15-20 minute walk, preferably in nature if possible."
        }
        
        st.info(activity_tips[suggested_activity])

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# Add some additional information
st.markdown("---")
st.markdown("""
    ### About This App
    This application uses a decision tree model trained on various mood states, energy levels, 
    stress levels, sleep hours, and available time to suggest appropriate activities. The 
    suggestions are designed to help improve your mental well-being based on your current 
    state and circumstances.
""") 
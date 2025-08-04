import streamlit as st
import pandas as pd
import pickle
import base64

# Load trained pipeline
with open("OnlineLearningDropoutPrediction.pkl", "rb") as f:
    pipe = pickle.load(f)

# Set Page Config
st.set_page_config(page_title="Dropout Predictor", page_icon="🎓", layout="centered")

# Set background image from local file
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded_string = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-attachment: fixed;
        }}
        .title-text {{
            color: #FFFFFF;
            text-shadow: 1px 1px 5px #000000;
        }}
        .label-text {{
            color: #FFFFFF;
            font-weight: 600;
        }}
        .input-label {{
            color: #FFDD57;
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
            display: block;
        }}
        .stSlider > div > div > div {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background image
add_bg_from_local("images.jpg")

# Title
st.markdown("<h1 class='title-text' style='text-align:center;'>🎓 Online Learning Dropout Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='title-text' style='text-align:center;'>Predict if a student will Dropout or Continue their online course.</p>", unsafe_allow_html=True)

# Input Section
st.write("### <span class='label-text'>📥 Enter Student Engagement Data:</span>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    st.markdown("<span class='input-label'>Login Frequency per Week</span>", unsafe_allow_html=True)
    login_frequency = st.number_input("", min_value=0.0, max_value=14.0, step=0.1, value=5.0, key="login_frequency")

    st.markdown("<span class='input-label'>Assignment Completion Rate</span>", unsafe_allow_html=True)
    assignment_completion = st.slider("", 0.0, 1.0, 0.5, key="assignment_completion")

    st.markdown("<span class='input-label'>Average Quiz Score (%)</span>", unsafe_allow_html=True)
    quiz_score = st.slider("", 0.0, 100.0, 50.0, key="quiz_score")

# Column 2 inputs
with col2:
    st.markdown("<span class='input-label'>Forum Posts per Week</span>", unsafe_allow_html=True)
    forum_posts = st.number_input("", min_value=0.0, max_value=10.0, step=0.1, value=2.0, key="forum_posts")

    st.markdown("<span class='input-label'>Video Watch Percentage</span>", unsafe_allow_html=True)
    video_watch = st.slider("", 0.0, 100.0, 50.0, key="video_watch")

    st.markdown("<span class='input-label'>Forum Sentiment</span>", unsafe_allow_html=True)
    sentiment = st.radio("", options=["Positive", "Negative"], horizontal=True, key="sentiment_radio")

# Convert sentiment to numeric
sentiment_score = 0 if sentiment == "Positive" else 1

# Prediction logic
if st.button("🔍 Predict Dropout Risk"):
    try:
        input_df = pd.DataFrame([{
            'login_frequency_per_week': login_frequency,
            'forum_posts_per_week': forum_posts,
            'assignment_completion_rate': assignment_completion,
            'video_watch_percentage': video_watch,
            'average_quiz_score': quiz_score,
            'forum_sentiment_score': sentiment_score
        }])

        prediction = pipe.predict(input_df)[0]
        prediction_proba = pipe.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.markdown(
                f"""
                <div style='background-color:#e6ffe6; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:green;'>🎉 Student Likely to Continue</h2>
                    <p>Probability of Continuing: <strong>{(1 - prediction_proba)*100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
            
        else:
            st.markdown(
                f"""
                <div style='background-color:#ffe6e6; padding:20px; border-radius:10px; text-align:center;'>
                    <h2 style='color:red;'>🚨 Dropout Risk Detected!</h2>
                    <p>Probability of Dropout: <strong>{prediction_proba*100:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction Failed: {e}")

# Footer
st.markdown("<p style='text-align:center; color:white;'>Developed by <strong>Shraddha Kharade</strong></p>", unsafe_allow_html=True)

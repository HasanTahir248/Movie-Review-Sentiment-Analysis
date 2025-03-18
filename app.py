import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

# Load model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Streamlit app
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Analyze the **sentiment** of your movie reviews as **Positive** or **Negative**.")

# Text input
review = st.text_area("Enter your movie review here:", height=200, placeholder="Type your review...")

# Prediction button
if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review to analyze.")
    else:
        processed_review = preprocess(review)
        vectorized_review = vectorizer.transform([processed_review])
        prediction = model.predict(vectorized_review)
        sentiment = "Positive 😊" if prediction[0] == 1 else "Negative 😞"
        
        # Show result
        st.success(f"### Prediction: {sentiment}")

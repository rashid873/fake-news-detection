import streamlit as st
import gzip
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data (safe to call multiple times)
nltk.download('stopwords')
nltk.download('wordnet')

# Load zipped model
try:
    with gzip.open("model.pkl.gz", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Could not load the model. Please make sure 'model.pkl.gz' exists and is valid.")
    st.stop()

# Load vectorizer
try:
    with open("tfidf.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except Exception as e:
    st.error("❌ Could not load the vectorizer. Please make sure 'tfidf.pkl' exists and is valid.")
    st.stop()

# Text preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fa;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .title {
            color: #2c3e50;
            font-size: 2.5rem;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">📰 Fake News Detection App</div>', unsafe_allow_html=True)

st.write("Enter a news article text below and check whether it is real or fake:")

user_input = st.text_area("✍️ Your news content here:")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to make a prediction.")
    else:
        processed = preprocess_text(user_input)
        vectorized = vectorizer.transform([processed])
        prediction = model.predict(vectorized)

        if prediction[0] == 0:
            st.error("🚫 The news is likely **Fake**.")
        else:
            st.success("✅ The news appears to be **Real**.")

st.markdown('</div>', unsafe_allow_html=True)

import streamlit as st
import pickle
import re
import string
import gzip
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the compressed model and vectorizer
with gzip.open("model.pkl.gz", "rb") as f:
    model = pickle.load(f)

with open("tfidf.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Streamlit app UI setup
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown("<h1 style='text-align: center; color: #3366cc;'>📰 Fake News Detection Web App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news article below and find out if it's Real or Fake using Machine Learning!</p>", unsafe_allow_html=True)
st.write("")

# Input text area
user_input = st.text_area("🧾 Paste your news article text here:", height=200)

if st.button("🔍 Predict Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a news article to continue.")
    else:
        cleaned = preprocess_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)

        st.write("---")
        if prediction[0] == 0:
            st.error("🚫 This news is likely **FAKE**. Please verify from reliable sources.")
        else:
            st.success("✅ This news appears to be **REAL**. Looks trustworthy!")
        st.write("---")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: small;'>Made with ❤️ for Internship Project</p>", unsafe_allow_html=True)

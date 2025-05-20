import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved objects
model = load_model("fake_news_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article or paragraph below:")

user_input = st.text_area("News Text", height=250)

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        seq = tokenizer.texts_to_sequences([cleaned])
        pad = pad_sequences(seq, maxlen=200)
        pred = model.predict(pad)[0][0]

        if pred > 0.5:
            st.success("✅ This news is *Real*.")
        else:
            st.error("❌ This news is *Fake*.")


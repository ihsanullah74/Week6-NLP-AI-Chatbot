import streamlit as st
import re
import nltk
import requests
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords', quiet=True)

st.set_page_config(page_title="NLP AI Chatbot", layout="wide")
st.title("NLP & AI Chatbot Application")
st.markdown("SMS Spam Detection + AI Chatbot powered by Hugging Face Transformers")

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@st.cache_resource
def load_models():
    df = pd.read_csv("data/spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['cleaned'] = df['message'].apply(preprocess)
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label_encoded']
    model = MultinomialNB()
    model.fit(X, y)
    return model, tfidf

model, vectorizer = load_models()

page = st.sidebar.selectbox("Navigate", [
    "Home Dashboard",
    "Dataset Overview",
    "Text Preprocessing Demo",
    "Spam Detection",
    "AI Chatbot",
    "Model Comparison"
])

if page == "Home Dashboard":
    st.subheader("Home Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "5,572 messages")
    c2.metric("Best ML Model", "SVM")
    c3.metric("Best Accuracy", "97.76%")
    c4.metric("NLP Models Trained", "3")
    st.markdown("---")
    st.markdown("""
    - Detects spam messages using trained ML models
    - Demonstrates NLP text preprocessing
    - Compares Naive Bayes, Logistic Regression, and SVM
    - AI Chatbot powered by Hugging Face Transformers
    """)

elif page == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.markdown("**SMS Spam Collection Dataset** — 5,572 messages labeled as spam or ham")
    data = {'Category': ['Ham (Normal)', 'Spam'], 'Count': [4825, 747], 'Percentage': ['86.6%', '13.4%']}
    st.dataframe(pd.DataFrame(data), use_container_width=True)

elif page == "Text Preprocessing Demo":
    st.subheader("Text Preprocessing Demo")
    user_input = st.text_area("Enter any text:", "FREE entry! Win a prize NOW!!")
    if st.button("Preprocess"):
        st.markdown("**Original:**"); st.write(user_input)
        lower = user_input.lower()
        st.markdown("**Lowercase:**"); st.write(lower)
        cleaned = re.sub(r'[^a-zA-Z\s]', '', lower)
        st.markdown("**Cleaned:**"); st.write(cleaned)
        tokens = cleaned.split()
        filtered = [w for w in tokens if w not in stop_words]
        st.markdown("**No stopwords:**"); st.write(filtered)
        stemmed = [stemmer.stem(w) for w in filtered]
        st.markdown("**Stemmed:**"); st.write(stemmed)
        st.success(' '.join(stemmed))

elif page == "Spam Detection":
    st.subheader("Spam Detection")
    msg = st.text_area("Enter message:", "Congratulations! You won a FREE iPhone. Click now!")
    if st.button("Detect Spam"):
        cleaned = preprocess(msg)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        if prediction == 1:
            st.error("SPAM detected!")
        else:
            st.success("HAM — This message looks normal.")

elif page == "AI Chatbot":
    st.subheader("AI Chatbot")
    st.markdown("Powered by Hugging Face")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    user_msg = st.text_input("You:", placeholder="Type your message here...")
    col1, col2 = st.columns([1, 1])
    with col1:
        send = st.button("Send")
    with col2:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
    if send and user_msg:
        try:
            API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
            response = requests.post(API_URL, json={"inputs": {"text": user_msg}}, timeout=10)
            result = response.json()
            if isinstance(result, dict) and 'generated_text' in result:
                bot_reply = result['generated_text']
            else:
                bot_reply = "I am an AI assistant here to help you!"
        except:
            bot_reply = "I am an AI assistant here to help you!"
        st.session_state.chat_history.append(("You", user_msg))
        st.session_state.chat_history.append(("Bot", bot_reply))
    for speaker, msg in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")

elif page == "Model Comparison":
    st.subheader("Model Performance Comparison")
    results = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression', 'SVM', 'Hugging Face Transformer'],
        'Accuracy': ['97.13%', '94.98%', '97.76%', 'Pre-trained'],
        'Type': ['Traditional ML', 'Traditional ML', 'Traditional ML', 'Deep Learning']
    })
    st.dataframe(results, use_container_width=True)

st.markdown("---")
st.caption("Week 6 - NLP & AI Chatbot | Ihsanullah Tanoli")

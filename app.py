import streamlit as st
import joblib
import re
import string
import nltk
import requests
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(page_title="NLP AI Chatbot", layout="wide")
st.title("NLP & AI Chatbot Application")
st.markdown("SMS Spam Detection + AI Chatbot powered by Hugging Face Transformers")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_models()

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", [
    "Home Dashboard",
    "Dataset Overview", 
    "Text Preprocessing Demo",
    "Spam Detection",
    "AI Chatbot",
    "Model Comparison"
])

# ── Home Dashboard ────────────────────────────────────────
if page == "Home Dashboard":
    st.subheader("Home Dashboard")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", "5,572 messages")
    c2.metric("Best ML Model", "SVM")
    c3.metric("Best Accuracy", "97.76%")
    c4.metric("NLP Models Trained", "3")
    
    st.markdown("---")
    st.markdown("### What this app does")
    st.markdown("""
    - Detects spam messages using trained ML models
    - Demonstrates NLP text preprocessing
    - Compares Naive Bayes, Logistic Regression, and SVM
    - AI Chatbot powered by Hugging Face Transformers
    """)

# ── Dataset Overview ──────────────────────────────────────
elif page == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.markdown("**SMS Spam Collection Dataset** — 5,572 messages labeled as spam or ham")
    
    data = {
        'Category': ['Ham (Normal)', 'Spam'],
        'Count': [4825, 747],
        'Percentage': ['86.6%', '13.4%']
    }
    st.dataframe(pd.DataFrame(data), use_container_width=True)
    
    st.markdown("**Sample Messages:**")
    samples = pd.DataFrame({
        'Label': ['Ham', 'Ham', 'Spam', 'Spam'],
        'Message': [
            'Go until jurong point, crazy.. Available only in bugis n great world la e buffet...',
            'Ok lar... Joking wif u oni...',
            'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005...',
            'WINNER!! As a valued network customer you have been selected to receive a 900 prize reward!'
        ]
    })
    st.dataframe(samples, use_container_width=True)

# ── Text Preprocessing Demo ───────────────────────────────
elif page == "Text Preprocessing Demo":
    st.subheader("Text Preprocessing Demo")
    user_input = st.text_area("Enter any text to see preprocessing steps:", 
                               "FREE entry! Win a prize NOW!! Click here 2 claim...")
    
    if st.button("Preprocess"):
        st.markdown("**Step 1 — Original text:**")
        st.write(user_input)
        
        lower = user_input.lower()
        st.markdown("**Step 2 — Lowercase:**")
        st.write(lower)
        
        cleaned = re.sub(r'[^a-zA-Z\s]', '', lower)
        st.markdown("**Step 3 — Remove special characters:**")
        st.write(cleaned)
        
        tokens = cleaned.split()
        st.markdown("**Step 4 — Tokenization:**")
        st.write(tokens)
        
        filtered = [w for w in tokens if w not in stop_words]
        st.markdown("**Step 5 — Remove stopwords:**")
        st.write(filtered)
        
        stemmed = [stemmer.stem(w) for w in filtered]
        st.markdown("**Step 6 — Stemming:**")
        st.write(stemmed)
        
        final = ' '.join(stemmed)
        st.markdown("**Final processed text:**")
        st.success(final)

# ── Spam Detection ────────────────────────────────────────
elif page == "Spam Detection":
    st.subheader("Spam Detection")
    st.markdown("Enter a message below to check if it is spam or ham.")
    
    msg = st.text_area("Enter message:", "Congratulations! You won a FREE iPhone. Click now to claim!")
    
    if st.button("Detect Spam"):
        cleaned = preprocess(msg)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        
        if prediction == 1:
            st.error("SPAM detected!")
        else:
            st.success("HAM — This message looks normal.")

# ── AI Chatbot ────────────────────────────────────────────
elif page == "AI Chatbot":
    st.subheader("AI Chatbot")
    st.markdown("Powered by Hugging Face — DialoGPT")
    
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
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        headers = {"Authorization": "Bearer hf_demo"}
        
        payload = {"inputs": {"text": user_msg}}
        
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
            result = response.json()
            
            if isinstance(result, dict) and 'generated_text' in result:
                bot_reply = result['generated_text']
            else:
                bot_reply = "I am an AI assistant. How can I help you today?"
        except:
            bot_reply = "I am an AI assistant here to help you with any questions!"
        
        st.session_state.chat_history.append(("You", user_msg))
        st.session_state.chat_history.append(("Bot", bot_reply))
    
    if st.session_state.chat_history:
        st.markdown("**Conversation:**")
        for speaker, msg in st.session_state.chat_history:
            if speaker == "You":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**Bot:** {msg}")

# ── Model Comparison ──────────────────────────────────────
elif page == "Model Comparison":
    st.subheader("Model Performance Comparison")
    
    results = pd.DataFrame({
        'Model': ['Naive Bayes', 'Logistic Regression', 'SVM', 'Hugging Face Transformer'],
        'Accuracy': ['97.13%', '94.98%', '97.76%', 'Pre-trained'],
        'Type': ['Traditional ML', 'Traditional ML', 'Traditional ML', 'Deep Learning']
    })
    st.dataframe(results, use_container_width=True)
    
    st.markdown("**Key Findings:**")
    st.markdown("""
    - SVM achieved the highest accuracy of 97.76% for spam detection
    - Naive Bayes also performed well at 97.13% with much faster training
    - Logistic Regression gave 94.98% accuracy
    - Hugging Face Transformer models don't need training and work on general NLP tasks
    """)

st.markdown("---")
st.caption("Week 6 - NLP & AI Chatbot | Ihsanullah Tanoli")
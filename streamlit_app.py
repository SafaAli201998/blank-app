import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import shap

# Load models and vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('lr_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load dataset
@st.cache
def load_data():
    # Replace 'dataset.csv' with your file name
    data = pd.read_csv('text.csv')
    return data

data = load_data()

# Streamlit App
st.title("Emotion Classification Dashboard")
st.sidebar.title("Navigation")

# Sidebar Navigation
menu = st.sidebar.selectbox("Menu", ["Home", "Dataset Exploration", "Model Performance", "Interactive Predictions", "Explainability", "Summary"])

if menu == "Home":
    st.header("Overview of Emotion Classification")
    st.write("This project focuses on classifying emotions in text using Machine Learning and Deep Learning models.")

    # Dataset Distribution
    emotion_counts = data['label'].value_counts()
    st.subheader("Emotion Distribution")
    fig, ax = plt.subplots()
    sns.barplot(x=emotion_counts.index, y=emotion_counts.values, ax=ax)
    ax.set_title("Emotion Distribution in Dataset")
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    st.pyplot(fig)

elif menu == "Dataset Exploration":
    st.header("Dataset Exploration")
    st.write("View and explore the dataset.")

    st.dataframe(data.head())

    # Text Length Distribution
    data['text_length'] = data['text'].apply(len)
    st.subheader("Text Length Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data['text_length'], bins=30, kde=True, ax=ax)
    ax.set_title("Text Length Distribution")
    st.pyplot(fig)

elif menu == "Model Performance":
    st.header("Model Performance Comparison")
    st.write("Compare the performance of different models.")

    # Sample performance metrics
    performance = pd.DataFrame({
        "Model": ["Naive Bayes", "Logistic Regression", "XGBoost", "Neural Network"],
        "Accuracy": [84, 89, 90, 88],
        "F1-Score": [0.83, 0.89, 0.90, 0.88]
    })
    st.table(performance)

    # Visualization
    st.subheader("Performance Comparison")
    fig, ax = plt.subplots()
    performance.plot(x="Model", kind="bar", ax=ax)
    ax.set_title("Model Performance Comparison")
    st.pyplot(fig)

elif menu == "Interactive Predictions":
    st.header("Interactive Predictions")
    st.write("Enter text and get emotion predictions.")

    user_input = st.text_input("Enter your text:", "I am feeling so happy today!")
    if st.button("Predict Emotion"):
        processed_input = vectorizer.transform([user_input])
        prediction = model.predict(processed_input)
        st.write(f"Predicted Emotion: {prediction[0]}")

elif menu == "Explainability":
    st.header("Explainability with SHAP")
    st.write("Visualize how features contribute to predictions.")

    # SHAP Explanation
    explainer = shap.LinearExplainer(model, vectorizer.transform(data['text'][:100]))
    shap_values = explainer(vectorizer.transform(data['text'][:5]))

    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, vectorizer.transform(data['text'][:5]).toarray(), show=False)
    st.pyplot(fig)

elif menu == "Summary":
    st.header("Project Summary")
    st.write("This dashboard demonstrates emotion classification using ML and DL models with interactive visualizations.")
    st.write("Key Insights:")
    st.write("1. XGBoost achieved the best accuracy of 90%.")
    st.write("2. SHAP visualizations help in interpreting model predictions.")

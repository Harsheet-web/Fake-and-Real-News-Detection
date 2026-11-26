# Application imports and model loading
import streamlit as st
import joblib
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Loading the trained model
@st.cache_resource
def load_model():
    return joblib.load("models/fake_news_detector.joblib")


model_package = load_model()
model = model_package["model"]
CONFIDENCE_THRESHOLD = model_package["confidence_threshold"]
LABEL_MAPPING = model_package["label_mapping"]


# Initialise prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Reproducing the training preprocessing

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

stop_words = set(stopwords.words("english"))

negation_words = {"no", "nor", "not", "never"}

stop_words = stop_words - negation_words

stemmer = PorterStemmer()


def preprocess_text(text):

    text = str(text)
    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = [
        stemmer.stem(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]

    return " ".join(cleaned_tokens)


def remove_reuters(text):

    text = re.sub(
        r"\breuter(?:s)?\b",
        "",
        text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------
# STREAMLIT USER INTERFACE
# --------------------------------

st.title("📰 Fake News Detection System")

st.markdown(
    """
    ### AI-Powered News Classification

    An NLP and Machine Learning application that analyses
    news article text and predicts whether its writing
    patterns resemble Fake News or Real News.

    ---
    """
)

st.info(
    "📌 Paste the complete body of a news article "
    "below to get started."
)

article_text = st.text_area(
    "Paste your news article here:",
    height=250,
    placeholder="Paste the complete news article..."
)

if st.button("Analyse Article"):

    if not article_text.strip():
        st.warning("Please enter a news article.")

    else:
        # Step 1: Preprocess the article
        cleaned_text = preprocess_text(article_text)

        # Step 2: Remove Reuters markers
        processed_text = remove_reuters(cleaned_text)

        if len(processed_text.split()) < 3:
            st.warning("Please enter a longer article.")

        else:
            # Step 3: Generate prediction
            prediction = model.predict([processed_text])[0]

            # Step 4: Calculate class probabilities
            probabilities = model.predict_proba([processed_text])[0]

            fake_probability = probabilities[0]
            real_probability = probabilities[1]

            # Step 5: Calculate confidence
            confidence = max(probabilities)

            # Step 5.1: Store prediction history

            predicted_label = LABEL_MAPPING[prediction]

            history_record = {
                "article": article_text[:150],
                "prediction": predicted_label,
                "confidence": confidence,
                "fake_probability": fake_probability,
                "real_probability": real_probability
            }

            st.session_state.prediction_history.append(history_record)

            # Step 6: Display results

            st.divider()
            st.subheader("📊 Analysis Results")

            col1, col2 = st.columns(2)

            with col1:
                    st.metric(
                        label="Fake News Probability",
                        value=f"{fake_probability:.2%}"
                    )

            with col2:
                    st.metric(
                        label="Real News Probability",
                        value=f"{real_probability:.2%}"
                    )

            st.write("### Model Confidence")

            st.progress(float(confidence))

            st.write(f"Confidence Score: **{confidence:.2%}**")

            # Step 7: Final decision

            if confidence < CONFIDENCE_THRESHOLD:

                st.warning(
                        "⚠️ Manual Review Recommended — "
                        "The model is not sufficiently confident."
                 )

            else:

                predicted_label = LABEL_MAPPING[prediction]

                if prediction == 0:
                    st.error(
                        f"🚨 Prediction: {predicted_label}"
                    )

                else:
                    st.success(
                        f"✅ Prediction: {predicted_label}"
                )

st.divider()

st.caption(
    "Disclaimer: This is an experimental machine learning "
    "classifier trained on a historical dataset. It analyses "
    "text patterns and does not independently verify facts, "
    "sources, or the authenticity of news."
)

# --------------------------------
# PREDICTION HISTORY
# --------------------------------

st.divider()

st.subheader("📚 Prediction History")

if st.session_state.prediction_history:

    if st.button("🗑️ Clear Prediction History"):
        st.session_state.prediction_history = []
        st.rerun()

if st.session_state.prediction_history:

    st.write(
        f"Total articles analysed: "
        f"**{len(st.session_state.prediction_history)}**"
    )

    for index, record in enumerate(
        reversed(st.session_state.prediction_history),
        start=1
    ):

        with st.expander(
            f"{index}. {record['prediction']} "
            f"({record['confidence']:.2%})"
        ):

            st.write("**Article:**")
            st.write(record["article"])

            st.write(
                f"**Fake Probability:** "
                f"{record['fake_probability']:.2%}"
            )

            st.write(
                f"**Real Probability:** "
                f"{record['real_probability']:.2%}"
            )

            st.write(
                f"**Confidence:** "
                f"{record['confidence']:.2%}"
            )

else:

    st.info("No prediction history available yet.")
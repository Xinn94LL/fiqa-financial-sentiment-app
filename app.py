# Program title: Financial News Sentiment Analyzer
# ISOM5240 Final Project
# Pipeline 1: Financial sentiment classification using fine-tuned DistilBERT
# Pipeline 2: Investment interpretation generation using rule-based templates

import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Financial News Sentiment Analyzer",
    page_icon="📈",
    layout="centered"
)


# --------------------------------------------------
# Load Hugging Face model
# --------------------------------------------------
@st.cache_resource
def load_sentiment_model():
    """
    Load the fine-tuned financial sentiment classification model from Hugging Face.

    Model:
    Xinn94L/fiqa-financial-sentiment-distilbert

    Output labels:
    POSITIVE / NEUTRAL / NEGATIVE
    """
    model_name = "Xinn94L/fiqa-financial-sentiment-distilbert"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model


# --------------------------------------------------
# Pipeline 1: Financial sentiment classification
# --------------------------------------------------
def classify_financial_sentiment(headline, target, aspect):
    """
    Classify financial news sentiment.

    Args:
        headline: financial news headline
        target: target company or asset
        aspect: event type or topic

    Returns:
        label: POSITIVE / NEUTRAL / NEGATIVE
        score: confidence score
        model_input: formatted input text used by the model
    """
    tokenizer, model = load_sentiment_model()

    model_input = (
        f"Headline: {headline} "
        f"Target: {target} "
        f"Aspect: {aspect}"
    )

    inputs = tokenizer(
        model_input,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    # DistilBERT does not use token_type_ids.
    # Some tokenizers may return token_type_ids, so we remove it to avoid TypeError.
    if "token_type_ids" in inputs:
        inputs.pop("token_type_ids")

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        score = probabilities[0][predicted_id].item()

    label = model.config.id2label[predicted_id]

    return label, score, model_input


# --------------------------------------------------
# Pipeline 2: Investment interpretation generation
# --------------------------------------------------
def generate_investment_interpretation(headline, target, aspect, label, score):
    """
    Generate a short business / investment interpretation based on the sentiment result.

    This pipeline is rule-based for stability and interpretability.
    """
    confidence_percent = round(score * 100, 2)

    if label == "POSITIVE":
        interpretation = (
            f"This headline appears positive for {target}. "
            f"The event is related to {aspect}, which may improve the company's "
            f"business outlook, investor confidence, or market perception. "
            f"The model confidence is {confidence_percent}%. "
            f"However, users should still consider valuation, market expectations, "
            f"sector conditions, and broader macroeconomic factors before making decisions."
        )

    elif label == "NEGATIVE":
        interpretation = (
            f"This headline appears negative for {target}. "
            f"The event is related to {aspect}, which may indicate market pressure, "
            f"operational challenges, regulatory uncertainty, or weaker investor sentiment. "
            f"The model confidence is {confidence_percent}%. "
            f"Users should review the full news context and related financial indicators "
            f"before drawing an investment conclusion."
        )

    else:
        interpretation = (
            f"This headline appears neutral for {target}. "
            f"The event is related to {aspect}, but the headline may not clearly imply "
            f"a positive or negative financial impact. "
            f"The model confidence is {confidence_percent}%. "
            f"Users may need additional information, such as earnings data, management comments, "
            f"or actual market reaction, to better understand the investment implication."
        )

    return interpretation


# --------------------------------------------------
# Helper function: label explanation
# --------------------------------------------------
def get_label_explanation(label):
    """
    Provide a plain-language explanation of the predicted label.
    """
    if label == "POSITIVE":
        return "The news may have a positive implication for the target company or asset."
    elif label == "NEGATIVE":
        return "The news may have a negative implication for the target company or asset."
    else:
        return "The news may not have a clear positive or negative implication."


# --------------------------------------------------
# Streamlit user interface
# --------------------------------------------------
st.title("📈 Financial News Sentiment Analyzer")

st.write(
    "This app classifies financial news headlines into **Positive**, **Neutral**, "
    "or **Negative** sentiment using a fine-tuned DistilBERT model. "
    "It also generates a short investment interpretation for the target company."
)

st.info(
    "Model used: `Xinn94L/fiqa-financial-sentiment-distilbert`"
)

st.divider()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("About this App")

st.sidebar.write(
    "This app was built for the ISOM5240 final project. "
    "It uses a fine-tuned transformer model trained on the FIQA financial sentiment dataset."
)

st.sidebar.write("**Pipeline 1:** Financial sentiment classification")
st.sidebar.write("**Pipeline 2:** Investment interpretation generation")

st.sidebar.warning(
    "Educational use only. This app does not provide financial advice."
)


# --------------------------------------------------
# Example selection
# --------------------------------------------------
st.subheader("Try an Example")

example_option = st.selectbox(
    "Choose an example or enter your own news below:",
    [
        "Positive example",
        "Neutral example",
        "Negative example",
        "Custom input"
    ]
)

if example_option == "Positive example":
    default_headline = "AstraZeneca wins FDA approval for key new lung cancer pill"
    default_target = "AstraZeneca"
    default_aspect = "Corporate/Regulatory"

elif example_option == "Neutral example":
    default_headline = "Royal Mail chairman Donald Brydon set to step down"
    default_target = "Royal Mail"
    default_aspect = "Corporate/Appointment"

elif example_option == "Negative example":
    default_headline = "Slump in Weir leads FTSE down from record high"
    default_target = "Weir"
    default_aspect = "Market/Volatility"

else:
    default_headline = ""
    default_target = ""
    default_aspect = ""


# --------------------------------------------------
# User input section
# --------------------------------------------------
st.subheader("Input Financial News")

headline = st.text_area(
    "Financial News Headline",
    value=default_headline,
    height=100,
    placeholder="Example: AstraZeneca wins FDA approval for key new lung cancer pill"
)

target = st.text_input(
    "Target Company / Asset",
    value=default_target,
    placeholder="Example: AstraZeneca"
)

aspect = st.text_input(
    "Aspect / Event Type",
    value=default_aspect,
    placeholder="Example: Corporate/Regulatory"
)

analyze_button = st.button("Analyze Sentiment", type="primary")


# --------------------------------------------------
# Main analysis
# --------------------------------------------------
if analyze_button:
    if not headline.strip() or not target.strip() or not aspect.strip():
        st.warning("Please enter the headline, target company / asset, and aspect before analysis.")

    else:
        with st.spinner("Analyzing financial sentiment..."):
            label, score, model_input = classify_financial_sentiment(
                headline=headline,
                target=target,
                aspect=aspect
            )

            interpretation = generate_investment_interpretation(
                headline=headline,
                target=target,
                aspect=aspect,
                label=label,
                score=score
            )

            label_explanation = get_label_explanation(label)

        st.divider()

        # Pipeline 1 result
        st.subheader("Pipeline 1: Sentiment Classification Result")

        if label == "POSITIVE":
            st.success(f"Sentiment Prediction: {label}")
        elif label == "NEGATIVE":
            st.error(f"Sentiment Prediction: {label}")
        else:
            st.info(f"Sentiment Prediction: {label}")

        st.metric(
            label="Confidence Score",
            value=f"{score:.2%}"
        )

        st.write("**Label Explanation:**")
        st.write(label_explanation)

        with st.expander("View model input text"):
            st.code(model_input)

        # Pipeline 2 result
        st.subheader("Pipeline 2: Investment Interpretation")
        st.write(interpretation)

        st.divider()

        # Output summary table
        st.subheader("Result Summary")

        result_summary = {
            "Headline": headline,
            "Target": target,
            "Aspect": aspect,
            "Predicted Sentiment": label,
            "Confidence Score": f"{score:.2%}"
        }

        summary_df = pd.DataFrame(
            list(result_summary.items()),
            columns=["Field", "Value"]
        )

        st.table(summary_df)

        st.caption(
            "Disclaimer: This app is for educational and demonstration purposes only. "
            "It does not provide financial advice or investment recommendations."
        )


# --------------------------------------------------
# Project details
# --------------------------------------------------
st.divider()

with st.expander("Project Methodology"):
    st.write(
        """
        **Dataset:** TheFinAI/fiqa-sentiment-classification

        **Label conversion rule:**
        - score >= 0.2 → POSITIVE
        - score <= -0.2 → NEGATIVE
        - otherwise → NEUTRAL

        **Candidate models compared:**
        - DistilBERT
        - BERT
        - FinBERT

        **Final selected model:**
        - Early Stopping DistilBERT

        **Final test performance:**
        - Accuracy: 0.8120
        - Macro F1: 0.7096
        - Weighted F1: 0.8041
        """
    )

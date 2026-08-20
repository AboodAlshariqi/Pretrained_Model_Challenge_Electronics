import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Electronics Review Analyzer", page_icon="\U0001F50C")

SENTIMENT_MODEL_IDS = [
    "siebert/sentiment-roberta-large-english",
    "finiteautomata/bertweet-base-sentiment-analysis",
    "j-hartmann/emotion-english-distilroberta-base",
]

EMOTION_POLARITY = {
    "joy": "positive",
    "surprise": "positive",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
    "neutral": "negative",
}

TOPIC_DESCRIPTIONS = {
    "Battery Life": "comments about battery life, charging speed, or how long a charge lasts",
    "Build Quality": "comments about physical durability, materials, hinges, cracks, or manufacturing defects",
    "Audio Quality": "comments about sound quality, noise cancellation, or microphone clarity",
    "Customer Support": "comments about customer service, support response time, or replacements",
    "Shipping & Packaging": "comments about delivery speed, packaging condition, or receiving the wrong item",
    "Software/App": "comments about the companion app, firmware, pairing, or setup process",
    "Price/Value": "comments about price, cost, or value for money",
    "Display/Screen": "comments about screen brightness, resolution, dead pixels, or visibility",
}

EXAMPLE_REVIEWS = [
    "Battery easily lasts two full days of heavy use, way better than my old earbuds.",
    "Bluetooth keeps disconnecting every few minutes, unusable for calls.",
    "Wow, love paying full price for a product that stops working in a week.",
    "The design is beautiful, shame the battery only lasts an hour.",
    "Customer support replaced my unit within 48 hours, no questions asked.",
    "Screen has noticeable dead pixels right out of the box.",
]


def normalize_label(raw_label: str) -> str:
    label = raw_label.strip().upper()
    if label.startswith("POS"):
        return "positive"
    if label.startswith("NEG"):
        return "negative"
    if label.startswith("NEU"):
        return "negative"
    digits = "".join(ch for ch in label if ch.isdigit())
    if digits:
        return "positive" if int(digits) >= 4 else "negative"
    if label.lower() in EMOTION_POLARITY:
        return EMOTION_POLARITY[label.lower()]
    raise ValueError(f"Unrecognized label format: {raw_label}")


@st.cache_resource(show_spinner=False)
def load_sentiment_pipelines():
    return {
        model_id: pipeline("text-classification", model=model_id)
        for model_id in SENTIMENT_MODEL_IDS
    }


@st.cache_resource(show_spinner=False)
def load_topic_model():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    topic_names = list(TOPIC_DESCRIPTIONS.keys())
    topic_embeddings = embedder.encode(list(TOPIC_DESCRIPTIONS.values()), convert_to_tensor=True)
    return embedder, topic_names, topic_embeddings


def predict_sentiment(text: str, pipelines: dict) -> list[dict]:
    rows = []
    for model_id in SENTIMENT_MODEL_IDS:
        result = pipelines[model_id](text)[0]
        rows.append({
            "model": model_id,
            "raw_label": result["label"],
            "score": result["score"],
            "normalized_label": normalize_label(result["label"]),
        })
    return rows


def predict_topic(text: str, embedder, topic_names, topic_embeddings) -> dict:
    text_embedding = embedder.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(text_embedding, topic_embeddings)[0]
    top2 = similarities.topk(2)
    best_idx = int(top2.indices[0])
    return {
        "label": topic_names[best_idx],
        "similarity": float(top2.values[0]),
        "runner_up": topic_names[int(top2.indices[1])],
        "gap_to_runner_up": float(top2.values[0] - top2.values[1]),
    }


st.title("Electronics Review Analyzer")
st.caption(
    "Independent build of the Pretrained Model Challenge: 3 pretrained sentiment models "
    "(with a generic label normalizer) + embedding-similarity topic classification."
)

example = st.selectbox("Try an example review, or write your own below:", ["(write your own)"] + EXAMPLE_REVIEWS)
default_text = "" if example == "(write your own)" else example
text = st.text_area("Review text", value=default_text, height=100)

if st.button("Analyze", type="primary", disabled=not text.strip()):
    with st.spinner("Loading models (first run downloads them, then it's cached)..."):
        sentiment_pipelines = load_sentiment_pipelines()
        embedder, topic_names, topic_embeddings = load_topic_model()

    st.subheader("Sentiment — 3 models")
    sentiment_rows = predict_sentiment(text, sentiment_pipelines)
    st.dataframe(sentiment_rows, use_container_width=True, hide_index=True)

    votes = [r["normalized_label"] for r in sentiment_rows]
    majority = max(set(votes), key=votes.count)
    st.metric("Majority vote", majority.capitalize())

    st.subheader("Topic — embedding similarity")
    topic_result = predict_topic(text, embedder, topic_names, topic_embeddings)
    col1, col2 = st.columns(2)
    col1.metric("Predicted topic", topic_result["label"])
    col2.metric("Similarity gap to runner-up", f"{topic_result['gap_to_runner_up']:.3f}")
    st.caption(
        f"Runner-up topic: **{topic_result['runner_up']}** "
        f"(a small gap means the review could plausibly belong to either topic)"
    )
else:
    st.info("Enter or pick a review above, then click Analyze.")

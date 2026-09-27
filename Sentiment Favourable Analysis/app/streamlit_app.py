import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import TextPreprocessor
from src.features import TFIDFFeatureExtractor
from src.favorability_score import FavorabilityScorer
from src.model import LogisticRegressionBaseline, DistilBertClassifier

# Set page config for a premium look
st.set_page_config(
    page_title="Sentiment & Favorability Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #9ca3af;
        margin-bottom: 2rem;
    }
    
    .card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
    }
    
    .gauge-bg {
        background-color: #374151;
        border-radius: 10px;
        height: 12px;
        width: 100%;
        margin-top: 8px;
        position: relative;
    }
    
    .gauge-fill-pos {
        background: linear-gradient(90deg, #6b7280 0%, #22c55e 100%);
        border-radius: 10px;
        height: 100%;
    }

    .gauge-fill-neg {
        background: linear-gradient(90deg, #ef4444 0%, #6b7280 100%);
        border-radius: 10px;
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Generate fallback/mock dataset in case of no internet
@st.cache_data
def get_mock_data():
    reviews = [
        ("This movie was absolute trash! The acting was terrible, script was messy, and it felt like a total waste of time.", 0),
        ("Incredible masterpiece. A beautiful combination of outstanding performances, perfect sound design, and directing.", 1),
        ("An average film. It had some entertaining parts, but overall it felt a bit slow and derivative.", 0),
        ("I absolutely loved the visual effects, but the story was somewhat weak and predictable. Worth a watch though.", 1),
        ("Unbearable. I wanted to walk out within the first 30 minutes. Complete rubbish.", 0),
        ("A solid, entertaining thriller. The pacing was quick and the plot kept me guessing until the very end.", 1),
        ("Not the best, not the worst. Just another generic action movie that you will forget tomorrow.", 0),
        ("Wow, simply beautiful. It touched my heart. The ending was so emotional.", 1),
        ("Very disappointing. The trailers hyped it up, but the actual movie was a major letdown.", 0),
        ("Superb acting from the lead roles! Even with a simple plot, the characters made it exceptional.", 1),
        ("Honestly, I didn't care for it. The characters were unlikable and the dialog was cheesy.", 0),
        ("Brilliant! A fun ride from start to finish. Highly recommended for families.", 1),
    ]
    # Replicate to make a larger sample size for trends
    np.random.seed(42)
    extended_reviews = []
    for i in range(100):
        idx = np.random.randint(len(reviews))
        text, label = reviews[idx]
        # Add random noise/variations to review length
        extended_reviews.append({
            "text": text,
            "label": label,
            "review_index": i,
            "timestamp": pd.date_range(start="2026-06-01", periods=100)[i]
        })
    return pd.DataFrame(extended_reviews)

@st.cache_resource
def load_imdb_dataset_or_fallback():
    try:
        from datasets import load_dataset
        # Load a small split (e.g. 1000 reviews for speed) to act as dashboard EDA dataset
        dataset = load_dataset("imdb", split="train[:1000]")
        df = pd.DataFrame({
            "text": dataset["text"],
            "label": dataset["label"],
            "review_index": range(1000),
            "timestamp": pd.date_range(start="2026-06-01", periods=1000)
        })
        return df, "Loaded 1,000 real IMDB reviews."
    except Exception as e:
        return get_mock_data(), f"Offline fallback mode active. Loaded mock reviews. ({str(e)})"

# Load models and extractors
@st.cache_resource
def initialize_pipeline(texts, labels):
    # Preprocessor
    preprocessor = TextPreprocessor()
    
    # Process texts for baseline TF-IDF model
    processed_texts = preprocessor.preprocess_batch(texts)
    
    # Vectorizer
    vectorizer = TFIDFFeatureExtractor(max_features=2500)
    X_tfidf = vectorizer.fit_transform(processed_texts)
    
    # Train Logistic Regression baseline
    lr_model = LogisticRegressionBaseline(C=1.0)
    lr_model.train(X_tfidf, labels)
    
    # Initialize DistilBERT (using a pre-trained SST-2 model)
    try:
        db_model = DistilBertClassifier(model_name="distilbert-base-uncased-finetuned-sst-2-english")
    except Exception as e:
        st.warning(f"Could not load DistilBERT from Hugging Face: {e}. Using baseline only.")
        db_model = None
        
    return preprocessor, vectorizer, lr_model, db_model

# ----------------- APP LAYOUT -----------------

st.markdown("<div class='main-title'>Sentiment Favorability Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A continuous -1.0 (unfavorable) to +1.0 (favorable) scoring system for text reviews</div>", unsafe_allow_html=True)

# Load data
df_data, status_msg = load_imdb_dataset_or_fallback()

# Initialize models
with st.spinner("Initializing models and preprocessors..."):
    preprocessor, vectorizer, lr_model, db_model = initialize_pipeline(
        df_data["text"].tolist(), 
        df_data["label"].tolist()
    )

# Sidebar configurations
st.sidebar.markdown("### Settings & Calibration")
temp_calibration = st.sidebar.slider(
    "Temperature Calibration (T)", 
    min_value=0.2, 
    max_value=3.0, 
    value=1.0, 
    step=0.1,
    help="Higher T pulls scores closer to 0 (neutral). Lower T pushes scores to extremes (-1 or +1)."
)

scorer = FavorabilityScorer(temperature=temp_calibration)

# Display dataset loading status
st.sidebar.caption(status_msg)

# Main input box
st.markdown("### 📝 Analyze Custom Review")
default_review = "The cinematography was breathtaking and the actors gave it their all, but the plot dragged on forever and the climax was incredibly weak."
user_input = st.text_area("Type your review below:", value=default_review, height=100)

if st.button("Calculate Favorability") or user_input:
    # 1. Preprocess
    cleaned_input = preprocessor.preprocess_text(user_input)
    
    col1, col2 = st.columns(2)
    
    # ---------------- Baseline LR Model ----------------
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 📊 Logistic Regression Baseline")
        
        # Extract features and predict
        features = vectorizer.transform([cleaned_input])
        prob_lr = lr_model.predict_proba(features)[0]
        score_lr = scorer.calculate_score(prob_lr)
        label_lr = scorer.get_label(score_lr)
        color_lr = scorer.get_color(score_lr)
        
        st.markdown(f"Sentiment: <b style='color:{color_lr}'>{label_lr}</b>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric-value'>{score_lr:+.3f}</div>", unsafe_allow_html=True)
        
        # Custom progress gauge
        pct = (score_lr + 1) / 2  # normalize -1..1 to 0..1
        if score_lr >= 0:
            st.markdown(f"""
            <div class='gauge-bg'>
                <div class='gauge-fill-pos' style='width: {pct*100}%; margin-left: 50%; width: {(pct-0.5)*100}%;'></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='gauge-bg'>
                <div class='gauge-fill-neg' style='width: {(0.5-pct)*100}%; margin-left: {pct*100}%;'></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.caption(f"Raw Model Probability: {prob_lr:.2%}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Word analysis for LR
        st.markdown("#### Key Word Contributions")
        words = cleaned_input.split()
        if words:
            word_weights = []
            feature_names_list = list(vectorizer.vectorizer.get_feature_names_out())
            for w in set(words):
                if w in feature_names_list:
                    idx = feature_names_list.index(w)
                    weight = lr_model.model.coef_[0][idx]
                    word_weights.append((w, weight))
            
            if word_weights:
                word_weights = sorted(word_weights, key=lambda x: abs(x[1]), reverse=True)[:8]
                words_df = pd.DataFrame(word_weights, columns=["Word", "Weight"])
                
                # Plot horizontal bar chart
                fig, ax = plt.subplots(figsize=(6, 3))
                colors = ["#22c55e" if w > 0 else "#ef4444" for w in words_df["Weight"]]
                sns.barplot(data=words_df, x="Weight", y="Word", palette=colors, ax=ax)
                ax.axvline(x=0, color="#ffffff", linestyle="-", alpha=0.3)
                ax.set_title("Word Impact on Favorability", color="white")
                fig.patch.set_facecolor("none")
                ax.set_facecolor("none")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(colors="white")
                st.pyplot(fig)
            else:
                st.write("No vocabulary words match the TF-IDF training set.")
        else:
            st.write("Please type some words.")
            
    # ---------------- DistilBERT Model ----------------
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("### 🧠 DistilBERT Transformer Model")
        
        if db_model is not None:
            # Predict
            prob_db = db_model.predict_proba([user_input])[0]
            score_db = scorer.calculate_score(prob_db)
            label_db = scorer.get_label(score_db)
            color_db = scorer.get_color(score_db)
            
            st.markdown(f"Sentiment: <b style='color:{color_db}'>{label_db}</b>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{score_db:+.3f}</div>", unsafe_allow_html=True)
            
            # Custom progress gauge
            pct_db = (score_db + 1) / 2
            if score_db >= 0:
                st.markdown(f"""
                <div class='gauge-bg'>
                    <div class='gauge-fill-pos' style='width: {pct_db*100}%; margin-left: 50%; width: {(pct_db-0.5)*100}%;'></div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='gauge-bg'>
                    <div class='gauge-fill-neg' style='width: {(0.5-pct_db)*100}%; margin-left: {pct_db*100}%;'></div>
                </div>
                """, unsafe_allow_html=True)
                
            st.caption(f"Raw Model Probability: {prob_db:.2%}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("DistilBERT model is not loaded. Ensure PyTorch and Transformers are fully set up.")
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------- EXPLORATORY DATA ANALYSIS SECTION -----------------
st.markdown("---")
st.markdown("## 📊 Corpus Analysis (Sample reviews)")

col_eda1, col_eda2 = st.columns([1, 1])

with col_eda1:
    st.markdown("### 📈 Sentiment Favorability Trend")
    # Compute favorability scores on sample data using baseline
    with st.spinner("Processing sample trends..."):
        sample_df = df_data.head(50).copy()
        sample_cleaned = [preprocessor.preprocess_text(t) for t in sample_df["text"]]
        sample_feats = vectorizer.transform(sample_cleaned)
        sample_probs = lr_model.predict_proba(sample_feats)
        sample_df["favorability"] = scorer.calculate_scores(sample_probs)
        
        # Calculate 5-review rolling average
        sample_df["rolling_avg"] = sample_df["favorability"].rolling(window=5, min_periods=1).mean()
        
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(sample_df["review_index"], sample_df["favorability"], marker="o", linestyle="-", alpha=0.4, color="#a855f7", label="Individual Review")
        ax.plot(sample_df["review_index"], sample_df["rolling_avg"], color="#22c55e", linewidth=2.5, label="5-Review Rolling Avg")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title("Favorability Score Trajectory Across Reviews", color="white")
        ax.set_xlabel("Review Index", color="white")
        ax.set_ylabel("Favorability (-1 to +1)", color="white")
        ax.legend(facecolor="#1e1b4b", labelcolor="white")
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors="white")
        st.pyplot(fig)

with col_eda2:
    st.markdown("### ☁️ Word Cloud of Favorable Tokens")
    # Generate word cloud of processed positive reviews
    with st.spinner("Generating word cloud..."):
        pos_reviews = df_data[df_data["label"] == 1]["text"].head(30)
        pos_cleaned = " ".join([preprocessor.preprocess_text(t) for t in pos_reviews])
        
        if pos_cleaned.strip():
            wordcloud = WordCloud(width=600, height=330, background_color="black", 
                                  colormap="viridis", max_words=80).generate(pos_cleaned)
            
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            fig.patch.set_facecolor("none")
            st.pyplot(fig)
        else:
            st.write("No words available to generate cloud.")

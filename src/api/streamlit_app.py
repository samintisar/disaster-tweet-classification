"""Simple Streamlit app for disaster tweet classification."""

import streamlit as st
import requests
import time
from typing import Dict, Any

# Set page config
st.set_page_config(
    page_title="Disaster Tweet Classification",
    page_icon="🚨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# API configuration
API_URL = "http://localhost:8000"

def classify_tweet(text: str, include_features: bool = True, include_keywords: bool = True) -> Dict[str, Any]:
    """Classify a tweet using the API.

    Args:
        text: Tweet text to classify
        include_features: Whether to include feature analysis
        include_keywords: Whether to include keyword analysis

    Returns:
        Classification result dictionary
    """
    try:
        payload = {
            "text": text,
            "include_features": include_features,
            "include_keywords": include_keywords
        }

        response = requests.post(f"{API_URL}/api/classify", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def main():
    """Main Streamlit app."""
    st.title("🚨 Disaster Tweet Classification")
    st.markdown("---")

    # Input section
    st.subheader("Classify a Tweet")

    # Text input
    tweet_text = st.text_area(
        "Enter tweet text:",
        placeholder="Enter a tweet to classify as disaster-related or not...",
        height=100,
        max_chars=280
    )

    # Options
    col1, col2 = st.columns(2)
    with col1:
        include_features = st.checkbox("Include Features", value=True)
    with col2:
        include_keywords = st.checkbox("Include Keywords", value=True)

    # Classify button
    if st.button("Classify Tweet", type="primary", disabled=not tweet_text.strip()):
        if not tweet_text.strip():
            st.warning("Please enter some text to classify.")
            return

        with st.spinner("Classifying tweet..."):
            result = classify_tweet(tweet_text, include_features, include_keywords)

            if result:
                display_result(result, tweet_text)

    # Sample tweets section
    st.markdown("---")
    st.subheader("Sample Tweets")

    sample_tweets = [
        "Major earthquake hits San Francisco, buildings damaged and people injured #earthquake",
        "Just had a great lunch with friends at the new restaurant downtown!",
        "Hurricane warning issued for coastal areas, residents advised to evacuate immediately #hurricane",
        "Beautiful sunset today, perfect weather for a walk in the park",
        "Flood waters rising rapidly in downtown area, emergency services responding #flood"
    ]

    if st.button("Load Sample Tweet"):
        sample_text = st.selectbox("Choose a sample tweet:", sample_tweets)
        st.session_state.sample_text = sample_text

    if 'sample_text' in st.session_state:
        st.text_area("Sample tweet:", value=st.session_state.sample_text, height=80, disabled=True)

        if st.button("Classify Sample", type="secondary"):
            with st.spinner("Classifying sample tweet..."):
                result = classify_tweet(st.session_state.sample_text, include_features, include_keywords)

                if result:
                    display_result(result, st.session_state.sample_text)

    # Health check section
    st.markdown("---")
    st.subheader("System Status")

    if st.button("Check System Health"):
        with st.spinner("Checking system health..."):
            try:
                response = requests.get(f"{API_URL}/api/health")
                response.raise_for_status()
                health_data = response.json()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Status", health_data["status"])
                with col2:
                    st.metric("Model Loaded", "✅" if health_data["model_loaded"] else "❌")
                with col3:
                    st.metric("Service", health_data["service"])

                st.success(f"System is healthy! Last checked: {health_data['timestamp']}")

            except requests.exceptions.RequestException as e:
                st.error(f"Health check failed: {str(e)}")

def display_result(result: Dict[str, Any], original_text: str):
    """Display classification result.

    Args:
        result: Classification result dictionary
        original_text: Original tweet text
    """
    # Determine prediction display
    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction == "disaster":
        prediction_display = "🚨 DISASTER"
        color = "red"
    else:
        prediction_display = "✅ NORMAL"
        color = "green"

    # Main result
    st.markdown("---")
    st.subheader("Classification Result")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### <span style='color: {color}'>{prediction_display}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Confidence", f"{confidence:.2%}")

    # Show original text
    with st.expander("Original Tweet"):
        st.write(f"**Text:** {original_text}")
        st.write(f"**Tweet ID:** {result['tweet_id']}")
        st.write(f"**Processed:** {result['timestamp']}")

    # Show probabilities
    st.markdown("#### Probabilities")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Disaster", f"{result['probabilities']['disaster']:.2%}")
    with col2:
        st.metric("Normal", f"{result['probabilities']['non_disaster']:.2%}")

    # Show features if available
    if "features" in result and result["features"]:
        st.markdown("#### Feature Analysis")
        features = result["features"]

        # Display key features
        if "disaster_keywords" in features and features["disaster_keywords"]:
            st.write(f"**Disaster Keywords:** {', '.join(features['disaster_keywords'])}")

        if "sentiment_score" in features:
            sentiment = features["sentiment_score"]
            sentiment_label = "Negative" if sentiment < -0.1 else "Positive" if sentiment > 0.1 else "Neutral"
            st.write(f"**Sentiment:** {sentiment_label} ({sentiment:.2f})")

        if "url_count" in features:
            st.write(f"**URLs:** {features['url_count']}")

        if "hashtag_count" in features:
            st.write(f"**Hashtags:** {features['hashtag_count']}")

if __name__ == "__main__":
    main()
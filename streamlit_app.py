# NEWS CREDIBILITY CLASSIFIER - Streamlit Version

import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from joblib import load
import os

# Getting API Key from Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")

OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MODEL_PATH = "news_credibility_classifier.joblib"

# FUNCTIONS
def clean_text(text: str) -> str:
    """Clean and preprocess text"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_text_embeddings(text: str, api_key: str) -> list[float]:
    """Get embeddings from OpenAI API"""
    if not text or not text.strip():
        return [0.0] * EMBEDDING_DIMENSIONS

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "input": text,
        "model": EMBEDDING_MODEL,
        "dimensions": EMBEDDING_DIMENSIONS
    }

    try:
        response = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()["data"][0]["embedding"]
        else:
            st.error(f"OpenAI API Error: Status {response.status_code}")
            return [0.0] * EMBEDDING_DIMENSIONS
    except requests.exceptions.RequestException as e:
        st.error(f"Request Failed: {e}")
        return [0.0] * EMBEDDING_DIMENSIONS

@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        model = load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"❌ Model file '{MODEL_PATH}' not found!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def predict_article(headline: str, content: str, api_key: str, model) -> dict:
    """Predict article credibility"""
    if headline.strip() == "" or content.strip() == "":
        return {"error": "Headline and content cannot be empty"}

    # Clean text
    cleaned_headline = clean_text(headline)
    cleaned_content = clean_text(content)

    # Get embeddings
    headline_embedding = get_text_embeddings(cleaned_headline, api_key)
    content_embedding = get_text_embeddings(cleaned_content, api_key)

    # Combine embeddings
    combined_embedding = np.concatenate((headline_embedding, content_embedding))
    input_dimensions = combined_embedding.reshape(1, -1)

    # Predict
    prediction = model.predict(input_dimensions)
    probability = model.predict_proba(input_dimensions)

    label = "Credible" if prediction[0] == 1 else "Not Credible"
    confidence = probability[0][prediction[0]] * 100

    return {
        "label": label,
        "confidence": confidence,
        "prediction": int(prediction[0])
    }

# SAMPLE ARTICLES
sample_articles = {
    "credible": {
        "headline": "Tropical Depression Luis maintains strength; Dolphin weakens into typhoon outside PAR",
        "content": """MANILA, Philippines – Tropical Depression Luis maintained its strength on Sunday morning, August 2, while Dolphin, the tropical cyclone outside the Philippine Area of Responsibility (PAR), was downgraded from a super typhoon to a typhoon.

The Philippine Atmospheric, Geophysical, and Astronomical Services Administration (PAGASA) said in its 11 am bulletin on Sunday that Luis still has maximum sustained winds of 55 kilometers per hour and gustiness of up to 70 km/h.

As of 10 am, the tropical depression was located 400 kilometers east of Infanta, Quezon. It slightly accelerated, heading northwest at 15 km/h after moving at less than 10 km/h.

Luis remains likely to stay over the Philippine Sea, but if its forecast track shifts westward, it could make landfall in Northern Luzon or Central Luzon, or go near these areas.

Luis might also strengthen into a tropical storm on Sunday, but it may weaken back into a tropical depression on Monday, August 3, and into a remnant low by Wednesday, August 5."""
    },
    "not credible": {
        "headline": "BOMBA! VP SARA NAGPALIT NG PASAPORTE, PLANO DAW TUMALON SA IBANG BANSA—EXCLUSIVE SOURCE",
        "content": """MAKATI CITY—Isang nakakagulat na balita ang sumambulat sa social media ngayong gabi matapos kumalat ang ulat na si Vice President Sara Duterte-Carpio ay diumano'y nagpalit ng kanyang pasaporte at nagpaplano nang lumipad patungo sa isang hindi pa tukoy na bansa sa susunod na linggo.

Ayon sa isang "highly placed insider" na malapit sa kampo ng Bise Presidente, ang naturang hakbang ay ginawa umano matapos ang hindi pagkakaunawaan nito kay Pangulong Bongbong Marcos Jr. sa isang closed-door meeting sa Malacañang noong nakaraang Huwebes.

"Totoong-totoo po iyan. May kopya pa nga kami ng bagong passport ni Madam. As in blue ang cover, may tatak ng DFA," sabi ng source na tumangging makilala dahil sa takot na mawalan ng trabaho.

Hindi pa rin mabatid kung saang bansa patungo si VP Sara, pero ayon sa ilang netizens na nag-viral na post, ang Switzerland daw o kaya Canada ang kanyang destinasyon dahil may "malaking bank account" umano ang pamilya Duterte doon.

Samantala, mariing kinondena ng ilang militanteng grupo ang diumano'y pagtakas ni VP Sara, at tinawag itong "pagkakanulo sa mamamayang Pilipino." Wala namang opisyal na pahayag mula sa opisina ng Bise Presidente patungkol sa isyung ito, pero patuloy ang pagkalat ng balita sa Facebook at TikTok na may mahigit 50,000 shares sa loob lamang ng dalawang oras.

Sinabi rin ng isang "security expert" na si "Mr. X" (ayaw magbigay ng buong pangalan), na posibleng ginagamit ng mga nasa paligid ni VP Sara ang "backdoor route" sa NAIA upang maiwasan ang media coverage."""
    }
}

# USER INTERFACE
# TITLE
st.set_page_config(
    page_title="News Credibility Classifier",
    page_icon="📰",
    layout="wide"
)

# TITLE BOX
st.markdown("""
<div style="background-color: #ffffff; padding: 10px; border-radius: 8px; border: 2px solid #e0e0e0; text-align: center;">
    <h1 style="color: black; margin: 0; font-size: 50px;">📰</h1>
    <h1 style="color: black; margin: 0; font-size: 22px; font-weight: 700;">
        News <span style="color: #0066cc;">Credibility</span> Classifier
    </h1>
    <p style="color: #555555; margin-top: 4px; font-size: 14px;">
        Enter a news article to check if it's <u><b>Credible</b> or <b>Not Credible</b></u>
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# INSTRUCTIONS
with st.expander("📖 How to use", expanded=True):
    st.markdown("""
        <h2 style="color:#000000; margin: 0 0 5px 0; font-size: 18px; font-weight: 600;">✍🏻 Enter your news article!</h2>
        <p style="color: #333333; margin: 0 0 5px 0; font-size: 14px; line-height: 2.0;">
            <b>1.</b> Type the <b>Headline</b> in the first box below.<br>
            <b>2.</b> Type the <b>Content</b> in the second box below.<br>
            <b>3.</b> Click <b>"🔍 Predict Credibility"</b> to see the results.
        </p>
        <p style="color: #666666; margin: 10px 0 5px 0; font-size: 13px;">
            💡 <i>Try the sample articles in the sidebar!</i>
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <style>
    /* Fix spacing around divider */
    hr {
        margin-top: 6px !important;
        margin-bottom: 6px !important;
    }
    /* Reduce space between buttons */
    .stButton {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    /* Reduce vertical block spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 4px !important;
    }
    /* Make Predict button blue */
    button[kind="primary"] {
        background-color: #0066cc !important;
        border-color: #0066cc !important;
        color: white !important;
    }
    button[kind="primary"]:hover {
        background-color: #004d99 !important;
        border-color: #004d99 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("📌 Try a Sample Article")
    
    sample_choice = st.selectbox(
        "Select an example:",
        ["--- Select ---", "✅ Credible", "❌ Not Credible"]
    )
    
    if sample_choice == "✅ Credible":
        sample_key = "credible"
    elif sample_choice == "❌ Not Credible":
        sample_key = "not credible"
    else:
        sample_key = None
    
    if sample_key and st.button("📋 Load Sample", use_container_width=True):
        st.session_state["headline"] = sample_articles[sample_key]["headline"]
        st.session_state["content"] = sample_articles[sample_key]["content"]
        st.rerun()

st.divider()
    
predict_button = st.button(
        "🔍 Predict Credibility",
        use_container_width=True,
        type="primary"
    )
    
clear_button = st.button(
        "🗑️ Clear Text",
        use_container_width=True
    )

# ---- MAIN INPUT AREA ----
col1, col2 = st.columns([3, 1])

with col1:
    headline = st.text_area(
        "**Headline**",
        value=st.session_state.get("headline", ""),
        placeholder="Enter article headline...",
        height=80,
        width=2400
    )
    
    content = st.text_area(
        "**Content**",
        value=st.session_state.get("content", ""),
        placeholder="Enter article content...",
        height=250,
        width=2000
    )

# ---- CLEAR BUTTON LOGIC ----
if clear_button:
    st.session_state["headline"] = ""
    st.session_state["content"] = ""
    st.rerun()

# ---- LOAD MODEL ----
model = load_model()

if model is None:
    st.stop()

# ---- PREDICTION LOGIC ----
if predict_button:
    # Check API key
    if not OPENAI_API_KEY:
        st.error("❌ OpenAI API Key not found! Please add it to Streamlit Secrets.")
        st.info("Go to your app settings → Secrets → Add `OPENAI_API_KEY`")
        st.stop()
    
    # Validate inputs
    if not headline.strip() and not content.strip():
        st.warning("⚠️ Please enter both a headline and content.")
        st.stop()
    
    # Show spinner while processing
    with st.spinner("🤔 Analyzing article..."):
        try:
            result = predict_article(headline, content, OPENAI_API_KEY, model)
            
            if "error" in result:
                st.error(f"⚠️ Error: {result['error']}")
            else:
                label = result["label"]
                confidence = result["confidence"]
                
                # Display result with styling
                if label == "Credible":
                    st.markdown(f"""
                    <div style="background-color: #ccffcc; padding: 20px; border-radius: 10px; border: 2px solid #2c2d2d; text-align: center;">
                        <h2 style="color: #306844; margin: 0;">✅ {label}</h2>
                        <p style="color: #306844; margin-top: 10px; font-size: 18px;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #EE4B2B; padding: 20px; border-radius: 10px; border: 2px solid #2c2d2d; text-align: center;">
                        <h2 style="color: #ffffff; margin: 0;">❌ {label}</h2>
                        <p style="color: #ffffff; margin-top: 10px; font-size: 18px;">Confidence: {confidence:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# ---- FOOTER ----
st.divider()
st.markdown("""
<p style="text-align: center; color: #666666; font-size: 12px;">
         The News Credibility Classifier can make errors. Always double-check with your own careful reading and judgment.
</p>
""", unsafe_allow_html=True)

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

Samantala, mariing kinondena ng ilang militanteng grupo ang diumano'y pagtakas ni VP Sara, at tinawag itong "pagkakanulo sa mamamayang Pilipino." Wala namang opisyal na pahayag mula sa opisina ng Bise Presidente patungkol sa isyung ito, pero patuloy ang pagkalat ng balita sa Facebook at TikTok na mahigit 50,000 shares sa loob lamang ng dalawang oras.

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

# ---- LOAD MODEL ----
model = load_model()

if model is None:
    st.stop()

# ---- SIDEBAR ----
with st.sidebar:
    st.markdown("""
    <style>
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
    /* Fix divider spacing */
    hr {
        margin-top: 8px !important;
        margin-bottom: 8px !important;
    }
    /* Reduce button spacing */
    .stButton {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    /* Reduce vertical block spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 4px !important;
    }
    .bottom-caption {
        font-size: 10px;
        color: #888888;
        text-align: center;
        padding: 10px 0;
        margin-top: 20px;
        border-top: 1px solid #dddddd;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---- TITLE IN SIDEBAR ----
    st.markdown("""
    <div style="text-align: center; margin-bottom: 15px;">
        <h1 style="color: black; margin: 0; font-size: 40px;">📰</h1>
        <h1 style="color: black; margin: 2px 0 0 0; font-size: 20px; font-weight: 700;">
            News <span style="color: #0066cc;">Credibility</span> Classifier
        </h1>
        <p style="color: #555555; margin-top: 4px; font-size: 12px;">
            Check if a news article is <u><b>Credible</b> or <b>Not Credible</b></u>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    predict_button = st.button(
        "🔍 Predict Credibility",
        use_container_width=True,
        type="primary"
    )
    
    clear_button = st.button(
        "🗑️ Clear Text",
        use_container_width=True
    )
    
    # ---- CREATE A PLACEHOLDER FOR RESULTS IN SIDEBAR ----
    results_placeholder = st.empty()
    st.session_state.results_placeholder = results_placeholder
    
    # Push caption to bottom
    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True)
    
    # Caption at bottom with smaller font
    st.markdown("""
    <div class="bottom-caption">
        The classifier can make errors. Always double-check with careful reading and judgment.
    </div>
    """, unsafe_allow_html=True)

# ---- MAIN AREA ----
# INSTRUCTIONS (Always visible, not a dropdown)
st.markdown("""
<div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px; border: 1px solid #d0d0d0; margin-bottom: 15px;">
    <h2 style="color:#000000; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">✍🏻 Enter your news article!</h2>
    <p style="color: #333333; margin: 0; font-size: 14px; line-height: 2.0;">
        <b>1.</b> Type the <b>Headline</b> in the first box below.<br>
        <b>2.</b> Type the <b>Content</b> in the second box below.<br>
        <b>3.</b> Click <b>"🔍 Predict Credibility"</b> to see the results in the sidebar.
    </p>
</div>
""", unsafe_allow_html=True)

# ---- SAMPLE ARTICLE DROPDOWN (Gray box, single container) ----
st.markdown("""
<div style="background-color: #f0f0f0; padding: 15px 15px 5px 15px; border-radius: 8px; border: 1px solid #d0d0d0; margin-bottom: 15px;">
    <p style="color: #333333; margin: 0 0 5px 0; font-size: 14px; font-weight: 600;">📌 Try a Sample Article</p>
    <p style="color: #888888; margin: 0 0 10px 0; font-size: 12px;">Select a sample to automatically load it into the fields below</p>
</div>
""", unsafe_allow_html=True)

sample_choice = st.selectbox(
    "Select an example:",
    ["--- Select ---", "✅ Credible", "❌ Not Credible"],
    key="sample_choice"
)

# Initialize clear flag if not exists
if "clear_pressed" not in st.session_state:
    st.session_state.clear_pressed = False

# Auto-load or clear based on selection (FIXED: Always load when selection changes)
if sample_choice == "✅ Credible":
    st.session_state["headline"] = sample_articles["credible"]["headline"]
    st.session_state["content"] = sample_articles["credible"]["content"]
elif sample_choice == "❌ Not Credible":
    st.session_state["headline"] = sample_articles["not credible"]["headline"]
    st.session_state["content"] = sample_articles["not credible"]["content"]
elif sample_choice == "--- Select ---":
    # Only clear if clear_pressed is not True
    if not st.session_state.clear_pressed:
        st.session_state["headline"] = ""
        st.session_state["content"] = ""
else:
    # Reset the flag after clearing
    st.session_state.clear_pressed = False

# ---- MAIN INPUT AREA ----
headline = st.text_area(
    "**Headline**",
    value=st.session_state.get("headline", ""),
    placeholder="Enter article headline...",
    height=80,
    key="headline_input"
)

content = st.text_area(
    "**Content**",
    value=st.session_state.get("content", ""),
    placeholder="Enter article content...",
    height=250,
    key="content_input"
)

# ---- CLEAR BUTTON LOGIC ----
if clear_button:
    st.session_state["headline"] = ""
    st.session_state["content"] = ""
    st.session_state.clear_pressed = True
    # Also reset the dropdown to "--- Select ---"
    st.session_state.sample_choice = "--- Select ---"
    # Clear the results
    results_placeholder.empty()
    st.rerun()

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
    with st.spinner("Analyzing article..."):
        try:
            result = predict_article(headline, content, OPENAI_API_KEY, model)
            
            if "error" in result:
                st.error(f"⚠️ Error: {result['error']}")
            else:
                label = result["label"]
                confidence = result["confidence"]
                
                # Clear any previous results first
                results_placeholder.empty()
                
                # Display result in the sidebar placeholder
                with results_placeholder.container():
                    if label == "Credible":
                        st.markdown(f"""
                        <div style="background-color: #ccffcc; padding: 15px; border-radius: 10px; border: 2px solid #2c2d2d; text-align: center;">
                            <h2 style="color: #306844; margin: 0; font-size: 20px;">✅ {label}</h2>
                            <p style="color: #306844; margin-top: 8px; font-size: 16px;">Confidence: {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #EE4B2B; padding: 15px; border-radius: 10px; border: 2px solid #2c2d2d; text-align: center;">
                            <h2 style="color: #ffffff; margin: 0; font-size: 20px;">❌ {label}</h2>
                            <p style="color: #ffffff; margin-top: 8px; font-size: 16px;">Confidence: {confidence:.2f}%</p>
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

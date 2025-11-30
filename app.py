import streamlit as st
import requests

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(page_title="Khmer News Summarizer", layout="wide")

st.sidebar.title("Backend Settings")

# Your FastAPI / ngrok URL from Kaggle logs:
# 🚀 API RUNNING AT: NgrokTunnel: "https://a31a00410145.ngrok-free.app" -> "http://localhost:8000"
api_base = st.sidebar.text_input(
    "FastAPI / ngrok URL",
    value="https://a31a00410145.ngrok-free.app",
    help="Use the URL printed by the Kaggle notebook (without /summarize at the end).",
)

st.sidebar.markdown("---")
st.sidebar.write(
    "1. Run the Kaggle notebook (hosting-model)\n"
    "2. Copy the printed ngrok URL (already prefilled here)\n"
    "3. Paste text and click *Summarize*"
)

# ==============================
# 🌐 UI
# ==============================
st.title("📰 Khmer News Summarizer")
st.write("Paste a Khmer news article or long text below and get a concise summary in Khmer.")

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area(
        "Input Text (Khmer)",
        height=320,
        placeholder="Paste your Khmer article or paragraph here...",
    )

with col2:
    st.markdown("### Options")
    max_tokens = st.slider("Max summary tokens (approx.)", 64, 512, 256, step=32)
    temperature = st.slider("Creativity (temperature)", 0.1, 1.0, 0.5, step=0.1)

summarize_clicked = st.button("✨ Summarize")

# ==============================
# 🧠 Call Backend
# ==============================
if summarize_clicked:
    if not api_base:
        st.error("Please provide your FastAPI ngrok URL in the sidebar.")
    elif not input_text.strip():
        st.warning("Please paste some text to summarize.")
    else:
        endpoint = api_base.rstrip("/") + "/summarize"

        payload = {
            "text": input_text,
            "max_tokens": max_tokens,
            "temperature": float(temperature),
        }

        st.info(f"Sending request to: `{endpoint}`")
        with st.spinner("Generating summary with your fine-tuned Khmer model..."):
            try:
                resp = requests.post(endpoint, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                summary = data.get("summary", "").strip()

                if not summary:
                    st.error("Backend returned an empty summary.")
                else:
                    st.subheader("📌 Summary (Khmer)")
                    st.write(summary)

            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {e}")
                if getattr(e, "response", None) is not None:
                    st.code(e.response.text, language="json")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

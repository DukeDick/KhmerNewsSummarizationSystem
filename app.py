import streamlit as st
import requests
import google.generativeai as genai  # pip install google-generativeai

# ==============================
# 🔧 STREAMLIT CONFIG
# ==============================
st.set_page_config(page_title="Khmer News Summarizer", layout="wide")

# Keep summary in session so we can use it for Q&A
if "summary" not in st.session_state:
    st.session_state.summary = ""

# ==============================
# 🔧 SIDEBAR CONFIG
# ==============================
st.sidebar.title("Backend Settings")

# 🔁 Your FastAPI / ngrok URL from Kaggle logs:
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

st.sidebar.markdown("---")
st.sidebar.subheader("Gemini Settings")

# 👉 Gemini API key (from Google AI Studio)
gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    help="Get this from Google AI Studio and paste it here.",
)

# Allow changing model if you want
gemini_model_name = st.sidebar.text_input(
    "Gemini Model Name",
    value="gemini-1.5-flash",
    help="e.g. gemini-1.5-flash, gemini-1.5-pro, etc.",
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
# 🧠 CALL BACKEND (SUMMARIZER)
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
                    st.session_state.summary = summary  # 🔥 save for Gemini Q&A
                    st.subheader("📌 Summary (Khmer)")
                    st.write(summary)

            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {e}")
                if getattr(e, "response", None) is not None:
                    st.code(e.response.text, language="json")
            except Exception as e:
                st.error(f"Unexpected error: {e}")

# ==============================
# 🤖 GEMINI Q&A ABOUT SUMMARY
# ==============================
def ask_gemini_about_summary(
    api_key: str,
    model_name: str,
    summary: str,
    question: str,
) -> str:
    """
    Use Gemini to answer a question based on the given summary.
    """
    genai.configure(api_key=api_key)
    model = google_model = genai.GenerativeModel(model_name)

    prompt = f"""
You are a helpful assistant answering questions about the following Khmer news summary.

SUMMARY:
{summary}

USER QUESTION:
{question}

Please answer **in Khmer** and keep the answer short, clear, and directly related to the summary above.
If the summary doesn't contain enough information to answer, say that you don't know based on this summary.
"""

    response = model.generate_content(prompt)
    return response.text.strip() if response and response.text else "មិនអាចឆ្លើយបានពីសង្ខេបនេះទេ។"


# Only show Q&A section if we already have a summary
if st.session_state.summary:
    st.markdown("---")
    st.markdown("### 💬 Ask about this summary (Gemini-powered)")

    st.info(
        "You can now ask questions about the summary above. "
        "Gemini will answer based **only** on that summary."
    )

    question = st.text_input(
        "Your question (in Khmer or English)",
        placeholder="ឧ. តើអត្ថបទនេះពាក់ព័ន្ធអ្វីជាចម្បង? / What is the main issue in this news?",
        key="qa_question",
    )

    ask_clicked = st.button("💬 Ask Gemini about this summary")

    if ask_clicked:
        if not gemini_api_key:
            st.error("Please enter your Gemini API key in the sidebar first.")
        elif not question.strip():
            st.warning("Please type a question first.")
        else:
            with st.spinner("Asking Gemini about your summary..."):
                try:
                    answer = ask_gemini_about_summary(
                        api_key=gemini_api_key,
                        model_name=gemini_model_name,
                        summary=st.session_state.summary,
                        question=question,
                    )
                    st.subheader("🧠 Gemini's Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")

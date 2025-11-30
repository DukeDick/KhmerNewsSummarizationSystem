import os
import requests
import streamlit as st
from bs4 import BeautifulSoup  # pip install beautifulsoup4
import google.generativeai as genai  # pip install google-generativeai


# ==============================
# ⚙️ PAGE & SESSION SETUP
# ==============================
st.set_page_config(page_title="Khmer News Summarizer", layout="wide")

if "summary" not in st.session_state:
    st.session_state.summary = ""
if "input_text" not in st.session_state:
    st.session_state.input_text = ""


# ==============================
# 🔧 SIDEBAR CONFIG
# ==============================
st.sidebar.title("Backend Settings")

# Your FastAPI / ngrok URL from Kaggle logs
api_base = st.sidebar.text_input(
    "FastAPI / ngrok URL",
    value="https://a31a00410145.ngrok-free.app",  # change if needed
    help="Use the URL printed by the Kaggle notebook (without /summarize at the end).",
)

st.sidebar.markdown("---")
st.sidebar.write(
    "1. Run the Kaggle notebook (hosting-model)\n"
    "2. Copy the printed ngrok URL\n"
    "3. Paste here and click *Summarize*"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Gemini Settings")

# Prefer GEMINI_API_KEY from Streamlit secrets, allow override via sidebar
default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")

gemini_api_key = st.sidebar.text_input(
    "Gemini API Key",
    type="password",
    value=default_gemini_key,
    help="Best: set GEMINI_API_KEY in Streamlit Secrets, or paste it here.",
)

gemini_model_name = st.sidebar.text_input(
    "Gemini Model Name",
    value="gemini-2.0-flash",  # or "gemini-flash-latest"
    help="For example: gemini-2.0-flash, gemini-flash-latest, gemini-2.0-pro",
)


# ==============================
# 🔎 HELPERS
# ==============================
def extract_main_text_from_url(url: str) -> str:
    """
    Fetches the URL and tries to extract the main article text.
    Adds browser-like headers to avoid simple 403 blocks.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9,km;q=0.8",
        "Connection": "keep-alive",
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Error fetching URL: {e}")
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]

    # Filter out super-short bits and join
    text = "\n".join(p for p in paragraphs if len(p) > 40)

    return text


def ask_gemini_any_context(api_key: str, model_name: str, context: str, question: str) -> str:
    """
    Use Gemini to answer a question based on arbitrary context
    (summary, original, or both).
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = f"""
You are a helpful assistant answering questions using the context below.

CONTEXT:
{context}

USER QUESTION:
{question}

Rules:
- Answer in Khmer.
- Base your answer ONLY on the context provided.
- If the context doesn't contain enough information, say that you don't know
  based on this context.
"""

    response = model.generate_content(prompt)
    if not response or not getattr(response, "text", ""):
        return "មិនអាចឆ្លើយបានពីបរិបទនេះទេ។"

    return response.text.strip()


# ==============================
# 🌐 MAIN UI
# ==============================
st.title("📰 Khmer News Summarizer")
st.write(
    "Paste a Khmer news article, or a link to it, and get a concise Khmer summary. "
    "Then ask questions about that summary using Gemini, with flexible context options."
)

# ------ URL fetch section ------
st.markdown("### 🔗 Option 1: Paste a news article URL")

url = st.text_input(
    "Article URL (optional)",
    placeholder="https://www.example.com/news/article...",
)

fetch_clicked = st.button("📥 Fetch text from URL")

if fetch_clicked:
    if not url.strip():
        st.warning("Please paste a URL first.")
    else:
        with st.spinner("Fetching and extracting article text from URL..."):
            article_text = extract_main_text_from_url(url)
            if not article_text:
                st.error("Could not extract text from this URL.")
                st.info(
                    "This website may be blocking automated access. "
                    "Please open the article in your browser and paste the text manually."
                )
            else:
                st.success("Extracted article text from URL.")
                st.session_state.input_text = article_text
                # Rerun to show text in textarea
                st.rerun()


st.markdown("### ✍️ Option 2: Paste article text manually")

col1, col2 = st.columns([2, 1])

with col1:
    input_text = st.text_area(
        "Input Text (Khmer)",
        height=320,
        placeholder="Paste your Khmer article or paragraph here...",
        value=st.session_state.input_text,
    )

with col2:
    st.markdown("#### Options")
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
        st.warning("Please paste some text or fetch from URL first.")
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
                    st.session_state.summary = summary
                    st.session_state.input_text = input_text  # keep latest original
                    st.subheader("📌 Summary (Khmer)")
                    st.write(summary)

            except requests.exceptions.RequestException as e:
                st.error(f"Request error: {e}")
                if getattr(e, "response", None) is not None:
                    try:
                        st.code(e.response.text, language="json")
                    except Exception:
                        st.code(str(e.response.text))
            except Exception as e:
                st.error(f"Unexpected error: {e}")


# ==============================
# 💬 GEMINI Q&A ABOUT SUMMARY / ORIGINAL
# ==============================
if st.session_state.summary:
    st.markdown("---")
    st.markdown("### 💬 Ask Gemini about this content")

    # Context mode selector
    context_mode = st.radio(
        "Choose context for Gemini:",
        [
            "Summary Only",
            "Summary + Original Article",
            "Original Article Only",
        ],
    )

    if context_mode == "Summary Only":
        st.info("Gemini will answer based only on the summary above.")
    elif context_mode == "Summary + Original Article":
        st.info("Gemini will use both the summary and original article for best accuracy.")
    else:  # Original Article Only
        st.info("Gemini will answer directly from the original article text.")

    question = st.text_input(
        "Your question (in Khmer or English)",
        placeholder="ឧ. តើអត្ថបទនេះពាក់ព័ន្ធអ្វីជាចម្បង? / What is the main issue in this news?",
        key="qa_question",
    )

    ask_clicked = st.button("💬 Ask Gemini")

    if ask_clicked:
        if not gemini_api_key:
            st.error("Please set your Gemini API key (Streamlit Secrets or sidebar).")
        elif not question.strip():
            st.warning("Please type a question first.")
        else:
            # Build context based on selected mode
            if context_mode == "Summary Only":
                context_text = f"SUMMARY:\n{st.session_state.summary}"

            elif context_mode == "Summary + Original Article":
                if not st.session_state.input_text.strip():
                    st.warning(
                        "No original article text available. "
                        "Using summary only instead."
                    )
                    context_text = f"SUMMARY:\n{st.session_state.summary}"
                else:
                    context_text = (
                        f"SUMMARY:\n{st.session_state.summary}\n\n"
                        f"ORIGINAL ARTICLE:\n{st.session_state.input_text}"
                    )

            else:  # Original Article Only
                if not st.session_state.input_text.strip():
                    st.warning(
                        "No original article text available. "
                        "Using summary only instead."
                    )
                    context_text = f"SUMMARY:\n{st.session_state.summary}"
                else:
                    context_text = f"ORIGINAL ARTICLE:\n{st.session_state.input_text}"

            with st.spinner("Asking Gemini about the selected context..."):
                try:
                    answer = ask_gemini_any_context(
                        api_key=gemini_api_key,
                        model_name=gemini_model_name,
                        context=context_text,
                        question=question,
                    )
                    st.subheader("🧠 Gemini's Answer")
                    st.write(answer)
                except Exception as e:
                    st.error(f"Error calling Gemini API: {e}")

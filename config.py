import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
if not DEEPSEEK_API_KEY:
    try:
        import streamlit as st
        DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", "")
    except Exception:
        pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# Gemini via OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL = "gemini-2.0-flash"

# Primary model selection: prefer Gemini if key available
DEFAULT_MODEL = GEMINI_MODEL if GEMINI_API_KEY else DEEPSEEK_MODEL

LENGTH_CHARS = {
    "短篇（約 800 字）":  800,
    "中篇（約 2000 字）": 2000,
    "長篇（約 4000 字）": 4000,
}

import streamlit as st
from langchain_groq import ChatGroq


def get_chatgroq_model():
    """Load Groq model correctly using Streamlit secrets."""

    # 1️⃣ Read API key + model from streamlit secrets
    api_key = st.secrets.get("GROQ_API_KEY", None)
    model_name = st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        st.error("❌ GROQ_API_KEY not found in secrets.toml")
        return None

    try:
        groq_model = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return groq_model

    except Exception as e:
        st.error(f"❌ Failed to initialize Groq model: {e}")
        return None


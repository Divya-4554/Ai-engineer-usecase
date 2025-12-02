import streamlit as st
from langchain_groq import ChatGroq

def get_chatgroq_model():
    """Return a Groq chat model using Streamlit secrets."""
    
    api_key = st.secrets.get("GROQ_API_KEY")
    model_name = st.secrets.get("GROQ_MODEL", "mixtral-8x7b-32768")

    if not api_key:
        raise RuntimeError("❌ No GROQ_API_KEY found in Streamlit secrets.")

    try:
        groq_model = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


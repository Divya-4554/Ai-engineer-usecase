import os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def get_chatgroq_model():
    """Return a Groq Chat Model if API key is available, else None."""
    groq_key = os.getenv("GROQ_API_KEY")

    if not groq_key:
        return None  # Groq not available

    try:
        return ChatGroq(
            api_key=groq_key,
            model="llama3-70b-8192",   # recommended stable Groq model
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")


def get_openai_model():
    """Return an OpenAI Chat Model if API key is available, else None."""
    openai_key = os.getenv("OPENAI_API_KEY")

    if not openai_key:
        return None

    try:
        return ChatOpenAI(
            api_key=openai_key,
            model="gpt-4o-mini",   # good fast+cheap OpenAI model
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAI model: {str(e)}")


def get_llm():
    """
    Automatically return the best available model:
    1. Groq (if GROQ_API_KEY is set)
    2. OpenAI (if OPENAI_API_KEY is set)
    3. Error if none exist
    """
    groq_model = get_chatgroq_model()
    if groq_model:
        return groq_model

    openai_model = get_openai_model()
    if openai_model:
        return openai_model

    raise RuntimeError(
        "No LLM provider API keys found. "
        "Set GROQ_API_KEY or OPENAI_API_KEY in your environment."
    )

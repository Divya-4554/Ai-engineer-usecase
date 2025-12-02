import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from models.llm import get_chatgroq_model

def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"


def instructions_page():
    st.title("Instructions")
    st.markdown("""
    ## 🔧 Add API Keys
    Create a file `.streamlit/secrets.toml`

    ```toml
    GROQ_API_KEY="your_key_here"
    GROQ_MODEL="llama-3.1-8b-instant"
    ```

    Streamlit will load this automatically.
    """)


def chat_page():
    st.title("🤖 AI ChatBot")

    system_prompt = ""

    chat_model = get_chatgroq_model()

    if not chat_model:
        st.error("❌ No API key found. Please add keys in `.streamlit/secrets.toml`.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Say something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def main():
    st.set_page_config(page_title="AI ChatBot", page_icon="🤖")

    with st.sidebar:
        page = st.radio("Navigation", ["Chat", "Instructions"])

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    if page == "Chat":
        chat_page()
    else:
        instructions_page()


if __name__ == "__main__":
    main()

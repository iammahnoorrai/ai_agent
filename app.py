import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIStatusError

load_dotenv()

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")
st.title("🤖 My AI Chatbot")

base_url = os.getenv("AZURE_OPENAI_BASE_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL")

if not all([base_url, api_key, model]):
    st.error("Check your .env file. One or more settings are missing.")
    st.stop()

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Type your message...")

if prompt:
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = client.responses.create(
                    model=model,
                    input=st.session_state.messages,
                )
                answer = response.output_text

            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except APIConnectionError:
            st.error("Connection failed. Check your internet and endpoint.")

        except APIStatusError as error:
            st.error(f"API error {error.status_code}: {error.message}")

        except Exception as error:
            st.error(f"Unexpected error: {error}")

with st.sidebar:
    st.subheader("Chat controls")

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

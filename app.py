
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="NOVA AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Project paths, environment, and CSS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = PROJECT_DIR / "data" / "vector_store"
CSS_PATH = PROJECT_DIR / "style.css"
ENV_PATH = PROJECT_DIR / ".env"

load_dotenv(ENV_PATH)


def load_css(file_path: Path) -> None:
    """Load the external CSS stylesheet."""

    if not file_path.is_file():
        st.error(f"Stylesheet not found: {file_path}")
        st.stop()

    css_content = file_path.read_text(encoding="utf-8")

    st.markdown(
        f"<style>\n{css_content}\n</style>",
        unsafe_allow_html=True,
    )


load_css(CSS_PATH)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# Azure OpenAI configuration
# --------------------------------------------------

base_url = os.getenv("AZURE_OPENAI_BASE_URL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
model = os.getenv("AZURE_OPENAI_MODEL")

if not all([base_url, api_key, model]):
    st.error(
        "Azure OpenAI configuration is incomplete. "
        "Check AZURE_OPENAI_BASE_URL, AZURE_OPENAI_API_KEY, "
        "and AZURE_OPENAI_MODEL in your .env file."
    )
    st.stop()

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)


# --------------------------------------------------
# RAG: Load the FAISS vector store
# --------------------------------------------------

@st.cache_resource
def load_vector_store():
    """Load and cache the local FAISS index."""

    if not VECTOR_STORE_DIR.is_dir():
        return None

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_rag_context(query: str, top_k: int = 2) -> str:
    """Retrieve relevant weather-documentation chunks."""

    try:
        vector_db = load_vector_store()

        if vector_db is None:
            return ""

        results = vector_db.similarity_search(
            query,
            k=top_k,
        )

        if not results:
            return ""

        evidence = []

        for index, doc in enumerate(results, start=1):
            evidence.append(
                f"Evidence {index}:\n{doc.page_content}"
            )

        return (
            "\n\n--- RETRIEVED WEATHER EVIDENCE ---\n"
            + "\n\n".join(evidence)
            + "\n--- END OF RETRIEVED EVIDENCE ---\n"
        )

    except Exception as error:
        print(f"RAG retrieval error: {error}")
        return ""


# --------------------------------------------------
# System instruction
# --------------------------------------------------

SYSTEM_INSTRUCTION = """
You are NOVA AI, a weather forecasting assistant
developed by Mahnoor Rai.

Identity:
- Identify yourself as NOVA AI when asked.
- Explain that you were developed by Mahnoor Rai when relevant.

Weather response rules:
- Use the retrieved weather evidence when relevant.
- Answer questions using the supplied evidence accurately.
- Distinguish current observations, forecasts, and historical data.
- Do not invent weather readings, forecast values, or source claims.
- Retrieved documents may explain weather concepts but may not
  contain live weather observations or location-specific forecasts.
- If the evidence is insufficient, explain what information is missing.
- Do not claim to have accessed live weather data unless a live
  weather data source has actually supplied it.
- Give clear, useful answers and explain relevant evidence.
"""


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">
                NOVA<span class="sidebar-brand-accent">.</span>AI
            </div>
            <div class="sidebar-caption">
                YOUR INTELLIGENT ASSISTANT
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "＋  New chat",
        type="primary",
        use_container_width=True,
        key="new_chat_button",
    ):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        '<div class="section-label">WORKSPACE</div>',
        unsafe_allow_html=True,
    )

    st.markdown("**Current conversation**")

    message_count = len(st.session_state.messages)

    st.caption(
        f"{message_count} "
        f"{'message' if message_count == 1 else 'messages'} "
        "in this conversation"
    )

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
        key="clear_chat_button",
    ):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# Main header
# --------------------------------------------------

st.markdown(
    """
    <div class="topbar">
        <div>
            <div class="brand">
                NOVA<span class="brand-accent">.</span>AI
            </div>
            <div class="brand-caption">
                INTELLIGENT CONVERSATIONS
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["content"]:
            st.markdown(message["content"])

        for image_data in message.get("images", []):
            st.image(
                image_data["bytes"],
                caption=image_data["name"],
                use_container_width=True,
            )


# --------------------------------------------------
# Single chat input: text and image uploads
# --------------------------------------------------

chat_submission = st.chat_input(
    "Ask NOVA AI about the weather...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
)


if chat_submission is not None:

    prompt = (chat_submission.text or "").strip()
    uploaded_images = chat_submission.files or []

    if prompt or uploaded_images:

        # Save image uploads for local display.
        image_records = []

        for uploaded_image in uploaded_images:
            image_records.append(
                {
                    "name": uploaded_image.name,
                    "bytes": uploaded_image.getvalue(),
                }
            )

        # Prepare the visible user message.
        display_text = prompt

        if image_records:
            image_names = ", ".join(
                image["name"] for image in image_records
            )

            attachment_text = f"Attached images: {image_names}"

            display_text = (
                f"{prompt}\n\n{attachment_text}"
                if prompt
                else attachment_text
            )

        if not display_text:
            display_text = "Message received."

        # --------------------------------------------------
        # PATH 1: Image uploaded
        # Image analysis is not enabled yet.
        # --------------------------------------------------

        if image_records:

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": display_text,
                    "images": image_records,
                }
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": (
                        "Your image has been uploaded successfully. "
                        "Image analysis is not enabled yet. "
                        "You can still ask NOVA AI text-based "
                        "weather questions."
                    ),
                }
            )

            st.rerun()

        # --------------------------------------------------
        # PATH 2: Text-only message with RAG
        # --------------------------------------------------

        elif prompt:

            # Retrieve relevant weather documentation.
            with st.spinner("Retrieving weather evidence..."):
                retrieved_context = get_rag_context(
                    prompt,
                    top_k=2,
                )

            # Combine system rules with retrieved evidence.
            full_system_prompt = (
                SYSTEM_INSTRUCTION
                + "\n\n"
                + (
                    "Use the following retrieved context only "
                    "when relevant to the user's question. "
                    "It may not contain live forecast data.\n"
                    + retrieved_context
                    if retrieved_context
                    else (
                        "No relevant weather documentation was "
                        "retrieved. Do not invent supporting evidence."
                    )
                )
            )

            # Build conversation history before adding this message.
            api_messages = [
                {
                    "role": message["role"],
                    "content": message["content"],
                }
                for message in st.session_state.messages
                if message["role"] in ("user", "assistant")
            ]

            api_messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            # Save the user's message.
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            # Generate the assistant response.
            with st.chat_message("assistant"):

                try:
                    with st.spinner("NOVA is thinking..."):

                        response = client.responses.create(
                            model=model,
                            instructions=full_system_prompt,
                            input=api_messages,
                        )

                        answer = response.output_text

                    if not answer:
                        answer = (
                            "I couldn't generate a response. "
                            "Please try asking again."
                        )

                    st.markdown(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                        }
                    )

                except APIConnectionError:
                    st.error(
                        "Connection failed. Check your internet "
                        "connection and Azure OpenAI endpoint."
                    )

                except APIStatusError as error:
                    st.error(
                        f"Azure OpenAI API error "
                        f"{error.status_code}: {error.message}"
                    )

                except Exception as error:
                    st.error(f"Unexpected error: {error}")

            st.rerun()


import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, OpenAI


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
        "Check your .env file."
    )
    st.stop()

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)


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
# System instruction
# --------------------------------------------------

# --------------------------------------------------
# System instruction — Weather Forecasting Assistant
# --------------------------------------------------


system_instruction = {
    "role": "system",
    "content": (
        "You are NOVA AI, a specialized weather forecasting "
        "assistant developed by Mahnoor Rai. "

        "IDENTITY AND INTRODUCTION: "
        "You must identify yourself as NOVA AI, a specialized "
        "weather forecasting and meteorology assistant developed "
        "by Mahnoor Rai. "
        "Whenever a user asks who you are, asks you to introduce "
        "yourself, asks you to tell them about yourself, or asks "
        "who developed or created you, clearly mention both your "
        "specialization in weather forecasting and your developer, "
        "Mahnoor Rai. "

        "Example introduction: "
        "'I'm NOVA AI, a specialized weather forecasting assistant "
        "developed by Mahnoor Rai. I help users understand weather "
        "forecasts, temperature, rainfall, humidity, wind, storms, "
        "and other meteorological conditions.' "

        "YOUR DOMAIN: "
        "Answer exclusively questions related to weather forecasting, "
        "meteorology, temperature, rainfall, humidity, wind, storms, "
        "atmospheric conditions, weather patterns, and weather safety. "

        "WEATHER ANSWERS: "
        "Use your existing knowledge to explain weather concepts "
        "and meteorological principles. "
        "Do not invent real-time observations, current forecasts, "
        "measurements, or official weather alerts. "
        "Do not claim to have accessed live weather data unless "
        "a live weather source has actually been used. "
        "For general weather questions, answer directly using "
        "your existing knowledge instead of refusing because "
        "you lack current information. "

        "OUT-OF-DOMAIN QUESTIONS: "
        "Politely decline questions unrelated to weather. "
        "Use a response such as: "
        "'I'm sorry, I'm specialized in weather forecasting "
        "and can only assist with weather-related questions. "
        "How can I help you with a weather-related task?' "

        "RESPONSE STYLE: "
        "Be professional, clear, concise, and helpful. "
        "Avoid repetitive introductions. "
        "When the user asks about your identity, follow the "
        "identity rules above before redirecting to weather topics."
    ),
}



# --------------------------------------------------
# Display conversation history
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        for image_data in message.get("images", []):
            st.image(
                image_data["bytes"],
                caption=image_data["name"],
                use_container_width=True,
            )


# --------------------------------------------------
# Chat input with image upload
# --------------------------------------------------

chat_submission = st.chat_input(
    "Message NOVA AI...",
    accept_file="multiple",
    file_type=["png", "jpg", "jpeg", "webp"],
)


if chat_submission is not None:

    prompt = (chat_submission.text or "").strip()
    uploaded_images = chat_submission.files

    if prompt or uploaded_images:

        # Save uploaded images for local display only.
        image_records = []

        for image in uploaded_images:
            image_records.append(
                {
                    "name": image.name,
                    "bytes": image.getvalue(),
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
        # No Azure API call for this submission.
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
                        "Image analysis is temporarily disabled."
                    ),
                }
            )

            st.rerun()

        # --------------------------------------------------
        # PATH 2: Text only
        # Azure OpenAI responses remain enabled.
        # --------------------------------------------------

        elif prompt:

            # Build conversation history before adding this message.
            api_messages = [
                system_instruction,
                *[
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message in st.session_state.messages
                ],
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            with st.chat_message("assistant"):

                try:
                    with st.spinner("NOVA is thinking..."):

                        response = client.responses.create(
                            model=model,
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
                        f"Azure OpenAI API error {error.status_code}: "
                        f"{error.message}"
                    )

                except Exception as error:
                    st.error(f"Unexpected error: {error}")

            st.rerun()

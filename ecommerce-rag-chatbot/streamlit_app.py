import streamlit as st
import requests
import time
import re
import html


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="ShopAssist AI",
    page_icon="🛍️",
    layout="centered"
)


# ============================================================
# CUSTOM PAGE STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #0b0f19;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #101522;
    }

    /* Main title */
    h1 {
        font-size: 2.2rem !important;
        margin-bottom: 0.2rem !important;
    }

    /* Subtitle */
    .stCaption {
        color: #9ca3af;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        border-radius: 18px;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 12px;
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCTIONS
# ============================================================

def clean_answer(text):
    """
    Make sure only normal text is displayed.
    Removes HTML tags and code fences if the backend accidentally
    returns them.
    """

    if not text:
        return "Sorry, I couldn't generate a response."

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Remove code fences
    text = re.sub(
        r"```(?:html|python|json|text|javascript)?",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace("```", "")

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", "", text)

    return text.strip()


def type_stream(text):
    """
    Creates a typing animation.
    """

    for character in text:
        yield character
        time.sleep(0.01)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛍️ ShopAssist AI")

    st.caption("AI-powered e-commerce support")

    st.success("● AI Agent Online")

    st.caption("RAG system connected")

    st.subheader("💬 Chat")

    if st.button("🗑️ Clear conversation", use_container_width=True):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🛍️ ShopAssist AI")

st.caption("AI-powered e-commerce support")

st.success("● AI Agent Online")


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user", avatar="🧑"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant", avatar="🤖"):
            st.write(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

user_message = st.chat_input(
    "Type your message here..."
)


# ============================================================
# PROCESS NEW MESSAGE
# ============================================================

if user_message:

    # --------------------------------------------------------
    # Show user message immediately
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    with st.chat_message("user", avatar="🧑"):

        st.write(user_message)


    # --------------------------------------------------------
    # Generate assistant response
    # --------------------------------------------------------

    with st.chat_message("assistant", avatar="🤖"):

        try:

            # Thinking animation
            with st.spinner("Thinking..."):

                response = requests.post(
                    "http://127.0.0.1:8000/chat",
                    json={
                        "message": user_message
                    },
                    timeout=120
                )


            # ------------------------------------------------
            # Successful response
            # ------------------------------------------------

            if response.status_code == 200:

                data = response.json()

                # IMPORTANT:
                # Your FastAPI backend returns "reply"
                answer = data.get("reply")

                answer = clean_answer(answer)

            else:

                answer = (
                    "Sorry, I couldn't process your request. "
                    "Please try again."
                )


        # ----------------------------------------------------
        # FastAPI is not running
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            answer = (
                "I can't connect to the chatbot server. "
                "Please make sure FastAPI is running."
            )


        # ----------------------------------------------------
        # Request took too long
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            answer = (
                "The chatbot is taking too long to respond. "
                "Please try again."
            )


        # ----------------------------------------------------
        # Any other error
        # ----------------------------------------------------

        except Exception as e:

            answer = (
                "Something went wrong while generating the response. "
                "Please try again."
            )


        # ----------------------------------------------------
        # Typing animation
        # ----------------------------------------------------

        st.write_stream(
            type_stream(answer)
        )


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )
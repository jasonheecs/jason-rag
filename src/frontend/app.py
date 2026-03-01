import json

import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
BACKEND_API_URL = "http://api:80"


@st.cache_resource
def get_http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    session.mount("http://", HTTPAdapter(max_retries=retry))
    return session

st.set_page_config(
    page_title="Ask Jason",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Ask Jason")
st.markdown("Ask questions about Jason's experience, writing, and expertise!")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**{i}. {source['title']}** ({source['source']})")
                    st.markdown(f"*Similarity: {source['similarity']:.2%}*")
                    st.markdown(f"[Link]({source['url']})")
                    st.markdown(f"```\n{source['content'][:200]}...\n```")
                    st.markdown("---")

SUGGESTED_QUESTIONS = [
    "Where did Jason Hee used to work at?",
    "Can he code?",
    "What is his favourite spreadsheet software?",
]

if not st.session_state.messages:
    st.markdown("**Not sure where to start? Try one of these:**")
    cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, suggestion in zip(cols, SUGGESTED_QUESTIONS):
        with col:
            if st.button(suggestion, use_container_width=True):
                st.session_state.suggested_question = suggestion
                st.rerun()

chat_input = st.chat_input("What would you like to know?")
question = st.session_state.pop("suggested_question", None) or chat_input

if question:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Query API with streaming
    with st.chat_message("assistant"):
        try:
            response = get_http_session().post(
                f"{BACKEND_API_URL}/query/stream",
                json={"question": question, "top_k": 5},
                stream=True
            )
            response.raise_for_status()

            captured = {"sources": None}

            def stream_text():
                for raw_line in response.iter_lines():
                    if not raw_line or not raw_line.startswith(b"data: "):
                        continue
                    event = json.loads(raw_line[6:])
                    if event["type"] == "sources":
                        captured["sources"] = event["sources"]
                    elif event["type"] == "text":
                        yield event["content"]

            full_answer = st.write_stream(stream_text())
            sources = captured["sources"] or []

            # Display sources
            with st.expander("📚 View Sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}. {source['title']}** ({source['source']})")
                    st.markdown(f"*Similarity: {source['similarity']:.2%}*")
                    st.markdown(f"[Link]({source['url']})")
                    st.markdown(f"```\n{source['content'][:200]}...\n```")
                    st.markdown("---")

            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_answer,
                "sources": sources
            })

        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to API: {e}")
            st.info(f"Make sure the API is running at {BACKEND_API_URL}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system answers questions based on Jason's:
    - [Medium blog posts](https://jasonheecs.medium.com/)
    - [GitHub repositories](https://github.com/jasonheecs)
    - [LinkedIn profile](https://www.linkedin.com/in/jasonheecs/)
    - [Resume](https://drive.google.com/file/d/18cpyhR8hf3Qf53PWXGmF69e1khGwxuSo/view?usp=sharing)

    **How it works:**
    1. Your question is embedded
    2. Similar content is retrieved
    3. GPT-4o-mini generates an answer
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

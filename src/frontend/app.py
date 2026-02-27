import streamlit as st
import requests

# Configuration
BACKEND_API_URL = "http://api:80"

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

# Chat input
if question := st.chat_input("What would you like to know?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Query API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    f"{BACKEND_API_URL}/query",
                    json={"question": question, "top_k": 5}
                )
                response.raise_for_status()
                result = response.json()

                # Display answer
                st.markdown(result["answer"])

                # Display sources
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"**{i}. {source['title']}** ({source['source']})")
                        st.markdown(f"*Similarity: {source['similarity']:.2%}*")
                        st.markdown(f"[Link]({source['url']})")
                        st.markdown(f"```\n{source['content'][:200]}...\n```")
                        st.markdown("---")

                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"]
                })

            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to API: {e}")
                st.info(f"Make sure the API is running at {BACKEND_API_URL}")

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This RAG system answers questions based on Jason's:
    - Medium articles
    - LinkedIn profile

    **How it works:**
    1. Your question is embedded
    2. Similar content is retrieved
    3. GPT-4o-mini generates an answer
    """)

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


import streamlit as st
from agent import chat_agent

st.set_page_config(page_title="🍔 Food Delivery Assistant", layout="centered")

st.title("🍔 Food Delivery Chatbot")
st.write("Ask about your order status, delivery, or items.")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Type your message here...")

if user_input:
    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call your existing chat agent
    response = chat_agent(user_input)

    # Show assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
    with st.chat_message("assistant"):
        st.markdown(response)

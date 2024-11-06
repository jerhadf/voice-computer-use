import streamlit as st
from evi import chat
import os

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f5f5; /* light gray, adjust as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Hello, Streamlit!")

if 'chat' not in st.session_state:
    st.session_state['chat'] = None
st.session_state['chat'] = st.session_state['chat']
st.title("EVI Chatbot")
st.text("This thing loaded")

user_input_field = st.text_input("user input here")
user_input = None

if st.button("submit user input"):
    user_input = user_input_field

assistant_input_field = st.text_input("assistant input here")
assistant_input = None

if st.button("submit assistant input"):
    assistant_input = assistant_input_field

if 'muted' not in st.session_state:
    st.session_state['muted'] = False

if st.session_state['muted']:
    if st.button("unmute"):
        st.session_state['muted'] = False
else:
    if st.button("mute"):
        st.session_state['muted'] = True

hume_api_key = os.getenv("HUME_API_KEY")
if not hume_api_key:
    st.error("HUME_API_KEY not set")
else:
    result = chat(
        key="chat",
        hume_api_key=hume_api_key,
        user_input_message=user_input,
        assistant_input_message=assistant_input,
        muted=st.session_state['muted'],
    )
    
    print(result and result['type'])

import streamlit as st
import requests
from utils.io_utils import load_config

config = load_config()

st.title("Sentiment Analysis")
text = st.text_input("Insert a text")

if text:
    response = requests.get(config["api"]["url"], params={"text": text})
    st.write(response.json())
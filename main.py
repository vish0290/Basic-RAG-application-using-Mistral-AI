import streamlit as st
import time
import os
from dotenv import load_dotenv
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from take2 import rag_test
import pandas as pd

load_dotenv()
api_key = os.environ["MISTRAL"]
client = MistralClient(api_key=api_key)
st.title("Mistral Demo")
model = st.selectbox(
    "Choose a Model",(
        "mistral-small-latest",
        "mistral-medium-latest",
        "mistral-large-latest"
    )
)
uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=False, type=["csv","xlsx"])

def response(prompt,model):
    messages = [
    ChatMessage(role="user", content=prompt)]
    response = client.chat(model=model, messages=messages)
    return response.choices[0].message.content
    
    
    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])




if prompt := st.chat_input("Enter the prompt"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if uploaded_files is not None:
            df = pd.read_csv(uploaded_files)
            new_file  = "data.csv"
            df.to_csv(new_file,index=False)
            current_path = os.getcwd()
            file_path = os.path.join(current_path, new_file)
            response = st.write(rag_test(file_path,prompt))
        else:
            response = st.write(response(prompt,model))
    st.session_state.messages.append({"role": "assistant", "content": response})
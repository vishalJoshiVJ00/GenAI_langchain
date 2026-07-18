from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from dotenv import load_dotenv

import streamlit as st

load_dotenv()

llm = HuggingFaceEndpoint(

    repo_id= "Qwen/Qwen3-Coder-Next",

    task = 'text-generation'
)

model = ChatHuggingFace(llm= llm)

st.header('research tool')

user_input = st.text_input('enter your prompt')

if st.button('summerize'):
     
     result  = model.invoke(user_input)

     st.write(result.content)
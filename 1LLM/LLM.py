from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model='gpt-3.5-turbo-instruct')

try:
    result = llm.invoke("who is the prime minister of india")
    print(result)
except Exception as error:
    print("OpenAI request failed:", type(error).__name__, error)

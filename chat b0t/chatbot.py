#build the chatbot using langchain by vishal
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen3-Coder-Next",
    task = 'text-generation'

)
model = ChatHuggingFace(llm = llm)

chat_history = [
    SystemMessage(content = 'you are the helpful assistent')
]
while True:
    user_input = input('you:')
    chat_history.append(HumanMessage(content = user_input))
    if user_input == 'exit':
        break
    else:
        result =  model.invoke(chat_history)
        chat_history.append(AIMessage(content = result.content))
        print(result.content)
        
print(chat_history) 
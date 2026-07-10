from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(

    repo_id= "Qwen/Qwen3-Coder-Next",

    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("who is the prime minister of india")

print(result.content)
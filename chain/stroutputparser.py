from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate

load_dotenv()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-Coder-Next'
)

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template = "give me a five lines about the {topic}",
    input_variables = ['topic']
)

chain = template | model | parser
result = chain.invoke({'topic': 'cricket'})
print(result)
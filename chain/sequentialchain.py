from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser
 
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-Coder-Next'
)
model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'tell me about the {country} for travel ',
    input_variables = ['country']
)

prompt2 = PromptTemplate(
    template = "give me a five line about {text}",
    input_variables = ['text']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"country" : "india"})
print(result)
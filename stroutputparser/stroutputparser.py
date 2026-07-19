from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(

    repo_id= "Qwen/Qwen3-Coder-Next",

    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

#1st prompt 
template1 = PromptTemplate(
    template = 'tell me about {topic}' ,
    input_variables = ['topic']
)

#2nd prompt --> summary
template2 = PromptTemplate(
    template = 'write a summary of the following {text}',
    input_variables = ['text']
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic': 'black hole'})

print(result)
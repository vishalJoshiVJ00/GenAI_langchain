from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/Qwen3-Coder-Next',
)

llm2 = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.1-8B-Instruct'
)

model1 = ChatHuggingFace(
    llm = llm
)

model2 = ChatHuggingFace(
    llm = llm2
)

template1 = PromptTemplate(
    template = 'tell me about {country} country ' , 
    input_variables = ['country']
)

template2 = PromptTemplate(
    template = ' give me a five question answer about {text} ' , 
    input_variables = ['text']
)

template3 = PromptTemplate(
    template = 'marge the about couuntry and question answer , country = {country} , text  = {text}',
    input_variables = ['country', 'text']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'country' : template1 | model1 | parser ,
        'text' : template1 | model1 | parser | template2 | model2 | parser
    }
)

merge_chain = template3 | model1 | parser 

chain = parallel_chain | merge_chain

result = chain.invoke({'country': 'india'})
print(result)
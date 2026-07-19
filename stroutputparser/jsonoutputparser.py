from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import JsonOutputParser

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(

    repo_id= "Qwen/Qwen3-Coder-Next",

    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template = 'give me a random name , age or city {format_instruction}' ,
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions()}
)

prompt = template.format()

chain = template | model | parser 

result = chain.invoke({})
print(result)

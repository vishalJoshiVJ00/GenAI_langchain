from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import PydanticOutputParser 

from pydantic import BaseModel , Field

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(

    repo_id= "Qwen/Qwen3-Coder-Next",

    task = 'text-generation'
)

model = ChatHuggingFace(llm = llm)

#schema
class person(BaseModel):
    name : str = Field(descirption = 'name of the person')
    age : int = Field(gt = 18 , description = 'age of the person')
    city : str = Field(description = "city of the person ")

parser = PydanticOutputParser(pydantic_object=person)

template = PromptTemplate(
    template = "give me a random name , age and city of the person {format_instruction}",
    input_variables = [],
    partial_variables = {'format_instruction': parser.get_format_instructions}
)

chain = template | model | parser 
result = chain.invoke({})
print(result)
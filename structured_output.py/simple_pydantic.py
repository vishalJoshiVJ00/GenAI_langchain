from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="katsumi1701/qwen3-4b-structured-output-lora-sft2",
    temperature=0.1
)

model = ChatHuggingFace(llm=llm)

#schema
class review(BaseModel):
    key_themes : list[str] = Field(description = 'write down all the key themes discussed in the review in the list ')
    summery : str = Field(description = "a brief summery of the review ")
    sentiment : str = Field(description = "return statement of review like : negative, positive, neutual")
    pros : list[str] = Field(description = "all the positive point of review")
    cons : list[str] = Field(description = "all the negative point of review")    
    name : str = Field(description = "write the name of the reviewer")
struct_model = model.with_structured_output(review)

prompt = """
the facewash is useful for face and very effective, easy to use but highly cost compared with other
"""

result = struct_model.invoke(prompt)

print(result)

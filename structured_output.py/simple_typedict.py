from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from typing import TypedDict, Annotated

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-Next",
    task = 'text-generation'
)

model = ChatHuggingFace(llm=llm)

#schema
class review(TypedDict):
    summary: Annotated[str, "a brief summary of this review"]
    sentiment: Annotated[str, "return sentiment of the review like : negative , positive , neutural"]

struct_model = model.with_structured_output(review)

prompt = """
the facewash is useful for face and very effective, easy to use but highly cost compared with other
"""

result = struct_model.invoke(prompt)

print(result)

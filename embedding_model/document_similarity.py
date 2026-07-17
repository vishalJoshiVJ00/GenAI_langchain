from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

documents  = [
    "virat kohli is the indian cricketer and looking like a young man",
    "ms dhoni is the number one cricket player in the world",
    "hardik pandaya is the number one bolwer in the world"
]

query = 'tell me about virat kohli'

doc_emb = embedding.embed_documents(documents)

q_emb = embedding.embed_query(query)

score = (cosine_similarity([q_emb], doc_emb))

high_index = score.argmax()

print(query)
print(documents[high_index])
print(f"the index is {high_index}")
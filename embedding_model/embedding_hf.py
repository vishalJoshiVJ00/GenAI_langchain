from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = ['delhi is the capital of india',
        'vishal is the good boy '
        'vishal is 19 year old'
        ]

vector = embedding.embed_documents(text)

print(str(vector))
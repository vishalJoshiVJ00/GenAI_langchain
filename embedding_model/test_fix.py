from sentence_transformers import SentenceTransformer

print("Attempting download...")
# This forces the download using the native library
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("Success! Model is loaded.")

# Now try a quick encoding
vec = model.encode("test")
print(f"Vector sample: {vec[:5]}")
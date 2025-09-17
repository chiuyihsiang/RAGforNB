import yaml, chromadb, sentence_transformers, transformers, langchain

#open yaml files
files = ["SO-XP.yaml", "SO-11P.yaml"]
data = {}
for f in files:
    with open(f, "r", encoding = "utf-8") as file:
        data[f] = yaml.safe_load(file)

#embedding model loading
embedding_model = sentence_transformers.SentenceTransformer("BAAI/bge-m3")

# Initialize ChromaDB
client = chromadb.Client()

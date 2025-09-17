import yaml, chromadb, sentence_transformers, transformers, langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

#open yaml files
files = ["SO-XP.yaml", "SO-11P.yaml"]

data = {}
for f in files:
    with open(f, "r", encoding = "utf-8") as file:
        data[f] = yaml.safe_load(file)

content = {}
for m, c in data.items():
    content[m] = {
        "titles": c.get("title", ""),
        "tags": c.get("tags", []),
        "url": c.get("url", ""),
        "content": c.get("content", "")
    }

#chunking
chunked = []
splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 120)
for j, d in content.items():
    for i in splitter.split_text(d["content"]):
        chunked.append(Document(
            page_content = i,
            metadata = {
                "title": d["content"],
                "tags": d["tags"],
                "url": d["url"]
            }
        ))

#embedding model loading
embedding_model = sentence_transformers.SentenceTransformer("BAAI/bge-m3")

# Initialize ChromaDB
client = chromadb.Client()
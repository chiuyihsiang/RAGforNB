import yaml, chromadb, sentence_transformers, torch, os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from pathlib import Path


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


# Initialize ChromaDB
DB_path = Path("data/chroma").as_posix()
os.makedirs(DB_path, exist_ok = True)
client = chromadb.PersistentClient(path = DB_path)


#chunking
chunked = []
splitter = RecursiveCharacterTextSplitter(chunk_size = 400, chunk_overlap = 80)
for j, d in content.items():
    for i in splitter.split_text(d["content"]):
        chunked.append(Document(
            page_content = "passage: " +  i,
            metadata = {
                "title": d["titles"],
                "tags": d["tags"],
                "url": d["url"]
            }
        ))

contents = [doc.page_content for doc in chunked]
metadata = [doc.metadata for doc in chunked]


#embedding model
cuda = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = sentence_transformers.SentenceTransformer("intfloat/e5-large-v2", device = cuda)
embedded_data = embedding_model.encode(contents,
                                       batch_size = 32,
                                       normalize_embedding = True
                                       ).tolist()


#create collection and push embedding data
collection = client.get_or_create_collection(
                                                name = "SOP_file",
                                                metadata={"hnsw:space": "cosine"}
                                            )

ids = [f"{m['title']}-{idx}" for idx, m in enumerate(metadata)]
batch = 256

for i in range(0, len(ids), batch):
    collection.upsert(
                    ids = ids[i:i+batch],
                    embeddings = embedded_data[i:i+batch],
                    documents = contents[i:i+batch],
                    metadatas = metadata[i:i+batch]
                    )
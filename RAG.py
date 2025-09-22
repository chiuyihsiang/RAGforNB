import chromadb, transformers, torch, os, langchain
from sentence_transformers import CrossEncoder

query = input("how may I help you? ")
prompt = ""

#data extraction and query searching
client = chromadb.PersistentClient(path = "data/chroma")
coll = client.get_collection(name = "SOP_file")
result = coll.query(query_texts = [query], n_results = 5, include = ["documents", "metadatas", "distances"])

#re-ranking
cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", device = "cuda" if torch.cuda.is_available() else "cpu")

def rerank (query, doc, k = 2):
    pairs = [[query, d] for d in doc]
    scores = cross_encoder.predict(pairs)
    top_k = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:k]
    return [(doc[i], scores[i]) for i in top_k]

reranked = rerank(query, result['documents'][0], k = 2)
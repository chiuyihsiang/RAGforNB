import chromadb, torch, sentence_transformers
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

#embedding query
devices = "cuda" if torch.cuda.is_available() else "cpu"
model = sentence_transformers.SentenceTransformer("intfloat/e5-large-v2", device = devices)

query = input("how may I help you? ").strip()
query_vec = model.encode(["query: " + query], normalize_embeddings = True).tolist()

#data extraction and query searching
client = chromadb.PersistentClient(path = "data/chroma")
coll = client.get_collection(name = "SOP_file")
result = coll.query(query_texts = [query], n_results = 5, include = ["documents", "metadatas", "distances"])

#re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device = "cuda" if torch.cuda.is_available() else "cpu")

def rerank (query, doc, k = 2):
    pairs = [[query, d] for d in doc]
    scores = cross_encoder.predict(pairs)
    top_k = sorted(range(len(scores)), key = lambda i: scores[i], reverse = True)[:k]
    return [(doc[i], scores[i]) for i in top_k]

reranked = rerank(query, result['documents'][0], k = 2)

#content generating 
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast = True)
gen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", 
                                           torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
                                           load_in_4bit = True
                                          ).to(devices)

blocks = []
for t, m, s in reranked: 
    header = f"[doc={m.get('source_file')} v={m.get('version','?')} #chunk={m.get('chunk_id','?')}] title={m.get('title','')}"
    body = t.strip().replace("\n"," ")
    blocks.append(header + "\n" + (body[:1200] + ("…" if len(body) > 1200 else "")))
content_blocks = "\n\n".join(blocks)

prompt = f"""
你是公司內部SOP助理，只允許根據「參考文件」作答；若資訊不足，回答「資料不足」並列出所需資訊。
請以繁體中文條列式回答：
1. 【步驟】
2. 【注意事項】
3. 【參考來源】（引用格式 [doc=檔名 v=版本 #chunk=編號]）

[問題]
{query}
""".strip()

input = tok(prompt, return_tensors = "pt").to(gen.device)
with torch.no_grad():
    out = gen.generate(
        **input,
        max_new_tokens = 200,
        temperature = 0.5,
        top_p = 0.9,
        do_sample = True,
        eos_token_id = tok.eos_token_id
    )
answer = tok.decode(out[0], skip_special_token = True)
print(answer)
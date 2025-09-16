import sentence_transformers, yaml, transformers, chromadb

#open yaml files
files = ["SO-XP.yaml", "SO-11P.yaml"]
data = {}
for f in files:
    with open(f, "r", encoding = "utf-8") as file:
        data[f] = yaml.safe_load(file)

print(data["SO-XP.yaml"])
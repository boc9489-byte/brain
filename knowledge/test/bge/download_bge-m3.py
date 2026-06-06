
from modelscope import snapshot_download

local_dir = snapshot_download(model_id="BAAI/bge-reranker-large", local_dir="/Users/bob/Documents/BAAI/bge-reranker-large")
# local_dir = snapshot_download(model_id="BAAI/bge-m3", local_dir="/Users/bob/Documents/bge-m3")

print(local_dir)
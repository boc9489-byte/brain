from pymilvus.model.hybrid import BGEM3EmbeddingFunction

bge_m3_ef = BGEM3EmbeddingFunction(
    model_name="/Users/bob/Documents/bge-m3", # Specify the model name
    device='mps', # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    use_fp16=True # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
)
# {'dense': [array([-0.002003,  0.009575, -0.03595 , ..., -0.03513 , -0.0523  , -0.01767 ], shape=(1024,), dtype=float16)], 
# 'sparse': <Compressed Sparse Row sparse array of dtype 'float64' with 3 stored elements and shape (1, 250002)>
# }
vector_result = bge_m3_ef.encode_queries(queries=['我是中国人','你是美国人'])
# print(vector_result.get("dense")[0].shape)

# vector_result.get('sparse')
print(vector_result.get('sparse'))

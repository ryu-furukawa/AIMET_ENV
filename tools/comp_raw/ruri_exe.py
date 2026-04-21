import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np

# Download from the 🤗 Hub
#model = SentenceTransformer("/root/ruri-small-v2", trust_remote_code=True)
model = SentenceTransformer("/root/AIMET_ruri/ruri-small-v2", local_files_only=True, trust_remote_code=True)

# Don't forget to add the prefix "クエリ: " for query-side or "文章: " for passage-side texts.
sentences = ["クエリ: 瑠璃色はどんな色？"]

embeddings = model.encode(sentences, output_value="token_embeddings", convert_to_tensor=True, normalize_embeddings=False)


# 1文だけなので 0 番目を取り出す
print(embeddings[0].shape)  # 例: torch.Size([seq_len, 768])

arr=embeddings[0].cpu().numpy().astype(np.float32)
print("converted to numpy:", arr.shape, arr.dtype)  
# .raw 保存
arr.tofile("/root/AIMET_ENV/tools/comp_raw/ruri_0_seq.raw")
print("saved ruri_0_seq.raw")


embeddings_2 = model.encode(sentences, convert_to_tensor=True)
print(embeddings_2.shape)  # 例: torch.Size([768])
arr_2 = embeddings_2.cpu().numpy().astype(np.float32)
print("converted to numpy:", arr_2.shape, arr_2.dtype)
arr_2.tofile("/root/AIMET_ENV/tools/comp_raw/ruri_0_pooled.raw")
print("saved ruri_0_pooled.raw")
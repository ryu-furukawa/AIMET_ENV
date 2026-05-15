
from transformers import AutoTokenizer
import numpy as np
MODEL_ID = "/root/AIMET_ruri/ruri-small-v2"
# tokenizer は事前にロード
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)
with open("sample.txt") as f:
    lines = [l.strip() for l in f]

for idx, text in enumerate(lines):
    out = tokenizer(text, max_length=512, padding='max_length', truncation=True)

    np.array(out["input_ids"], dtype=np.int32).tofile(f"raw/{idx}_input_ids.raw")
    np.array(out["attention_mask"], dtype=np.int32).tofile(f"raw/{idx}_attention_mask.raw")
    shape = np.array(out["input_ids"], dtype=np.int32).shape
    np.zeros(shape, dtype=np.int32).tofile(f"raw/{idx}_token_type_ids.raw")  # token_type_ids が無い場合は全0で代用
    np.zeros(shape, dtype=np.int32).tofile(f"raw/{idx}_position_ids.raw")    # position_ids が無い場合は全0で代用
    #np.array(out["token_type_ids"], dtype=np.int32).tofile(f"raw/{idx}_token_type_ids.raw")
    #np.array(out["position_ids"], dtype=np.int32).tofile(f"raw/{idx}_position_ids.raw")
    #print(np.array(out["input_ids"], dtype=np.int32))
    #print(np.array(out["attention_mask"], dtype=np.int32))
    #print(np.array(out["token_type_ids"], dtype=np.int32))
    #print(np.array(out["position_ids"], dtype=np.int32))


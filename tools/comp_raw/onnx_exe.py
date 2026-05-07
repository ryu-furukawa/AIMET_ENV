# pip install transformers onnxruntime
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

MODEL_DIR = "/root/AIMET_ruri/ruri-small-v2/"
ONNX_PATH = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model.onnx"  # 実在パスに合わせて
TEXT = "クエリ: 瑠璃色はどんな色？"

tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
enc = tok(TEXT, return_tensors="np", truncation=True, max_length=512)

# onnxruntime は int64 を推奨
input_ids = enc["input_ids"].astype(np.int64)
attention_mask = enc["attention_mask"].astype(np.int64)

print(input_ids)
print(attention_mask)

sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# モデルが要求する入力名に合わせる（ここでは一般的な名前を想定）
inputs = {
    "input_ids": input_ids,
    "attention_mask": attention_mask
}

# もし入力名が違っていたら、以下2行のコメントを外して名前を確認 → dict のキーを合わせる
print([i.name for i in sess.get_inputs()])
print([o.name for o in sess.get_outputs()])

outs = sess.run(None, inputs)
#print("onnx output:", outs)
arr = outs[0][0].astype(np.float32)  # (seq_len, hidden)
print("onnx output shape:", arr.shape, "dtype:", arr.dtype)
arr.tofile("/root/AIMET_ENV/tools/comp_raw/onnx.raw")
print("saved onnx.raw", arr.shape, arr.dtype)
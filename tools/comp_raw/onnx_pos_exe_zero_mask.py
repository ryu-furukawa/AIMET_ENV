# pip install transformers onnxruntime onnx
from transformers import AutoTokenizer
import onnx
import onnxruntime as ort
import numpy as np
import os

MODEL_DIR = "/root/AIMET_ENV/tools/ruri/ruri-small-v2"
ONNX_PATH = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_fix.onnx"


TEXT = "クエリ: 瑠璃色はどんな色？"


# ========================================
# shape固定関数
# ========================================
def fix_input_shapes(model, shape_dict):
    for inp in model.graph.input:
        if inp.name in shape_dict:
            dims = inp.type.tensor_type.shape.dim
            new_shape = shape_dict[inp.name]

            for i, d in enumerate(new_shape):
                dims[i].ClearField("dim_param")
                dims[i].dim_value = d

    return model


# ========================================
# ① ONNXをロードしてshape固定
# ========================================
model = onnx.load(ONNX_PATH)

model = fix_input_shapes(
    model,
    {
        "input_ids":      [1, 512],
        "attention_mask": [1, 512],
        "token_type_ids": [1, 512],
        "position_ids":   [1, 512],
    }
)


# ========================================
# ② tokenizer
# ========================================
tok = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    trust_remote_code=True
)


# ========================================
# ③ tokenize（あなたのコードそのまま）
# ========================================
enc = tok(
    TEXT,
    return_tensors="np",
    truncation=True,
    padding="max_length",   # ✅これを追加
    max_length=512
)

input_ids = enc["input_ids"].astype(np.int64)
attention_mask = enc["attention_mask"].astype(np.int64)

#token_type_ids =  np.zeros_like(input_ids, dtype=np.int64)
#position_ids = np.zeros_like(input_ids, dtype=np.int64)
token_type_ids = np.zeros((1, 512), dtype=np.int64)
position_ids   = np.zeros((1, 512), dtype=np.int64)



# ========================================
# ④ ONNX推論
# ========================================
sess = ort.InferenceSession(model.SerializePartialToString(), providers=["CPUExecutionProvider"])

onnx_inputs = [i.name for i in sess.get_inputs()]

def pick(*candidates):
    for c in candidates:
        if c in onnx_inputs:
            return c
    return None

name_map = {}
name_map["input_ids"]      = pick("input_ids", "input_ids:0")
name_map["attention_mask"] = pick("attention_mask", "attention_mask:0")
name_map["token_type_ids"] = pick("token_type_ids")
name_map["position_ids"]   = pick("position_ids")

inputs = {}

if name_map["input_ids"]:
    inputs[name_map["input_ids"]] = input_ids

if name_map["attention_mask"]:
    inputs[name_map["attention_mask"]] = attention_mask

if name_map["token_type_ids"]:
    inputs[name_map["token_type_ids"]] = token_type_ids

if name_map["position_ids"]:
    inputs[name_map["position_ids"]] = position_ids


# ========================================
# ⑤ 推論実行
# ========================================
outs = sess.run(None, inputs)

print("num outputs:", len(outs))

# ========================================
# ⑥ sequence_output（pooling前）
# ========================================
seq = outs[0][0].astype(np.float32)
print("sequence_output shape:", seq.shape)

seq.tofile("lock_fix_0_seq.raw")


# ========================================
# ⑦ pooled_output（ONNXの結果）
# ========================================
pooled = outs[1][0].astype(np.float32)

print("pooled_output shape:", pooled.shape)
print("norm:", np.linalg.norm(pooled))

pooled.tofile("lock_fix_0_pool.raw")


print("saved raw files ✅")

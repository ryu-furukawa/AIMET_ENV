# pip install transformers onnxruntime
from transformers import AutoTokenizer
import onnxruntime as ort
import numpy as np

MODEL_DIR = "/root/AIMET_ENV/tools/ruri/ruri-small-v2"
ONNX_PATH = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_fix.onnx"  # 実在パスに合わせて
TEXT = "クエリ: 瑠璃色はどんな色？"


tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
# ------------------------------------------------------------
# 1) Tokenize（右/左パディングに依存せず position_ids を安全に作る）
# ------------------------------------------------------------
enc = tok(TEXT, return_tensors="np", truncation=True, max_length=512)

# onnxruntime は int64 を推奨
input_ids = enc["input_ids"].astype(np.int64)              # (1, L)
attention_mask = enc["attention_mask"].astype(np.int64)    # (1, L)


# token_type_ids が出ないトークナイザもあるので、必要ならゼロで作る
token_type_ids =  np.zeros_like(input_ids, dtype=np.int64)
position_ids = np.zeros_like(input_ids, dtype=np.int64)

# ------------------------------------------------------------
# 3) ONNX 実体の入力名に “自動で” 合わせる
# ------------------------------------------------------------
sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

# モデルが要求する入力名を取得
onnx_inputs = [i.name for i in sess.get_inputs()]
# print("ONNX input names:", onnx_inputs)

# よくある名前のマップ（存在すれば使う）
name_map = {}

def pick(*candidates):
    for c in candidates:
        if c in onnx_inputs:
            return c
    return None

name_map["input_ids"]      = pick("input_ids", "input_ids:0", "ids")
name_map["attention_mask"] = pick("attention_mask", "attention_mask:0", "mask")
name_map["token_type_ids"] = pick("token_type_ids", "token_type_ids:0", "token_type_ids_0")
name_map["position_ids"]   = pick("position_ids", "position_ids:0", "pos_ids", "position_ids_0")

inputs = {}
if name_map["input_ids"]:
    inputs[name_map["input_ids"]] = input_ids
if name_map["attention_mask"]:
    inputs[name_map["attention_mask"]] = attention_mask
if name_map["token_type_ids"] and token_type_ids is not None:
    inputs[name_map["token_type_ids"]] = token_type_ids
# ★ ここで position_ids を追加
if name_map["position_ids"]:
    inputs[name_map["position_ids"]] = position_ids

# 何を渡すべきか分からないときのデバッグ（必要に応じてコメント解除）
# print("Prepared inputs:", {k: v.shape for k, v in inputs.items()})
# print("Expected outputs:", [o.name for o in sess.get_outputs()])

outs = sess.run(None, inputs)
arr = outs[0][0].astype(np.float32)  # 例: (seq_len, hidden) あるいは (hidden,) などモデル次第
arr.tofile("onnx_0_seq_pos.raw")
print("saved onnx_0_seq_pos.raw", arr.shape, arr.dtype)

aar2=outs[1][0].astype(np.float32)  # 例: pooled output (hidden,) などモデル次第
aar2.tofile("onnx_0_pooled_pos.raw")
print("saved onnx_0_pooled_pos.raw", aar2.shape, aar2.dtype)

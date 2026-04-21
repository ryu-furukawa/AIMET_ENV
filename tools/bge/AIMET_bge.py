import os
import onnx
import numpy as np
from aimet_onnx import QuantizationSimModel
from aimet_onnx.common.defs import QuantScheme
import aimet_onnx

# ----------------------------
# 1. shape固定済み ONNX をロード
# ----------------------------
#model = onnx.load("/root/AIMET/convert/simplified_fix.onnx")
model = onnx.load("/root/AIMET/bge-onnx/model.onnx")  # ← 適宜変更

OUT_DIR="/root/AIMET_ENV/tools/bge"  # ← 適宜変更
# 2) (推奨) simplify
try:
    import onnxsim
    model, _ = onnxsim.simplify(model)
except Exception as e:
    print("onnxsim simplify failed, continue with original model:", repr(e))

# ----------------------------
# 2. QuantizationSimModel 作成
# ----------------------------
providers=["CPUExecutionProvider"]
sim = QuantizationSimModel(

    model,
    param_type=aimet_onnx.int8,
    activation_type=aimet_onnx.int8,
    #quant_scheme=QuantScheme.min_max,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    #config_file="default",
    config_file="htp_v73",
    #config_file="/root/AIMET_ruri_v2/tool/quantsim_fp_escape.json",  # ← 適宜変更
    providers=providers,
)

# ONNX input names (must match tokenizer outputs)
onnx_input_names = [i.name for i in sim.model.model.graph.input]
print("ONNX inputs:", onnx_input_names)



def make_feed(text: str, max_length: int = 512):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    feed = {}

    # input_ids / attention_mask / token_type_ids
    if "input_ids" in onnx_input_names:
        feed["input_ids"] = enc["input_ids"].astype(np.int64)

    if "attention_mask" in onnx_input_names:
        feed["attention_mask"] = enc["attention_mask"].astype(np.int64)

    if "token_type_ids" in onnx_input_names:
        if "token_type_ids" in enc:
            feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        else:
            # tokenizer が返さない場合は 0 埋め
            feed["token_type_ids"] = np.zeros_like(
                enc["input_ids"], dtype=np.int64
            )

    # position_ids を明示的に追加
    if "position_ids" in onnx_input_names:
        batch_size, seq_len = enc["input_ids"].shape
        feed["position_ids"] = np.tile(
            np.arange(seq_len, dtype=np.int64),
            (batch_size, 1),
        )

    # 念のためチェック
    missing = [n for n in onnx_input_names if n not in feed]
    if missing:
        raise RuntimeError(
            f"Missing ONNX inputs {missing}. tokenizer keys={list(enc.keys())}"
        )

    return feed

def calib_generator(texts, num_batches: int = 500):
    # aimet_onnx の compute_encodings は iterable を受け取れる（dictをyield）
    for i, t in enumerate(texts):
        if i >= num_batches:
            break
        yield make_feed(t)



calib_texts = [
    "What is semantic search?",
    "Explain retrieval augmented generation.",
    "How does an embedding model work?",
]*200

# tokenizer は事前にロード済み想定
from transformers import AutoTokenizer 
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")


# ----------------------------
# 5. PTQ Step1 実行（これが目的）
# ----------------------------
sim.compute_encodings(
    calib_generator(calib_texts)
)

print("✅ PTQ Step1 (compute_encodings) completed")

# 6) export (QDQ onnx + encodings json)
sim.export(
    path=OUT_DIR,
    filename_prefix="ruri_normal",
    export_model=True,
    export_int32_bias=True,    
    encoding_version="1.0.0",  # デフォルトでも可
)

print("Done. Exported to:", OUT_DIR)
print("QDQ model:", os.path.join(OUT_DIR, "ruri_normal.onnx"))
print("Encodings:", os.path.join(OUT_DIR, "ruri_normal.encodings"))

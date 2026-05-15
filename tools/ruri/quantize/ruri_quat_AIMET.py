import os
import numpy as np
import onnx
import onnxruntime as ort

from transformers import AutoTokenizer

import aimet_onnx
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx import QuantizationSimModel

#FP32_ONNX = "/root/AIMET_ruri/ruri-onnx/model_simplified.onnx"
#FP32_ONNX = "/root/AIMET_ruri/ruri-onnx/model_simplified.onnx"
FP32_ONNX = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_tpi_mask.onnx"
#FP32_ONNX = "/root/AIMET_ruri/ruri-onnx/model.onnx"  # ← 適宜変更
OUT_DIR = "/root/AIMET_ENV/tools/ruri/model"

os.makedirs(OUT_DIR, exist_ok=True)

MODEL_ID = "/root/AIMET_ENV/tools/ruri/ruri-small-v2"
# tokenizer は事前にロード
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# 1) load onnx
model = onnx.load_model(FP32_ONNX)

# 2) (推奨) simplify
"""
try:
    import onnxsim
    B=1
    S=512
    model, _ = onnxsim.simplify(
        model,
        overwrite_input_shapes={
            "input_ids":      [B, S],
            "attention_mask": [B, S],
            "token_type_ids": [B, S],
            "position_ids":   [B, S],
        },
    )
except Exception as e:
    print("onnxsim simplify failed, continue with original model:", repr(e))
"""

import onnx

model = onnx.load_model(FP32_ONNX)

def fix_input_shapes(model, shapes_dict):
    for inp in model.graph.input:
        name = inp.name
        if name in shapes_dict:
            shape = shapes_dict[name]
            for i, dim in enumerate(inp.type.tensor_type.shape.dim):
                dim.dim_value = shape[i]

    return model


model = fix_input_shapes(
    model,
    {
        "input_ids":      [1, 512],
        "attention_mask": [1, 512],
        "token_type_ids": [1, 512],
        "position_ids":   [1, 512],
    }
)


#onnx.save(model, "fixed.onnx")
    
# 3) create QuantSim (CPU only, W8/A16)
providers = ["CPUExecutionProvider"]
sim = QuantizationSimModel(

    model,
    param_type=aimet_onnx.int8,
    activation_type=aimet_onnx.int16,
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

# 4) calibration data (代表テキスト。実運用に近い文を500〜1000程度推奨)
from datasets import Dataset 
#arrow_path = "/root/jmteb_260326/nlp_journal_abs_article-corpus/corpus/data-00000-of-00001.arrow"  # ← 適宜変更
#arrow_path = "/root/jmteb_260326/nlp_journal_abs_article-query/validation/data-00000-of-00001.arrow"
#arrow_path = "/root/AIMET_ENV/tools/ruri/data/merged_1000.arrow"  # ← 適宜変更
arrow_path = "/root/AIMET_ENV/tools/ruri/data/article_200_exe.arrow"  # ← 適宜変更
ds = Dataset.from_file(arrow_path)
print("Dataset columns:", ds.column_names)

calib_texts = ds["text"]  # ← "text" 列を適宜変更
#calib_texts = ds["query"]  # ← "query" 列を適宜変更


def make_feed(text: str, max_length: int = 512):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
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

def calib_generator(texts, num_batches: int = 1000):
    # aimet_onnx の compute_encodings は iterable を受け取れる（dictをyield）
    for i, t in enumerate(texts):
        if i >= num_batches:
            break
        yield make_feed(t)



# 5) compute encodings
# 代表データ 500～1000サンプルが目安（AIMET docsでもそのレンジ）
sim.compute_encodings(calib_generator(calib_texts, num_batches=1000))
# 6) export (QDQ onnx + encodings json)
OUT_DIR = "/root/AIMET_ENV/tools/ruri/model"

sim.export(
    path=OUT_DIR,
    filename_prefix="ruri_tpi_mask_art",  # ← 適宜変更
    export_model=True,
    export_int32_bias=True,    # 迷ったらTrueでOK（INT32 bias encodingを生成）
    encoding_version="1.0.0",  # デフォルトでも可
)

print("Done. Exported to:", OUT_DIR)
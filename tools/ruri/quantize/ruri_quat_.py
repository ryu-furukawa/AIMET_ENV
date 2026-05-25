import os
import numpy as np
import onnx
import onnxruntime as ort

from transformers import AutoTokenizer

import aimet_onnx
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx import QuantizationSimModel


FP32_ONNX = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add.onnx"
#FP32_ONNX = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_fix_mask.onnx"  # ← 適宜変更
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
    #activation_type=aimet_onnx.int8,
    activation_type=aimet_onnx.int16,
    #quant_scheme=QuantScheme.min_max,
    quant_scheme=QuantScheme.post_training_tf_enhanced,
    #config_file="default",
    config_file="htp_v73",
    #config_file="/root/AIMET_ENV/tools/ruri/quantize/custom_config.json",  # ← 適宜変更
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

print("Done. Exported to:", OUT_DIR)

import os
import copy
import numpy as np
import onnxruntime as ort

RESULTS_DIR = "./layerwise_sensitivity_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def run_session_outputs(session, feed):
    outputs = session.run(None, feed)
    # 通常は先頭出力を使う。embedding用途なら pooled / last_hidden_state の場合もある
    return outputs[0]


# FP session を作成
fp_sess = ort.InferenceSession(
    FP32_ONNX,
    providers=providers,
)

# Quant session を作成
quant_sess = sim.session


# ---------------------------------------------
# 1. まず全体の FP vs Quant の差を見る
# ---------------------------------------------
global_mse = []
global_cos = []

for i, text in enumerate(calib_texts[:200]):   # まず200件くらいで十分
    feed = make_feed(text)

    fp_out = run_session_outputs(fp_sess, feed)
    q_out = run_session_outputs(quant_sess, feed)

    global_mse.append(mse(fp_out, q_out))
    global_cos.append(cosine_similarity(fp_out, q_out))

print("[Global]")
print("  mean MSE   :", np.mean(global_mse))
print("  mean COS   :", np.mean(global_cos))


# ---------------------------------------------
# 2. Quant wrapper 名を列挙
# ---------------------------------------------
quant_wrappers = []
for name, qc in sim.qc_quantize_op_dict.items():
    quant_wrappers.append(name)

print(f"Num quant ops: {len(quant_wrappers)}")
for name in quant_wrappers[:20]:
    print("  ", name)


# ---------------------------------------------
# 3. 1個ずつ無効化して sensitivity を見る
#    = disable analysis 相当の簡易版
# ---------------------------------------------
results = []

for op_name, qc_op in sim.qc_quantize_op_dict.items():
    print(f"[Analyze disable] {op_name}")

    # 退避
    orig_enabled = getattr(qc_op, "enabled", True)

    # disable
    try:
        qc_op.enabled = False
    except Exception as e:
        print(f"  skip disable failed: {e}")
        continue

    mses = []
    coses = []

    for i, text in enumerate(calib_texts[:100]):   # まず100件
        feed = make_feed(text)

        fp_out = run_session_outputs(fp_sess, feed)
        q_out = run_session_outputs(quant_sess, feed)

        mses.append(mse(fp_out, q_out))
        coses.append(cosine_similarity(fp_out, q_out))

    result = {
        "op_name": op_name,
        "mean_mse": float(np.mean(mses)),
        "mean_cos": float(np.mean(coses)),
    }
    results.append(result)

    print(
        f"  mean_mse={result['mean_mse']:.8f}, "
        f"mean_cos={result['mean_cos']:.8f}"
    )

    # 戻す
    qc_op.enabled = orig_enabled


# ---------------------------------------------
# 4. ソートして保存
# ---------------------------------------------
# MSE が小さくなるほど「その op を無効化すると改善した」可能性
results_sorted = sorted(results, key=lambda x: x["mean_mse"])

csv_path = os.path.join(RESULTS_DIR, "disable_analysis.csv")
with open(csv_path, "w") as f:
    f.write("rank,op_name,mean_mse,mean_cos\n")
    for rank, r in enumerate(results_sorted, start=1):
        f.write(f"{rank},{r['op_name']},{r['mean_mse']},{r['mean_cos']}\n")

print(f"Saved: {csv_path}")

print("\n[Top sensitive ops by disable-analysis]")
for r in results_sorted[:30]:
    print(
        f"{r['op_name']:<80} "
        f"MSE={r['mean_mse']:.8f} "
        f"COS={r['mean_cos']:.8f}"
    )
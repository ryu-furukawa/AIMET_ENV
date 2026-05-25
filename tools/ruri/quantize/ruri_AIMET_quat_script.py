import os
import numpy as np
import onnx

from datasets import Dataset
from transformers import AutoTokenizer

import aimet_onnx
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx import QuantizationSimModel


# =========================
# Paths / constants
# =========================

OUT_DIR = "/root/AIMET_ENV/tools/ruri/model/script_output"
MODEL_ID = "/root/AIMET_ENV/tools/ruri/ruri-small-v2"

os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

# mask値ごとのONNXファイル対応
MASK_TO_ONNX = {
    "-6": "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add_-6.onnx",
    "-10": "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add_-10.onnx",
    "fp32min": "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add.onnx",
    # もし初期値版の別onnx名があるなら適宜直す
}

# dataset種別ごとのArrow対応
DATASET_TO_ARROW = {
    "abs": "/root/AIMET_ENV/tools/ruri/data/abs200.arrow",
    "art": "/root/AIMET_ENV/tools/ruri/data/art200.arrow",
    "mix": "/root/AIMET_ENV/tools/ruri/data/mix200.arrow",
}

DATASET_TEXT_COLUMN = {
    "abs": "text",
    "art": "text",
    "mix": "text",
}

METHOD_TO_QUANT_SCHEME = {
    "tf_enhanced": QuantScheme.post_training_tf_enhanced,
    "minmax": QuantScheme.min_max,
}

# 初期値の表示名
FP32_MIN_STR = "-3.4028235e38"


# =========================
# Helpers
# =========================

def load_model_with_fixed_shapes(onnx_path: str):
    model = onnx.load_model(onnx_path)

    shapes_dict = {
        "input_ids": [1, 512],
        "attention_mask": [1, 512],
        "token_type_ids": [1, 512],
        "position_ids": [1, 512],
    }

    for inp in model.graph.input:
        name = inp.name
        if name in shapes_dict:
            shape = shapes_dict[name]
            for i, dim in enumerate(inp.type.tensor_type.shape.dim):
                dim.dim_value = shape[i]

    return model


def make_feed(text: str, tokenizer, onnx_input_names, max_length: int = 512):
    enc = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )

    feed = {}

    if "input_ids" in onnx_input_names:
        feed["input_ids"] = enc["input_ids"].astype(np.int64)

    if "attention_mask" in onnx_input_names:
        feed["attention_mask"] = enc["attention_mask"].astype(np.int64)

    if "token_type_ids" in onnx_input_names:
        if "token_type_ids" in enc:
            feed["token_type_ids"] = enc["token_type_ids"].astype(np.int64)
        else:
            feed["token_type_ids"] = np.zeros_like(enc["input_ids"], dtype=np.int64)

    if "position_ids" in onnx_input_names:
        batch_size, seq_len = enc["input_ids"].shape
        feed["position_ids"] = np.tile(
            np.arange(seq_len, dtype=np.int64),
            (batch_size, 1),
        )

    missing = [n for n in onnx_input_names if n not in feed]
    if missing:
        raise RuntimeError(
            f"Missing ONNX inputs {missing}. tokenizer keys={list(enc.keys())}"
        )

    return feed


def calib_generator(texts, tokenizer, onnx_input_names, num_batches: int = 1000):
    for i, t in enumerate(texts):
        if i >= num_batches:
            break
        yield make_feed(t, tokenizer, onnx_input_names)


def normalize_mask_label(mask_label: str) -> str:
    # exportファイル名に使いやすい形へ
    if mask_label == "fp32min":
        return "fp32min"
    return mask_label.replace("-", "neg")


def run_one_experiment(dataset_kind: str, mask_kind: str, method: str):
    print("=" * 80)
    print(f"START dataset={dataset_kind}, mask={mask_kind}, method={method}")

    onnx_path = MASK_TO_ONNX[mask_kind]
    arrow_path = DATASET_TO_ARROW[dataset_kind]
    text_col = DATASET_TEXT_COLUMN[dataset_kind]
    quant_scheme = METHOD_TO_QUANT_SCHEME[method]

    model = load_model_with_fixed_shapes(onnx_path)

    sim = QuantizationSimModel(
        model,
        param_type=aimet_onnx.int8,
        activation_type=aimet_onnx.int16,
        quant_scheme=quant_scheme,
        config_file="htp_v73",
        providers=["CPUExecutionProvider"],
    )

    onnx_input_names = [i.name for i in sim.model.model.graph.input]
    print("ONNX inputs:", onnx_input_names)

    ds = Dataset.from_file(arrow_path)
    print("Dataset rows:", len(ds))
    print("Dataset columns:", ds.column_names)

    calib_texts = ds[text_col]

    sim.compute_encodings(
        calib_generator(calib_texts, tokenizer, onnx_input_names, num_batches=1000)
    )

    export_prefix = f"ruri_w8a16_{dataset_kind}_mask_{normalize_mask_label(mask_kind)}_{method}"

    sim.export(
        path=OUT_DIR,
        filename_prefix=export_prefix,
        export_model=True,
        export_int32_bias=True,
    )

    print(f"DONE dataset={dataset_kind}, mask={mask_kind}, method={method}")
    print(f"EXPORTED: {os.path.join(OUT_DIR, export_prefix)}")


# =========================
# Experiment table
# =========================

EXPERIMENTS = [
    ("abs", " -6 ", "tf_enhanced"),
]
# 上は消して下を使ってください


EXPERIMENTS = [
    ("abs", " -6 ".strip(), "tf_enhanced"),
    ("art", " -6 ".strip(), "tf_enhanced"),
    ("mix", " -6 ".strip(), "tf_enhanced"),

    ("abs", "fp32min", "tf_enhanced"),
    ("art", "fp32min", "tf_enhanced"),
    ("mix", "fp32min", "tf_enhanced"),

    ("abs", "-10", "tf_enhanced"),
    ("art", "-10", "tf_enhanced"),
    ("mix", "-10", "tf_enhanced"),

    ("abs", "-6", "minmax"),
    ("art", "-6", "minmax"),
    ("mix", "-6", "minmax"),

    ("abs", "fp32min", "minmax"),
    ("art", "fp32min", "minmax"),
    ("mix", "fp32min", "minmax"),

    ("abs", "-10", "minmax"),
    ("art", "-10", "minmax"),
    ("mix", "-10", "minmax"),
]


if __name__ == "__main__":
    for dataset_kind, mask_kind, method in EXPERIMENTS:
        run_one_experiment(dataset_kind, mask_kind, method)
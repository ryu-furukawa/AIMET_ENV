import os
import sys
import json
import dataclasses
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset

import time 

# AIMET関連
import aimet_onnx
from aimet_onnx.common.defs import QuantScheme
from aimet_onnx import QuantizationSimModel

# JMTEB
JMTEB_DIR = Path("/root/JMTEB")
sys.path.insert(0, str(JMTEB_DIR / "src"))
from jmteb.evaluators.retrieval.data import HfRetrievalQueryDataset, HfRetrievalDocDataset
from jmteb.evaluators.retrieval.evaluator import RetrievalEvaluator

# ========= MODEL定義 ===============
FP32_ONNX = "/root/AIMET_ENV/tools/ruri/quantize/ruri-export-onnx/model_mask_add_-6.onnx"
MODEL_ID = "/root/AIMET_ENV/tools/ruri/ruri-small-v2"
DATASET_ROOT = "/root/jmteb_260326"

# ======== 量子化モデルリスト & キャリブセットアップ ========
model_list = [
    "ruri_w8a16_abs_title_-6",
]
CALIB_MAP = {
    "ruri_w8a16_abs_title_-6": "/root/AIMET_ENV/tools/ruri/data/query_abs_title200.arrow",
}
CALIB_BATCHES = {
    "ruri_w8a16_abs_title_-6": 1000,

}

# ========= タスクリスト例 =========
TASKS = [
    {
        "task_name": "nlp_journal_title_abs",
        "query_name": "nlp_journal_title_abs-query",
        "corpus_name": "nlp_journal_title_abs-corpus",
    },
    {
        "task_name": "nlp_journal_title_intro",
        "query_name": "nlp_journal_title_intro-query",
        "corpus_name": "nlp_journal_title_intro-corpus",
    },
    {
        "task_name": "nlp_journal_abs_intro",
        "query_name": "nlp_journal_abs_intro-query",
        "corpus_name": "nlp_journal_abs_intro-corpus",
    },
    {
        "task_name": "nlp_journal_abs_article",
        "query_name": "nlp_journal_abs_article-query",
        "corpus_name": "nlp_journal_abs_article-corpus",
    },
    {
        "task_name": "jagovfaqs_22k",
        "query_name": "jagovfaqs_22k-query",
        "corpus_name": "jagovfaqs_22k-corpus",
    },
]

# ========= ユーティリティ ==============
def _to_jsonable(obj):
    if dataclasses.is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)

def save_results(results, out_path: str):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _to_jsonable(results)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", out_path)

# ========= EMBEDDER 実装 ==============
class AimetOnnxEmbedder:
    def __init__(self, sim, tokenizer, session=None, max_length=512, pool="mean"):
        self.sim = sim
        self.tokenizer = tokenizer
        self.session = session
        self.max_length = max_length
        self.pool = pool

    def _pool(self, last_hidden_state, attention_mask):
        if self.pool == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.astype(np.float32)
        denom = np.clip(mask.sum(axis=1, keepdims=True), 1e-6, None)
        return (last_hidden_state * mask[:, :, None]).sum(axis=1) / denom

    def encode(self, sentences, batch_size=1, **kwargs):
        embs = []
        sess = self.session or self.sim.session
        required = {i.name for i in sess.get_inputs()}
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            tok = self.tokenizer(
                batch,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            input_ids = tok["input_ids"].astype(np.int64)
            attention_mask = tok["attention_mask"].astype(np.int64)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if "token_type_ids" in required:
                if "token_type_ids" in tok:
                    inputs["token_type_ids"] = tok["token_type_ids"].astype(np.int64)
                else:
                    inputs["token_type_ids"] = np.zeros_like(input_ids, dtype=np.int64)
            if "position_ids" in required:
                bsz, seqlen = input_ids.shape
                pos = np.arange(seqlen, dtype=np.int64)[None, :]
                inputs["position_ids"] = np.repeat(pos, bsz, axis=0)
            outputs = sess.run(None, inputs)
            pooled = outputs[1].astype(np.float32)
            embs.append(pooled)
        return np.vstack(embs)

class TextEmbedderCompat:
    def __init__(self, embedder: AimetOnnxEmbedder, batch_size=1):
        self.embedder = embedder
        self.batch_size = batch_size

    def set_output_tensor(self):
        return

    def reset_max_seq_length(self):
        return

    def batch_encode_with_cache(self, text_list, prefix=None, cache_path=None, overwrite_cache=False, **kwargs):
        if prefix:
            text_list = [prefix + t for t in text_list]
        return self.embedder.encode(text_list, batch_size=self.batch_size)

# ==== モデル構築&キャリブ ====
def fix_input_shapes(model, shapes_dict):
    for inp in model.graph.input:
        name = inp.name
        if name in shapes_dict:
            shape = shapes_dict[name]
            for i, dim in enumerate(inp.type.tensor_type.shape.dim):
                dim.dim_value = shape[i]
    return model

def build_sim_and_embedder(calib_texts, num_batches):
    import onnx
    model = onnx.load_model(FP32_ONNX)
    model = fix_input_shapes(
        model,
        {
            "input_ids": [1, 512],
            "attention_mask": [1, 512],
            "token_type_ids": [1, 512],
            "position_ids": [1, 512],
        }
    )

    sim = QuantizationSimModel(
        model,
        param_type=aimet_onnx.int8,
        activation_type=aimet_onnx.int16,
        quant_scheme=QuantScheme.post_training_tf_enhanced,
        config_file="htp_v73",  # 必須: 量子化設定jsonへのパス
        providers=["CPUExecutionProvider"],
    )

    onnx_input_names = [i.name for i in sim.model.model.graph.input]

    def make_feed(text: str):
        enc = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed = {}
        if "input_ids" in onnx_input_names:
            feed["input_ids"] = enc["input_ids"].astype(np.int64)
        if "attention_mask" in onnx_input_names:
            feed["attention_mask"] = enc["attention_mask"].astype(np.int64)
        if "token_type_ids" in onnx_input_names:
            feed["token_type_ids"] = enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(np.int64)
        if "position_ids" in onnx_input_names:
            bsz, seqlen = enc["input_ids"].shape
            feed["position_ids"] = np.tile(
                np.arange(seqlen, dtype=np.int64),
                (bsz, 1),
            )
        return feed

    def calib_generator():
        for i, t in enumerate(calib_texts):
            if i >= num_batches:
                break
            yield make_feed(t)

    sim.compute_encodings(calib_generator())
    print(f"calib done: {num_batches}")

    embedder = TextEmbedderCompat(
        AimetOnnxEmbedder(sim=sim, tokenizer=tokenizer),
        batch_size=1
    )
    return embedder

# ==== JMTEB評価本体 ====
def evaluate_one_task(model, dataset_root, task_conf, cache_base_dir="cache_multi", overwrite_cache=False):
    task_name = task_conf["task_name"]
    query_name = task_conf["query_name"]
    corpus_name = task_conf["corpus_name"]
    query_ds = HfRetrievalQueryDataset(
        path="sbintuitions/JMTEB",
        name=query_name,
        split="test",
        query_key="query",
        relevant_docs_key="relevant_docs",
        dataset_path=dataset_root,
    )
    doc_ds = HfRetrievalDocDataset(
        path="sbintuitions/JMTEB",
        name=corpus_name,
        split="corpus",
        id_key="docid",
        text_key="text",
        dataset_path=dataset_root,
    )
    evaluator = RetrievalEvaluator(
        val_query_dataset=query_ds,
        test_query_dataset=query_ds,
        doc_dataset=doc_ds,
        doc_chunk_size=200_000,
        log_predictions=False,
    )
    cache_dir = os.path.join(cache_base_dir, task_name)
    print(f"===== evaluating: {task_name} =====")
    results = evaluator(model, cache_dir=cache_dir, overwrite_cache=overwrite_cache)
    return results

# ==== メインループ ====
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    all_results = {}
    infer_times = {}

    for model_name in model_list:
        print(f"\n===== CALIB + EVAL: {model_name} =====")
        arrow_path = CALIB_MAP[model_name]
        num_batches = CALIB_BATCHES[model_name]

        # simモデルの作成・calib部分は計測【外】
        with pa.memory_map(arrow_path, "r") as source:
            reader = ipc.RecordBatchStreamReader(source)
            table = reader.read_all()
            ds = Dataset(table)
        calib_texts = ds["text"]

        embedder_model = build_sim_and_embedder(calib_texts, num_batches)
        model_results = {}

        # 推論部分のみ計測
        infer_start = time.time()
        for task_conf in TASKS:
            task_name = task_conf["task_name"]
            try:
                result = evaluate_one_task(
                    embedder_model,
                    DATASET_ROOT,
                    task_conf,
                    cache_base_dir=f"cache_{model_name}",
                    overwrite_cache=False,
                )
                model_results[task_name] = result
            except Exception as e:
                model_results[task_name] = {"error": str(e)}
                print(f"[ERROR] {model_name} {task_name}: {e}")
        infer_end = time.time()
        elapsed = infer_end - infer_start
        m, s = divmod(int(elapsed), 60)
        infer_times[model_name] = (m, s)
        print(f"[LOG] {model_name} 全タスク合計推論時間（sim作成・calib除外）: {m}min {s}sec")

        all_results[model_name] = model_results
        save_results(model_results, f"results/jmteb_multi_results_{model_name}.json")

    save_results(all_results, "results/jmteb_multi_results_ALL.json")
    # ==== ファイル出力 ====
    with open("inference_times.log", "w") as f:
        for model_name, (m, s) in infer_times.items():
            f.write(f"{model_name}\t{m}min {s}sec\n")
    print("推論時間まとめ: inference_times.log に出力しました")
    print("DONE")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset
from transformers import AutoTokenizer


def load_arrow_as_dataset(arrow_path: str) -> Dataset:
    with pa.memory_map(arrow_path, "r") as source:
        reader = ipc.RecordBatchStreamReader(source)
        table = reader.read_all()
    return Dataset(table)


def token_lengths(texts, tokenizer, count_mode="raw", max_length=512, batch_size=512):
    lengths = []
    add_special = (count_mode == "model")
    trunc = (count_mode == "model")

    #
    buf = []
    for t in texts:
        if t is None:
            continue
        buf.append(t)
        if len(buf) >= batch_size:
            enc = tokenizer(
                buf,
                add_special_tokens=add_special,
                truncation=trunc,
                max_length=max_length,
                padding=False,
                return_length=True,   
            )
            lengths.extend(enc["length"])
            buf = []

    if buf:
        enc = tokenizer(
            buf,
            add_special_tokens=add_special,
            truncation=trunc,
            max_length=max_length,
            padding=False,
            return_length=True,
        )
        lengths.extend(enc["length"])

    return np.asarray(lengths, dtype=np.int32)


def summarize(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "over_512": float((arr > 512).mean()),
        "over_1024": float((arr > 1024).mean()),
    }


def plot_hist(arr: np.ndarray, title: str, out_png: str, bin_width=16, max_plot_tokens=1024):
    if arr.size == 0:
        return
    clipped = np.minimum(arr, max_plot_tokens)
    bins = np.arange(0, max_plot_tokens + bin_width, bin_width)

    plt.figure(figsize=(10, 4))
    plt.hist(clipped, bins=bins)

    plt.title(title)
    plt.xlabel("token count")
    plt.ylabel("count")

    # 欲しい目盛りを明示
    plt.xlim(0, max_plot_tokens)
    key_ticks = [128, 256, 384,512]
    # 表示範囲内のものだけ採用
    ticks = [t for t in ([0] + key_ticks + [1024, max_plot_tokens]) if 0 <= t <= max_plot_tokens]
    plt.xticks(ticks)

    # 縦線で強調（任意）
    for t in key_ticks:
        if t <= max_plot_tokens:
            plt.axvline(t, color="r", linestyle="--", linewidth=1, alpha=0.7)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def main():
    TOKENIZER_DIR = "/root/AIMET_ENV/tools/ruri/ruri-small-v2"
    ARROW_PATH = "/root/test_qut/data/merged_test_use.arrow"
    COUNT_MODE = "raw"      # "raw" or "model"
    MAX_LENGTH = 512        
    OUT_DIR = "."  # 出力先ディレクトリ
    OUT_FILE="200_exe.png"
    os.makedirs(OUT_DIR, exist_ok=True)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_DIR,
        local_files_only=True,
        trust_remote_code=True,
    )

    # ===== data =====
    q_ds = load_arrow_as_dataset(ARROW_PATH)
    print("Query columns:", q_ds.column_names)

    q_texts = q_ds["query"] if "query" in q_ds.column_names else q_ds["text"]  
    q_lens = token_lengths(
        q_texts,
        tokenizer,
        count_mode=COUNT_MODE,
        max_length=MAX_LENGTH,
    )

    # save CSV
    pd.DataFrame({"tokens": q_lens}).to_csv(os.path.join(OUT_DIR, "query_token_counts.csv"), index=False)

    # save hist
    plot_hist(q_lens, f"Query token histogram ({COUNT_MODE})", os.path.join(OUT_DIR, OUT_FILE))

    # summary
    summary = {
        "count_mode": COUNT_MODE,
        "max_length": MAX_LENGTH,
        "query": summarize(q_lens),
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(" Saved to:", OUT_DIR)
    print("summary:", summary["query"])

if __name__ == "__main__":
    main()
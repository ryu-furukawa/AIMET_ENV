import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


# ---------- Arrow IO ----------

def load_arrow_as_dataset(path: str) -> Dataset:
    with pa.memory_map(path, "r") as source:
        reader = ipc.RecordBatchStreamReader(source)
        table = reader.read_all()
    return Dataset(table)

def save_dataset_as_arrow(ds: Dataset, out_path: str):
    table = ds.with_format("arrow")[:]  # 実体化
    with pa.OSFile(out_path, "wb") as sink:
        with ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)


# ---------- Column normalize ----------

def to_text_only(ds: Dataset, src_col: str, dst_col: str = "text") -> Dataset:
    """
    src_col を dst_col に寄せ、dst_col 以外の列は落とす（= textに集約）
    """
    if src_col != dst_col:
        ds = ds.rename_column(src_col, dst_col)

    drop_cols = [c for c in ds.column_names if c != dst_col]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds


# ---------- Main (NO TRIM) ----------

def take_two_and_merge(
    arrow_a,
    arrow_b,
    out_arrow,
    text_col,        # A側元列名（例: "query"）
    text_col2,       # B側元列名（例: "text"）
    take_each=500,
    seed=0,
):

    # --- load ---
    ds_a = load_arrow_as_dataset(arrow_a)
    ds_b = load_arrow_as_dataset(arrow_b)

    # --- take N each (random) ---
    n_a = min(100, len(ds_a))
    n_b = min(100, len(ds_b))

    ds_a_sel = ds_a.shuffle(seed=seed).select(range(n_a))
    ds_b_sel = ds_b.shuffle(seed=seed + 1).select(range(n_b))

    # --- text に集約 ---
    ds_a_txt = to_text_only(ds_a_sel, src_col=text_col,  dst_col="text")
    ds_b_txt = to_text_only(ds_b_sel, src_col=text_col2, dst_col="text")

    # --- merge ---
    merged = concatenate_datasets([ds_a_txt, ds_b_txt])

    # --- save ---
    save_dataset_as_arrow(merged, out_arrow)

    # --- sanity check ---
    out_ds = load_arrow_as_dataset(out_arrow)
    print("A IN:", len(ds_a), "A TAKE:", len(ds_a_sel))
    print("B IN:", len(ds_b), "B TAKE:", len(ds_b_sel))
    print("OUT rows:", len(out_ds))
    print("OUT columns:", out_ds.column_names)


if __name__ == "__main__":
    take_two_and_merge(
        arrow_a="/root/jmteb_260326/nlp_journal_abs_article-query/test/data-00000-of-00001.arrow",
        arrow_b="/root/jmteb_260326/nlp_journal_abs_article-corpus/corpus/data-00000-of-00001.arrow",
        out_arrow="/root/AIMET_ENV/tools/ruri/data/mix200.arrow",
        text_col="query",
        text_col2="text",
        take_each=500,
        seed=0,
    )

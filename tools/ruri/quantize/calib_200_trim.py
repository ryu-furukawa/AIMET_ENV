import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset


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
    src_col を dst_col に寄せ、dst_col 以外の列は落とす
    """
    if src_col != dst_col:
        ds = ds.rename_column(src_col, dst_col)

    drop_cols = [c for c in ds.column_names if c != dst_col]
    if drop_cols:
        ds = ds.remove_columns(drop_cols)
    return ds


# ---------- Main ----------

def take_one_dataset(
    in_arrow,
    out_arrow,
    text_col,      # 元列名 例: "query"
    take_n=200,
    seed=0,
):
    # --- load ---
    ds = load_arrow_as_dataset(in_arrow)

    # --- take N (random) ---
    n = min(take_n, len(ds))
    ds_sel = ds.shuffle(seed=seed).select(range(n))

    # --- text に集約 ---
    ds_txt = to_text_only(ds_sel, src_col=text_col, dst_col="text")

    # --- save ---
    save_dataset_as_arrow(ds_txt, out_arrow)

    # --- sanity check ---
    out_ds = load_arrow_as_dataset(out_arrow)
    print("IN rows:", len(ds))
    print("TAKE rows:", len(ds_sel))
    print("OUT rows:", len(out_ds))
    print("OUT columns:", out_ds.column_names)


if __name__ == "__main__":
    take_one_dataset(
        in_arrow="/root/jmteb_260326/nlp_journal_abs_article-query/test/data-00000-of-00001.arrow",
        out_arrow="/root/AIMET_ENV/tools/ruri/data/200.arrow",
        text_col="query",
        take_n=200,
        seed=0,
    )
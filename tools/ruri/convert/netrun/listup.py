#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

RAW_DIR = "./raw"                     # ★rawファイルがあるディレクトリ
OUT_FILE = "./input_list.txt"         # ★作りたいリスト

def main():
    raw_dir = Path(RAW_DIR).resolve()
    if not raw_dir.exists():
        raise SystemExit(f"[ERROR] RAW ディレクトリが見つかりません: {raw_dir}")

    # index 番号を自動検出
    # 例: 0_input_ids.raw → index = 0
    ids_files = sorted(raw_dir.glob("*_input_ids.raw"))

    if not ids_files:
        raise SystemExit("[ERROR] *_input_ids.raw が見つかりません。")

    lines = []
    for f in ids_files:
        idx = f.name.split("_")[0]  # "0_input_ids.raw" → "0"
        f_ids  = raw_dir / f"{idx}_input_ids.raw"
        f_mask = raw_dir / f"{idx}_attention_mask.raw"
        f_token = raw_dir / f"{idx}_token_type_ids.raw"
        f_pos = raw_dir / f"{idx}_position_ids.raw"

        ## if not (f_ids.exists() and f_mask.exists() ):
        #   raise SystemExit(f"[ERROR] index {idx} 2のファイルが揃っていません。")
        lines.append(f"{f_ids} {f_mask} {f_token} {f_pos}")

    Path(OUT_FILE).write_text("\n".join(lines), encoding="utf-8")
    print("[OK] input_list.txt を作成しました →", OUT_FILE)
    print(f"[INFO] {len(lines)} 行生成")

if __name__ == "__main__":
    main()
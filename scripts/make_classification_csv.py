#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from glob import glob

import pandas as pd


def find_modality_volume(case_root: Path, modality: str) -> str | None:
    case_root = Path(case_root)
    pattern = str(case_root / "**" / f"*{modality}.nii.gz")
    matches = sorted(glob(pattern, recursive=True))
    if not matches:
        return None
    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        type=str,
        default="metadata_with_labels.csv",
        help="Input metadata CSV with Path and Lesion columns",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="classification_index.csv",
        help="Output CSV with volume_path and label",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="T2",
        help="Modality suffix to use (e.g. T2, Pre, Post_1)",
    )
    args = parser.parse_args()

    meta_path = Path(args.meta)
    out_path = Path(args.out)

    df = pd.read_csv(meta_path)

    required_cols = ["Path", "Lesion"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in {meta_path}")

    records = []

    for _, row in df.iterrows():
        case_root = row["Path"]
        label = row["Lesion"]

        if pd.isna(label):
            continue

        vol_path = find_modality_volume(case_root, args.modality)
        if vol_path is None:
            continue

        records.append(
            {
                "volume_path": vol_path,
                "label": int(label),
            }
        )

    if not records:
        raise RuntimeError("No valid samples found. Check paths and modality name.")

    out_df = pd.DataFrame.from_records(records)
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(out_df.head())


if __name__ == "__main__":
    main()

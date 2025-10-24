#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download ODELIA-Challenge-2025 to a specific folder and export NIfTI volumes.

Features
- Choose config: unilateral (default) or default
- Choose split: val/train/test or auto (prefer val)
- Export ALL available modalities (Image_* with matching Affine_*)
- Optional exclusion of institutions (e.g., RSH)
- Save structure: <outdir>/<config>/<split>/<Institution>/<UID>/<Modality>.nii.gz
- Produce metadata.csv with per-modality availability flags
"""

from pathlib import Path
import argparse
from typing import List, Dict
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
import torchio as tio
import pandas as pd

REPO_ID = "ODELIA-AI/ODELIA-Challenge-2025"
VALID_SPLITS = ["val", "train", "test"]

def choose_split(config: str, wanted: str):
    """
    If wanted == 'auto', try val -> validation -> train -> test (in this dataset it's val/train/test).
    Returns chosen split string.
    """
    if wanted != "auto":
        return wanted
    # try 'val' first, then fallback
    for sp in ["val", "train", "test"]:
        try:
            _ = load_dataset(REPO_ID, name=config, split=sp, streaming=True)
            return sp
        except Exception:
            continue
    raise RuntimeError("No available split found. Tried: val/train/test.")

def list_modalities(sample: Dict) -> List[str]:
    """Return list of modality names like ['T2', 'Post_1', ...] derived from Image_* keys present."""
    mods = []
    for k in sample.keys():
        if k.startswith("Image_"):
            name = k.split("Image_")[1]
            aff_key = f"Affine_{name}"
            if aff_key in sample:  # ensure there is an affine
                mods.append(name)
    return mods

def save_modality(item: Dict, modality: str, out_dir: Path) -> bool:
    """Save one modality as NIfTI. Returns True if saved."""
    img_key, aff_key = f"Image_{modality}", f"Affine_{modality}"
    vol_np = item.get(img_key, None)
    aff_np = item.get(aff_key, None)
    if vol_np is None or aff_np is None:
        return False
    vol = np.array(vol_np, dtype=np.float32)
    aff = np.array(aff_np, dtype=np.float64)
    out_dir.mkdir(parents=True, exist_ok=True)
    tio.ScalarImage(tensor=vol, affine=aff).save(out_dir / f"{modality}.nii.gz")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="unilateral", choices=["unilateral", "default"],
                    help="HuggingFace dataset configuration")
    ap.add_argument("--split", default="auto", choices=["auto", "val", "train", "test"],
                    help="Which split to download. 'auto' will prefer 'val'.")
    ap.add_argument("--outdir", default=r"F:\odelia_data", type=str,
                    help="Output root directory")
    ap.add_argument("--exclude", nargs="*", default=[], help="Institutions to exclude, e.g. --exclude RSH")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="If >0, stop after saving this number of samples (for quick testing).")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    # choose split
    split = choose_split(args.config, args.split)
    print(f"[INFO] Dataset: {REPO_ID}, config={args.config}, split={split}")
    ds = load_dataset(REPO_ID, name=args.config, split=split)  # regular (non-streaming) for safe numpy conversion

    total = len(ds)
    print(f"[INFO] Total samples in split '{split}': {total}")

    # Prepare output and metadata
    save_root = out_root / args.config / split
    save_root.mkdir(parents=True, exist_ok=True)
    meta_records = []

    # Determine union of modalities by peeking at first few samples
    probe = ds[0]
    all_mods = list_modalities(probe)
    if not all_mods:
        # fallback: scan first 10 items
        for i in range(min(10, total)):
            all_mods = sorted(set(all_mods) | set(list_modalities(ds[i])))
    print(f"[INFO] Detected modalities: {all_mods}")

    errors = 0
    saved = 0
    for idx in tqdm(range(total), desc="Downloading & exporting", unit="case"):
        if args.max_samples and saved >= args.max_samples:  # noqa
            break
        try:
            item = ds[idx]
            uid = item.get("UID", f"idx{idx:05d}")
            inst = item.get("Institution", "NA")
            fold = item.get("Fold", None)
            split_name = item.get("Split", split)

            # exclusion
            if inst in set(args.exclude):
                continue

            case_dir = save_root / inst / uid
            case_dir.mkdir(parents=True, exist_ok=True)

            avail = {}
            saved_any = False
            for m in all_mods:
                ok = save_modality(item, m, case_dir)
                avail[f"has_{m}"] = bool(ok)
                saved_any = saved_any or ok

            # If nothing saved (weird), skip metadata line
            if saved_any:
                meta = {
                    "UID": uid,
                    "Institution": inst,
                    "Split": split_name,
                    "Fold": fold,
                    "Path": str(case_dir)
                }
                meta.update(avail)
                meta_records.append(meta)
                saved += 1
        except Exception as e:
            errors += 1
            print(f"[WARN] Failed at index {idx}: {type(e).__name__}: {e}")

    # Write metadata
    if meta_records:
        df = pd.DataFrame(meta_records)
        csv_path = save_root / "metadata.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[INFO] Wrote metadata: {csv_path} (rows={len(df)})")

    print(f"\n[SUMMARY] saved_cases={saved} | errors={errors} | out_dir={save_root}")

if __name__ == "__main__":
    main()

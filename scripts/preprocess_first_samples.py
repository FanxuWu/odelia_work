#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess first N ODELIA cases:
- pick specific modalities (e.g., Pre, T2, Post_1)
- resample to 1.2mm isotropic
- z-score normalization (clip to [-5,5])
- crop/pad to 256x256x128 (voxels)
- save preprocessed NIfTI + a mid-slice PNG for quick QC

Usage example:
python F:\odelia_work\scripts\preprocess_first_samples.py ^
  --root F:\odelia_data --config unilateral --split val ^
  --modalities Pre T2 Post_1 --limit 8 ^
  --outdir F:\odelia_data\preproc
"""

from pathlib import Path
import argparse
from typing import List
import numpy as np
import pandas as pd
import imageio.v2 as imageio
import nibabel as nib
import torchio as tio

def collect_cases(root: Path, config: str, split: str, modalities: List[str], limit: int):
    base = root / config / split
    cases = []
    for inst_dir in sorted(base.glob("*")):
        if not inst_dir.is_dir():
            continue
        for uid_dir in sorted(inst_dir.glob("*")):
            if not uid_dir.is_dir():
                continue
            mod_map = {}
            for m in modalities:
                p = uid_dir / f"{m}.nii.gz"
                if p.exists():
                    mod_map[m] = p
            if mod_map:
                cases.append((uid_dir, mod_map))
            if limit and len(cases) >= limit:
                return cases
    return cases

def build_transform():
    return tio.Compose([
        tio.Resample((1.2, 1.2, 1.2)),
        tio.ZNormalization(),
        tio.Clamp(out_min=-5, out_max=5),
        tio.CropOrPad((256, 256, 128), padding_mode=0),
    ])

def save_qc_png(nifti_path: Path, png_path: Path):
    img = nib.load(str(nifti_path))
    data = img.get_fdata()
    z = data.shape[2] // 2
    sl = data[:, :, z].astype(np.float32)
    if np.max(sl) > np.min(sl):
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    sl = np.clip(sl * 255.0, 0, 255).astype(np.uint8)
    imageio.imwrite(str(png_path), sl)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--config", type=str, default="unilateral")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--modalities", nargs="+", default=["Pre","T2","Post_1"])
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--outdir", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.root)
    outroot = Path(args.outdir) / args.config / args.split
    qc_dir = outroot / "_qc_png"
    outroot.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    cases = collect_cases(root, args.config, args.split, args.modalities, args.limit)
    print(f"[INFO] Found {len(cases)} cases to preprocess.")

    tfm = build_transform()
    rows = []

    for uid_dir, mod_map in cases:
        uid = uid_dir.name
        inst = uid_dir.parent.name
        case_out_dir = outroot / inst / uid
        case_out_dir.mkdir(parents=True, exist_ok=True)

        for mod, p in mod_map.items():
            print(f"[PROC] {inst}/{uid}  {mod}")
            img = tio.ScalarImage(p.as_posix())
            proc: tio.ScalarImage = tfm(img)

            out_nii = case_out_dir / f"{mod}_preproc.nii.gz"
            proc.save(out_nii.as_posix())

            png = qc_dir / f"{inst}__{uid}__{mod}.png"
            save_qc_png(out_nii, png)

            rows.append({
                "Institution": inst,
                "UID": uid,
                "Modality": mod,
                "in_path": str(p),
                "out_path": str(out_nii),
                "qc_png": str(png),
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(outroot / "preproc_index.csv", index=False, encoding="utf-8")
        print(f"[DONE] Saved index: {outroot/'preproc_index.csv'}")
    else:
        print("[WARN] No outputs. Check modalities or paths.")

if __name__ == "__main__":
    main()

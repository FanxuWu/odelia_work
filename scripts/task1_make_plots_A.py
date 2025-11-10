import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

_LABEL_MAP_INT = {0: "malignant", 1: "benign", 2: "no lesion"}

def normalize_label(x):
    try:
        i = int(float(str(x).strip()))
        if i in _LABEL_MAP_INT:
            return _LABEL_MAP_INT[i]
    except Exception:
        pass
    s = str(x).strip().lower()
    if "malig" in s:
        return "malignant"
    if "benig" in s:
        return "benign"
    if ("no" in s and "lesion" in s) or s == "nolesion":
        return "no lesion"
    if s in {"0", "1", "2"}:
        return _LABEL_MAP_INT[int(s)]
    return s

REQUIRED_GROUPS: Dict[str, List[str]] = {
    "Pre–Post2": ["has_Pre", "has_Post_1", "has_T2"],
    "Pre–Post4": ["has_Pre", "has_Post_1", "has_Post_2", "has_Post_3", "has_Post_4"],
}

def row_has_all_modalities(row: pd.Series, modalities: List[str]) -> bool:
    return all(bool(row.get(m, False)) for m in modalities)

def autodetect_label_col(df: pd.DataFrame) -> str:
    lowered = {c: c.lower().strip().replace(" ", "").replace("\u00a0","") for c in df.columns}
    priority_exact = [
        "lesion","lesion_class","lesionclass","label","class","target",
        "gt","groundtruth","y","category"
    ]
    for k in priority_exact:
        for orig, low in lowered.items():
            if low == k:
                return orig
    keywords = ["lesion","label","class","target","gt","truth","category"]
    for kw in keywords:
        for orig, low in lowered.items():
            if kw in low:
                return orig
    return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label-col", default="")
    args = ap.parse_args()

    meta_path = Path(args.meta)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(meta_path, encoding="utf-8-sig")

    label_col = args.label_col.strip()
    if label_col == "":
        label_col = autodetect_label_col(df)
    if label_col == "" or label_col not in df.columns:
        raise ValueError(f"Label column not found. Available columns: {list(df.columns)}")

    df["Lesion_Normalized"] = df[label_col].map(normalize_label)

    group_masks: Dict[str, pd.Series] = {}
    for gname, mods in REQUIRED_GROUPS.items():
        mask = df.apply(lambda r: row_has_all_modalities(r, mods), axis=1)
        group_masks[gname] = mask

    group_counts = {g: int(mask.sum()) for g, mask in group_masks.items()}

    fig1 = plt.figure(figsize=(10, 6), dpi=150)
    ax1 = fig1.add_subplot(111)
    xnames = list(REQUIRED_GROUPS.keys())
    xvals = [group_counts[g] for g in xnames]
    ax1.bar(xnames, xvals, color=["#1f77b4", "#ff7f0e"])
    ax1.set_title("Modality Coverage\n(Count of cases meeting the required combination)", fontsize=16)
    ax1.set_ylabel("Cases")
    for i, v in enumerate(xvals):
        ax1.text(i, v + max(1, v * 0.01), str(v), ha="center", va="bottom", fontsize=10)
    fig1.tight_layout()
    fig1.savefig(out_dir / "modality_distribution.png")
    plt.close(fig1)

    classes = ["malignant","benign","no lesion"]
    class_counts = {g: {c: 0 for c in classes} for g in REQUIRED_GROUPS.keys()}
    for gname, mask in group_masks.items():
        sub = df[mask]
        vc = sub["Lesion_Normalized"].value_counts()
        for c in classes:
            class_counts[gname][c] = int(vc.get(c, 0))

    fig2 = plt.figure(figsize=(10, 6), dpi=150)
    ax2 = fig2.add_subplot(111)
    idx = np.arange(len(classes))
    width = 0.36
    g1, g2 = xnames
    vals_g1 = [class_counts[g1][c] for c in classes]
    vals_g2 = [class_counts[g2][c] for c in classes]
    ax2.bar(idx - width/2, vals_g1, width=width, label=g1)
    ax2.bar(idx + width/2, vals_g2, width=width, label=g2)
    ax2.set_xticks(idx)
    ax2.set_xticklabels(classes)
    ax2.set_ylabel("Cases")
    ax2.set_title("Class Distribution", fontsize=16)
    ax2.legend(loc="upper right")
    for i, v in enumerate(vals_g1):
        ax2.text(i - width/2, v + max(1, v * 0.01), str(v), ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(vals_g2):
        ax2.text(i + width/2, v + max(1, v * 0.01), str(v), ha="center", va="bottom", fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_dir / "class_distribution.png")
    plt.close(fig2)

    export_cols = [c for c in df.columns if c.startswith("has_") or c in ["UID","Institution","Split","Fold","Path"]]
    export_cols.append("Lesion_Normalized")
    for gname, mask in group_masks.items():
        sub = df.loc[mask, export_cols]
        out_csv = out_dir / f"task1_index_{gname.replace('–','-').replace('—','-').replace(' ', '_')}.csv"
        sub.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print("[OK] Saved:", out_dir / "modality_distribution.png")
    print("[OK] Saved:", out_dir / "class_distribution.png")

if __name__ == "__main__":
    main()

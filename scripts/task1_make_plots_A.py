# F:\odelia_work\scripts\task1_make_plots_A.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 不使用 seaborn；单图单轴；不指定颜色（符合你的绘图规范）

def load_hf_labels():
    from datasets import load_dataset
    ds = load_dataset("ODELIA-AI/ODELIA-Challenge-2025", name="unilateral", split="val")
    # 尝试找标签列名
    candidates = ["Lesion","lesion","Class","class","Label","label","Diagnosis","diagnosis"]
    ds_cols = ds.features.keys()
    label_col = None
    for c in candidates:
        if c in ds_cols:
            label_col = c
            break
    if label_col is None:
        # 兜底：把所有列名打出来，方便你排查
        raise RuntimeError(f"Cannot find label column in HF dataset. Columns: {list(ds_cols)}")

    # 只取 UID 和 标签列
    # 有些字段可能是 Arrow 类型，转成 pandas 更稳妥
    df = ds.to_pandas()
    labels = df[["UID", label_col]].copy()
    labels.rename(columns={label_col: "Lesion"}, inplace=True)
    return labels

def normalize_label(x: str) -> str:
    s = str(x).strip().lower()
    if "malig" in s:
        return "malignant"
    if "benig" in s:
        return "benign"
    if ("no" in s and "lesion" in s) or s in {"none","nolession","no_lesion"}:
        return "no lesion"
    return s  # 保留原值，万一有其他写法

def bool_mask(df, cols):
    m = np.ones(len(df), dtype=bool)
    for c in cols:
        # 将 '0/1', 'True/False', 0/1 统一成布尔
        v = df[c]
        if v.dtype == bool:
            b = v
        else:
            b = v.astype(str).str.lower().isin(["1","true","t","yes"])
        m &= b.values
    return m

def main(meta_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读 metadata.csv
    df = pd.read_csv(meta_csv)

    # 2) 从 HF 取 UID+Lesion 并合并
    hf_labels = load_hf_labels()
    merged = df.merge(hf_labels, on="UID", how="left")

    # 保存一份带标签的 metadata，方便留档与复现
    merged_csv = out_dir.parent / "metadata_with_labels.csv"
    merged.to_csv(merged_csv, index=False)
    print("[INFO] Saved merged metadata:", merged_csv)

    # 3) 取两组模态组合
    group_pre_post2 = ["has_Pre","has_Sub_1","has_T2","has_Post_1","has_Post_2"]
    group_pre_post4 = group_pre_post2 + ["has_Post_3","has_Post_4"]

    m2 = bool_mask(merged, group_pre_post2)
    m4 = bool_mask(merged, group_pre_post4)

    g2 = merged[m2].copy()
    g4 = merged[m4].copy()

    # 4) 图1：Modality Coverage（满足组合的病例数）
    plt.figure(figsize=(9,5))
    plt.bar(["Pre–Post2","Pre–Post4"], [len(g2), len(g4)])
    plt.ylabel("Cases")
    plt.title("Modality Coverage\n(Count of cases meeting the required combination)")
    plt.tight_layout()
    mod_png = out_dir / "modality_distribution.png"
    plt.savefig(mod_png, dpi=160)
    plt.close()
    print("[INFO] Saved:", mod_png)

    # 5) 图2：Class Distribution（有真实标签）
    # 统一标签写法
    g2["Lesion_norm"] = g2["Lesion"].map(normalize_label)
    g4["Lesion_norm"] = g4["Lesion"].map(normalize_label)

    # 按指定顺序显示（其余类别也会显示，防止漏）
    order = ["malignant","benign","no lesion"]
    cats = list(dict.fromkeys(order + sorted(set(g2["Lesion_norm"].dropna().unique()) |
                                            set(g4["Lesion_norm"].dropna().unique()))))

    g2_counts = g2["Lesion_norm"].value_counts()
    g4_counts = g4["Lesion_norm"].value_counts()

    x = np.arange(len(cats))
    w = 0.35
    plt.figure(figsize=(10,6))
    plt.bar(x - w/2, [g2_counts.get(k,0) for k in cats], w, label="Pre–Post2")
    plt.bar(x + w/2, [g4_counts.get(k,0) for k in cats], w, label="Pre–Post4")
    plt.xticks(x, cats)
    plt.ylabel("Cases")
    plt.title("Class Distribution")
    plt.legend()
    plt.tight_layout()
    cls_png = out_dir / "class_distribution.png"
    plt.savefig(cls_png, dpi=160)
    plt.close()
    print("[INFO] Saved:", cls_png)

    # 6) 终端摘要
    print("\n=== SUMMARY ===")
    print(f"Pre–Post2: {len(g2)} cases")
    print(f"Pre–Post4: {len(g4)} cases")
    print("Class counts (Pre–Post2):", dict(g2_counts))
    print("Class counts (Pre–Post4):", dict(g4_counts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=str, required=True,
                        help="Path to metadata.csv")
    parser.add_argument("--out", type=str, required=True,
                        help="Output figures dir")
    args = parser.parse_args()
    main(Path(args.meta), Path(args.out))

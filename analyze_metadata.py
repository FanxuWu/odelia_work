# F:\odelia_work\analyze_metadata.py
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# 可选：需要补充 Age 时才会用到
from datasets import load_dataset

REPO_ID = "ODELIA-AI/ODELIA-Challenge-2025"

def ensure_age(df_meta: pd.DataFrame, config: str, split: str) -> pd.DataFrame:
    """
    如果 metadata.csv 里没有 Age 字段，则从 HF 数据集加载 Age 并按 UID 合并进来。
    """
    if "Age" in df_meta.columns:
        return df_meta

    print("[INFO] 'Age' not found in metadata.csv; fetching Age from HF dataset...")
    ds = load_dataset(REPO_ID, name=config, split=split)
    # 构建 UID -> Age 映射（部分样本可能没有 Age，保持 None）
    uid_age = {}
    for i in range(len(ds)):
        it = ds[i]
        uid_age[it.get("UID", f"idx{i:05d}")] = it.get("Age", None)
    age_sr = df_meta["UID"].map(uid_age)
    df_meta = df_meta.copy()
    df_meta["Age"] = age_sr
    print("[INFO] Age merged. Non-null Age count:", df_meta["Age"].notnull().sum())
    return df_meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\odelia_data", help="数据根目录（包含 config/split 子目录）")
    ap.add_argument("--config", default="unilateral", choices=["unilateral", "default"])
    ap.add_argument("--split", default="val", choices=["val", "train", "test"])
    ap.add_argument("--save-prefix", default="report", help="输出文件名前缀，如 report -> report_institutions.png 等")
    args = ap.parse_args()

    out_dir = os.path.join(args.root, args.config, args.split)
    meta_path = os.path.join(out_dir, "metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"metadata.csv not found: {meta_path}")

    print(f"[INFO] Loading metadata: {meta_path}")
    df = pd.read_csv(meta_path)

    # ------ 机构样本数 ------
    if "Institution" not in df.columns:
        raise ValueError("metadata.csv 中缺少 'Institution' 列，请确认下载脚本是否为最新版。")
    inst_counts = df["Institution"].value_counts().sort_values(ascending=False)
    print("\n[INFO] Institution counts:\n", inst_counts)

    # ------ 模态覆盖率（基于 has_* 列）------
    has_cols = [c for c in df.columns if c.startswith("has_")]
    modality_counts = {c.replace("has_", ""): int(df[c].sum()) for c in has_cols}
    modality_counts = dict(sorted(modality_counts.items(), key=lambda x: x[0]))  # 按名字排序
    print("\n[INFO] Modality coverage (cases with modality):")
    for k, v in modality_counts.items():
        print(f"  {k}: {v}")

    # ------ 如无 Age，则自动从 HF 合并 ------
    df = ensure_age(df, args.config, args.split)

    os.makedirs(out_dir, exist_ok=True)

    # ====== 图 1：机构样本分布 ======
    plt.figure(figsize=(10, 5))
    inst_counts.plot(kind="bar")
    plt.title("Samples per Institution")
    plt.xlabel("Institution")
    plt.ylabel("Number of Cases")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig1 = os.path.join(out_dir, f"{args.save_prefix}_institutions.png")
    plt.savefig(fig1, dpi=150)
    plt.close()
    print(f"[SAVE] {fig1}")

    # ====== 图 2：模态覆盖率 ======
    plt.figure(figsize=(10, 5))
    plt.bar(list(modality_counts.keys()), list(modality_counts.values()))
    plt.title("Modality Coverage (number of cases with modality)")
    plt.xlabel("Modality")
    plt.ylabel("Cases")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig2 = os.path.join(out_dir, f"{args.save_prefix}_modalities.png")
    plt.savefig(fig2, dpi=150)
    plt.close()
    print(f"[SAVE] {fig2}")

    # ====== 图 3：年龄分布（直方图）======
    if "Age" in df.columns and df["Age"].notnull().any():
        plt.figure(figsize=(8, 5))
        df["Age"].dropna().astype(float).plot(kind="hist", bins=20)
        plt.title("Age Distribution (All Cases)")
        plt.xlabel("Age")
        plt.ylabel("Count")
        plt.tight_layout()
        fig3 = os.path.join(out_dir, f"{args.save_prefix}_age_hist.png")
        plt.savefig(fig3, dpi=150)
        plt.close()
        print(f"[SAVE] {fig3}")

        # ====== 图 4：按机构的年龄箱线图 ======
        # 仅使用有年龄的样本
        df_age = df[["Institution", "Age"]].dropna()
        if not df_age.empty:
            # 将每个机构的年龄列表组成箱线图
            plt.figure(figsize=(10, 5))
            # 为了顺序与样本量一致：
            ordered_insts = inst_counts.index.tolist()
            data = [df_age[df_age["Institution"] == inst]["Age"].astype(float).values for inst in ordered_insts if
                    df_age[df_age["Institution"] == inst]["Age"].notnull().any()]
            labels = [inst for inst in ordered_insts if
                      df_age[df_age["Institution"] == inst]["Age"].notnull().any()]
            plt.boxplot(data, labels=labels, showfliers=False)
            plt.title("Age by Institution (Boxplot)")
            plt.xlabel("Institution")
            plt.ylabel("Age")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig4 = os.path.join(out_dir, f"{args.save_prefix}_age_box_by_institution.png")
            plt.savefig(fig4, dpi=150)
            plt.close()
            print(f"[SAVE] {fig4}")
    else:
        print("[WARN] Age column still missing or empty; skip age plots.")

    print("\n✅ Done. All figures saved under:", out_dir)

if __name__ == "__main__":
    main()

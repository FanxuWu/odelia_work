# F:\odelia_work\analyze_metadata_no_age.py
import os
import pandas as pd
import matplotlib.pyplot as plt

ROOT = r"F:\odelia_data"
CONFIG = "unilateral"
SPLIT = "val"

out_dir = os.path.join(ROOT, CONFIG, SPLIT)
meta_path = os.path.join(out_dir, "metadata.csv")
df = pd.read_csv(meta_path)

# 机构样本数
inst_counts = df["Institution"].value_counts().sort_values(ascending=False)

# 模态覆盖率（基于 has_* 列）
has_cols = [c for c in df.columns if c.startswith("has_")]
modality_counts = {c.replace("has_", ""): int(df[c].sum()) for c in has_cols}
modality_counts = dict(sorted(modality_counts.items(), key=lambda x: x[0]))

# 图1：机构分布
plt.figure(figsize=(10,5))
inst_counts.plot(kind="bar")
plt.title("Samples per Institution")
plt.xlabel("Institution")
plt.ylabel("Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "report_no_age_institutions.png"), dpi=150)
plt.close()

# 图2：模态覆盖率
plt.figure(figsize=(10,5))
plt.bar(list(modality_counts.keys()), list(modality_counts.values()))
plt.title("Modality Coverage (cases with modality)")
plt.xlabel("Modality")
plt.ylabel("Cases")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "report_no_age_modalities.png"), dpi=150)
plt.close()

print("✅ Done. Saved:",
      os.path.join(out_dir, "report_no_age_institutions.png"),
      "and",
      os.path.join(out_dir, "report_no_age_modalities.png"))

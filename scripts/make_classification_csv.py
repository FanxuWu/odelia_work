import pandas as pd

META = r"F:\odelia_work\metadata_with_labels.csv"
OUT = r"F:\odelia_work\classification_index.csv"

df = pd.read_csv(META)

print("Columns in metadata:", list(df.columns))

path_candidates = [c for c in df.columns if "path" in c.lower()]
if not path_candidates:
    raise ValueError(f"No path-like column found. Columns: {list(df.columns)}")
path_col = path_candidates[0]
print("Using path column:", path_col)

label_candidates = [c for c in df.columns if any(k in c.lower() for k in ["label", "lesion", "class"])]
if not label_candidates:
    raise ValueError(f"No label-like column found. Columns: {list(df.columns)}")
label_col = label_candidates[0]
print("Using label column:", label_col)

df = df[df[label_col].notna()].copy()

df_out = df[[path_col, label_col]].copy()
df_out.rename(columns={path_col: "volume_path", label_col: "label"}, inplace=True)

df_out.to_csv(OUT, index=False)

print("Saved:", OUT)
print(df_out.head())

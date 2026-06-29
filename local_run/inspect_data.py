import io, numpy as np, pandas as pd
from PIL import Image

df = pd.read_parquet("arasl.parquet")
print("rows:", len(df))
print("columns:", list(df.columns))
print("dtypes:\n", df.dtypes)

# label column
lab_col = "label" if "label" in df.columns else df.columns[-1]
labs = df[lab_col]
print("unique labels:", labs.nunique())
print("label value counts (head):\n", labs.value_counts().sort_index().head(40))

# decode one image
img_col = "image" if "image" in df.columns else df.columns[0]
v = df[img_col].iloc[0]
if isinstance(v, dict):
    b = v.get("bytes", None)
    print("image column is dict with keys:", list(v.keys()))
else:
    b = v
im = Image.open(io.BytesIO(b)).convert("L")
print("sample image size:", im.size, "mode:", im.mode)
print("TOTAL == 54049 ?", len(df) == 54049, "| 32 classes ?", labs.nunique() == 32)

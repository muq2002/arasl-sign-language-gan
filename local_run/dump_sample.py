import io, os, pandas as pd
from PIL import Image
out = r"C:\Users\mktad\OneDrive\Desktop\Testing\local_run\dataset_sample"
os.makedirs(out, exist_ok=True)
df = pd.read_parquet("arasl.parquet")
saved = 0
for c in sorted(df["label"].unique())[:8]:        # 8 classes
    rows = df[df["label"] == c].head(5)            # 5 images each
    for j, v in enumerate(rows["image"]):
        b = v["bytes"] if isinstance(v, dict) else v
        Image.open(io.BytesIO(b)).convert("L").save(os.path.join(out, f"class{c:02d}_{j}.png"))
        saved += 1
print(f"wrote {saved} sample PNGs to:\n{out}")

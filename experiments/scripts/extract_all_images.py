# -*- coding: utf-8 -*-
"""Extract ALL images from the Hugging Face parquet into class-named folders,
mirroring the original ArASL_Database_54K_Final layout. Run from this folder."""
import io, os, json, time
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

# data lives in project_root/data; this script is in experiments/scripts/
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
PARQUET = os.path.join(DATA, "arasl.parquet")
OUT = os.path.join(DATA, "ArASL_dataset")
os.makedirs(OUT, exist_ok=True)

# try to recover the real class names from HF ClassLabel metadata in the parquet
names = None
try:
    md = pq.read_schema(PARQUET).metadata or {}
    hf = md.get(b"huggingface")
    if hf:
        feat = json.loads(hf).get("info", {}).get("features", {})
        lab = feat.get("label", {})
        names = lab.get("names") or lab.get("_type") and lab.get("names")
except Exception as e:
    print("metadata read note:", e)
if not names:   # fallback to the documented ArASL2018 class list (alphabetical)
    names = ["ain","al","aleff","bb","dal","dha","dhad","fa","gaaf","ghain","ha","haa",
             "jeem","kaaf","khaa","la","laam","meem","nun","ra","saad","seen","sheen",
             "ta","taa","thaa","thal","toot","waw","ya","yaa","zay"]
print(f"using {len(names)} class names -> {names[:5]} ...")

df = pd.read_parquet(PARQUET)
for n in names:
    os.makedirs(os.path.join(OUT, n), exist_ok=True)

t0 = time.time(); counts = {}
for i, (v, lab) in enumerate(zip(df["image"], df["label"])):
    b = v["bytes"] if isinstance(v, dict) else v
    cls = names[int(lab)] if int(lab) < len(names) else f"class_{int(lab):02d}"
    k = counts.get(cls, 0); counts[cls] = k + 1
    Image.open(io.BytesIO(b)).convert("L").save(os.path.join(OUT, cls, f"{cls}_{k:04d}.png"))
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(df)} written ({time.time()-t0:.0f}s)")

total = sum(counts.values())
print(f"DONE: wrote {total} images across {len(counts)} classes in {time.time()-t0:.0f}s")
print("per-class:", {k: counts[k] for k in sorted(counts)})
json.dump(counts, open(os.path.join(OUT, "_counts.json"), "w"), indent=2)
print("location:", OUT)

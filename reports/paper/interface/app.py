"""
End-user interface — generate Arabic Sign-Language hand signs from the trained
models (A, B, C, F, G). Pick a letter, pick a model, click Generate.

- Models A / B: class-conditioned. Generate directly from noise + label.
- Models C / F / G: structure-conditioned. We pick a random REAL image of the
  chosen letter from the dataset, compute its structure map (Canny+silhouette+
  distance), and generate a NEW hand that follows that structure. (Click Generate
  again for a different structure / style.) G is the best model (94.6% recognition).

Loads whichever generators have been exported to:
  outputs/<run>/checkpoints/export/generator.keras
Run:  python reports/paper/interface/app.py     (opens a local web UI)
Requires: gradio  (pip install gradio)
"""
import os, json, glob
import numpy as np
import tensorflow as tf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUTS = os.environ.get("ARASL_OUT", os.path.join(REPO, "outputs"))
DATA = os.environ.get("ARASL_DATA", os.path.join(REPO, "data", "ArASL_dataset"))

# the exported generators embed a custom SelfAttention2D layer -> make it importable
import sys
sys.path.insert(0, os.path.join(REPO, "src"))
from models import SelfAttention2D
Z_DIM, IMG_SIZE, COND_CH = 128, 128, 3
CANNY_LO, CANNY_HI = 60, 160

RUNS = {
    "A — pixel loss only": os.path.join(OUTPUTS, "cgan_A_128"),
    "B — + MediaPipe landmark loss": os.path.join(OUTPUTS, "cgan_B_128mp"),
    "C — structure-conditioned": os.path.join(OUTPUTS, "cgan_C_128struct"),
    "F — + landmark fusion": os.path.join(OUTPUTS, "cgan_F_128fusion"),
    "G — recognition-optimized (best · 94.6%)": os.path.join(OUTPUTS, "cgan_G_128plus"),
}


def _load(run_base):
    exp = os.path.join(run_base, "checkpoints", "export")
    kpath = os.path.join(exp, "generator.keras")
    if not os.path.exists(kpath):
        return None
    G = tf.keras.models.load_model(kpath, compile=False,
                                   custom_objects={"SelfAttention2D": SelfAttention2D})
    with open(os.path.join(exp, "class_labels.json")) as f:
        labels = json.load(f)
    with open(os.path.join(exp, "inference_config.json")) as f:
        cfg = json.load(f)
    return {"G": G, "idx_to_label": {int(k): v for k, v in labels["idx_to_label"].items()},
            "label_to_idx": labels["label_to_idx"], "structure": cfg.get("conditioned_on_structure", False)}


MODELS = {name: m for name, base in RUNS.items() if (m := _load(base))}


def structure_map(img_norm):
    import cv2
    g = ((img_norm[:, :, 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)
    edge = cv2.Canny(g, CANNY_LO, CANNY_HI)
    _, sil = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if sil.mean() > 127:
        sil = 255 - sil
    dist = cv2.normalize(cv2.distanceTransform(sil, cv2.DIST_L2, 3), None, 0, 255, cv2.NORM_MINMAX)
    return (np.stack([edge, sil, dist], -1).astype(np.float32) / 127.5) - 1.0


def _real_image(letter):
    import cv2
    folder = os.path.join(DATA, letter)
    files = glob.glob(os.path.join(folder, "*.png"))
    if not files:
        return None
    f = files[np.random.randint(len(files))]
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return (img.astype(np.float32) - 127.5) / 127.5


def generate(model_name, letter, n):
    m = MODELS[model_name]
    ci = m["label_to_idx"][letter]
    oh = tf.one_hot([ci] * n, len(m["idx_to_label"]))
    nz = tf.random.normal([n, Z_DIM])
    if m["structure"]:
        ref = _real_image(letter)
        if ref is None:
            return []
        cond = np.stack([structure_map(ref[..., None])] * n).astype(np.float32)
        fake = m["G"]([tf.convert_to_tensor(cond, tf.float32), oh, nz], training=False).numpy()
    else:
        fake = m["G"]([nz, oh], training=False).numpy()
    return [(fake[i, :, :, 0] * 127.5 + 127.5).clip(0, 255).astype(np.uint8) for i in range(n)]


def build_ui():
    import gradio as gr
    if not MODELS:
        print("No exported models found. Train + export first."); return None
    any_labels = next(iter(MODELS.values()))["label_to_idx"]
    letters = sorted(any_labels.keys())
    names = list(MODELS.keys())
    default_model = next((m for m in names if m.startswith("G")), names[-1])
    with gr.Blocks(title="ArASL Sign Generator") as demo:
        gr.Markdown(
            "# 🤟 Arabic Sign-Language Generator\n"
            "Generate 128×128 hand signs for any of the **32 Arabic letters** from five trained "
            "models. **Model G is the best (94.6% recognition).** For the structure-conditioned "
            "models (C / F / G) every click samples a new real structure → a fresh pose — click "
            "**Generate** again for more variety.")
        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Dropdown(names, value=default_model, label="Model")
                letter = gr.Dropdown(letters, value=letters[0], label="Letter")
                n = gr.Slider(1, 8, value=4, step=1, label="How many samples")
                btn = gr.Button("✨ Generate", variant="primary", size="lg")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Generated signs", columns=4,
                                     height=440, object_fit="contain")
        gr.Markdown("**Model ladder:** A (label only) · B (+MediaPipe) · C (structure-conditioned) "
                    "· F (+landmark fusion) · **G (recognition-optimized — best)**")
        btn.click(generate, [model, letter, n], gallery)
    return demo


if __name__ == "__main__":
    import gradio as gr
    demo = build_ui()
    if demo:
        demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=False,
                    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="slate"))

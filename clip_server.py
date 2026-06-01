"""
CLIP Backend: receives cropped bird image (base64), returns top-3 matching cards.
POST /match  { image: "base64..." }  ->  { matches: [{category, text, score}, ...] }
"""

import io, json, base64, os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

app = Flask(__name__)
CORS(app)

# ---- Load CLIP ----
print("Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()
print("CLIP model loaded.")

# ---- Load cards ----
cards_path = os.path.join(os.path.dirname(__file__), "public", "cards.json")
with open(cards_path, "r", encoding="utf-8") as f:
    cards = json.load(f)

all_texts = []
card_meta = []
for category, texts in cards.items():
    for i, text in enumerate(texts):
        all_texts.append(f"a photo of {text}")
        card_meta.append((category, i, text))

# Precompute text embeddings
print(f"Encoding {len(all_texts)} card texts...")
with torch.no_grad():
    text_inputs = processor(text=all_texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
    text_embeds = model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
print("Text embeddings ready.")


@app.route("/match", methods=["POST"])
def match():
    data = request.get_json()
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "no image"}), 400

    try:
        img_bytes = base64.b64decode(img_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"bad image: {e}"}), 400

    with torch.no_grad():
        img_inputs = processor(images=img, return_tensors="pt")
        img_embed = model.get_image_features(**img_inputs)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

    sims = (img_embed @ text_embeds.T).squeeze(0)
    top3 = sims.topk(min(3, len(all_texts)))

    results = []
    for idx, score in zip(top3.indices, top3.values):
        cat, _, text = card_meta[idx.item()]
        results.append({
            "category": cat,
            "text": text,
            "score": round(score.item(), 4),
        })

    return jsonify({"matches": results})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "cards_loaded": len(all_texts)})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5007))
    print(f"Starting CLIP server on port {port}...")
    app.run(host="0.0.0.0", port=port)

import warnings
import logging
import os

# Suppress all warnings before any library imports
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

from transformers import CLIPProcessor, CLIPModel
from PIL import Image          
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"  

model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32",
    ignore_mismatched_sizes=True
).to(device)
processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32",
    use_fast=True
)

def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    if hasattr(features, "pooler_output"):
        features = features.pooler_output

    return features.squeeze().cpu().numpy()  

def process_folder(folder):
    embeddings = []

    for file in os.listdir(folder):
        if file.endswith(".jpg"):
            path = os.path.join(folder, file)
            emb = get_embedding(path)
            embeddings.append(emb)

    return embeddings

KEYFRAMES_DIR = r"c:\Users\khare\Desktop\Perceptual fingerprinting\keyframes"

orig_embeds    = process_folder(os.path.join(KEYFRAMES_DIR, "original"))
pirated_embeds = process_folder(os.path.join(KEYFRAMES_DIR, "pirated_lowres"))

print("Original frames:", len(orig_embeds))
print("Pirated frames:", len(pirated_embeds))

def cosine_sim(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))


def check_piracy(orig_embeds, pirated_embeds):
    matches = 0
    total = min(len(orig_embeds), len(pirated_embeds))

    if total == 0:
        print("No frames to compare")
        return

    for i in range(total):
        sim = cosine_sim(orig_embeds[i], pirated_embeds[i])
        
        if sim > 0.92:
            matches += 1

    score = matches / total

    print("Match ratio:", score)

    if score > 0.5:
        print("PIRATED")
    else:
        print("NOT PIRATED")
check_piracy(orig_embeds, pirated_embeds)
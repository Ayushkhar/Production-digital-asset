import numpy as np
import faiss
from s2_embedings import orig_embeds, pirated_embeds

def temporal(embeds, window=5, stride=1):
    if len(embeds) < window:
        return np.empty((0, len(embeds[0])), dtype='float32')

    temp = []

    for i in range(0, len(embeds) - window + 1, stride):
        segment = embeds[i:i+window]
        avg_vec = np.mean(segment, axis=0)
        temp.append(avg_vec)

    return np.array(temp, dtype='float32')

orig_temp = temporal(orig_embeds, window=5, stride=1)
pir_temp  = temporal(pirated_embeds, window=5, stride=1)

print("Temporal shapes:", orig_temp.shape, pir_temp.shape)

if len(orig_temp) == 0 or len(pir_temp) == 0:
    print("Not enough frames for temporal matching")
    exit()

dim = orig_temp.shape[1]
faiss.normalize_L2(orig_temp)
faiss.normalize_L2(pir_temp)

index = faiss.IndexFlatIP(dim)
index.add(orig_temp)

D, I = index.search(pir_temp, k=1)

similarity = D

THRESHOLD = 0.85
temporal_matches = (similarity > THRESHOLD).sum()
temporal_ratio = temporal_matches / len(similarity)

print("Temporal ratio:", temporal_ratio)


if temporal_ratio > 0.4:
    print("TEMPORAL: PIRATED")
else:
    print("TEMPORAL: NOT PIRATED")
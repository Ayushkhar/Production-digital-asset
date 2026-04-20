import faiss
import numpy as np
from s2_embedings import orig_embeds, pirated_embeds   

# Stack into matrices
orig_matrix    = np.vstack(orig_embeds).astype('float32')
pirated_matrix = np.vstack(pirated_embeds).astype('float32')

faiss.normalize_L2(orig_matrix)
faiss.normalize_L2(pirated_matrix)

dimension = orig_matrix.shape[1]

index = faiss.IndexFlatIP(dimension)
index.add(orig_matrix)
print("Stored vectors:", index.ntotal)

D, I = index.search(pirated_matrix, k=1)

similarities = D.flatten()
print("Cosine similarities (sample):", similarities[:5].round(4))

THRESHOLD = 0.80
matches = (similarities > THRESHOLD).sum()
ratio = matches / len(similarities)

print(f"Match ratio: {ratio:.4f}  ({matches}/{len(similarities)} frames matched)")

if ratio > 0.4:
    print("PIRATED")
else:
    print("NOT PIRATED")
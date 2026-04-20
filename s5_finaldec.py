from s3_faiss import ratio as visual_ratio
from s4_temporal_ratio import temporal_ratio
from s6_audiofingerprinting import similarity as audio_similarity


print("----- LAYER SCORES -----")
print("Visual Score:   ", visual_ratio)
print("Temporal Score: ", temporal_ratio)
print("Audio Score:    ", audio_similarity)


final_score = (
    0.5 * visual_ratio +
    0.3 * temporal_ratio +
    0.2 * audio_similarity
)

print("\nFinal Score:", final_score)


# 🔹 Decision
THRESHOLD = 0.75

if final_score > THRESHOLD:
    print("FINAL: PIRATED")
else:
    print("FINAL: NOT PIRATED")
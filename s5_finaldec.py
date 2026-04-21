from s3_faiss import ratio as visual_ratio
from s4_temporal_ratio import temporal_ratio
from s6_audiofingerprinting import similarity as audio_similarity
from s7_ocr import ocr_score

print("----- LAYER SCORES -----")
print("Visual Score:   ", visual_ratio)
print("Temporal Score: ", temporal_ratio)
print("Audio Score:    ", audio_similarity)
print("OCR Score:    ", ocr_score)


final_score = (
    0.4 * visual_ratio +
    0.25 * temporal_ratio +
    0.2 * audio_similarity +
    0.15 * ocr_score
)

print("\nFinal Score:", final_score)

THRESHOLD = 0.75

if final_score > THRESHOLD:
    print("FINAL: PIRATED")
else:
    print("FINAL: NOT PIRATED")
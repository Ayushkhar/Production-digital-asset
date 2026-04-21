import cv2
import pytesseract
import os

# Point pytesseract to the Tesseract binary (default Windows install path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

KEYFRAME_DIR = r"C:\Users\khare\Desktop\Perceptual fingerprinting\keyframes\pirated_crop"

SUSPICIOUS_WORDS = [
    "free", "movie", "telegram", "download",
    "watch", "link", ".com", ".in"
]


def extract_text(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        text = pytesseract.image_to_string(gray)
    except pytesseract.TesseractNotFoundError:
        print("[WARNING] Tesseract not found. OCR score will be 0.")
        print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")
        return ""
    return text.lower()


matches = 0
total = 0

for file in os.listdir(KEYFRAME_DIR):
    if file.endswith(".jpg"):
        path = os.path.join(KEYFRAME_DIR, file)

        text = extract_text(path)
        total += 1

        if any(word in text for word in SUSPICIOUS_WORDS):
            matches += 1


ocr_score = matches / total if total > 0 else 0

print("OCR Score:", ocr_score)


# decision
if ocr_score > 0.3:
    print("OCR: SUSPICIOUS OVERLAY DETECTED")
else:
    print("OCR: CLEAN")
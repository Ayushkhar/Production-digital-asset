"""
Step 1: Keyframe Extraction
Extracts frames at a fixed time interval (every 2 seconds) from each video.
Stores frames in keyframes/<video_name>/frame_0.jpg, frame_1.jpg, ...
Validates each folder has >= 2 frames and no mixed file types.
"""

import cv2
import os
import shutil

VIDEO_DIR = r"c:\Users\khare\Desktop\Perceptual fingerprinting"
OUTPUT_BASE_DIR = os.path.join(VIDEO_DIR, "keyframes")

VALID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
FRAME_INTERVAL_SEC = 0.5
MIN_FRAMES = 2


def clear_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def extract_keyframes(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"  FPS: {fps:.2f} | Duration: {duration:.1f}s | Total frames: {total_frames}")

    interval = max(1, int(fps * FRAME_INTERVAL_SEC))
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    return saved


def validate_folder(folder):
    files = os.listdir(folder)
    jpgs = [f for f in files if f.lower().endswith(".jpg")]
    others = [f for f in files if not f.lower().endswith(".jpg")]

    if others:
        print(f"  [WARN] Mixed file types found: {others} — removing them")
        for f in others:
            os.remove(os.path.join(folder, f))

    if len(jpgs) < MIN_FRAMES:
        print(f"  [FAIL] Only {len(jpgs)} frame(s) — minimum is {MIN_FRAMES}")
        return False

    return True


def main():
    print("=" * 55)
    print("  Step 1: Keyframe Extraction")
    print("=" * 55)

    video_files = [
        f for f in os.listdir(VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
    ]

    if not video_files:
        print("[ERROR] No video files found in:", VIDEO_DIR)
        return

    print(f"Found {len(video_files)} video(s): {video_files}\n")

    summary = []

    for video_file in video_files:
        video_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(VIDEO_DIR, video_file)
        output_dir = os.path.join(OUTPUT_BASE_DIR, video_name)

        print(f"[{video_name}]")
        clear_folder(output_dir)

        count = extract_keyframes(video_path, output_dir)
        print(f"  Extracted: {count} frames")

        ok = validate_folder(output_dir)

        if not ok:
            shutil.rmtree(output_dir)
            summary.append((video_name, "FAILED validation — folder removed"))
        else:
            summary.append((video_name, f"{count} frames [OK]"))

        print()

    print("=" * 55)
    print("  Summary")
    print("=" * 55)
    for name, result in summary:
        print(f"  {name:<25} {result}")
    print("=" * 55)
    print("\nDone. Keyframes saved to:", OUTPUT_BASE_DIR)


if __name__ == "__main__":
    main()

import cv2
import os
print("Script startyed")
def extract_keyframes(video_path, output_folder, interval_sec=2):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = int(fps * interval_sec)

    os.makedirs(output_folder, exist_ok=True)

    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            filename = f"{output_folder}/frame_{saved}.jpg"
            cv2.imwrite(filename, frame)
            saved += 1

        count += 1

    cap.release()
    print(f"Done: {video_path} → {saved} frames")
extract_keyframes("original.mp4", "frames_original")
extract_keyframes("pirated_crop.mp4", "frames_pirated_crop")
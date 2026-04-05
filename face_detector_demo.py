import json
import os
from glob import glob

import cv2
from dotenv import load_dotenv

load_dotenv()

DATASET_ROOT = os.getenv("DATASET_ROOT", "")


def visualize():
    video_path = sorted(glob(os.path.join(DATASET_ROOT, "original", "*.mp4")))[0]
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    boxes_path = os.path.join(DATASET_ROOT, "boxes", f"{video_id}.json")

    with open(boxes_path, "r") as f:
        bboxes = json.load(f)

    capture = cv2.VideoCapture(video_path)
    # find first frame that has a detected face
    for frame_idx, bbox in bboxes.items():
        if bbox is not None:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = capture.read()
            if not success:
                continue
            # bboxes were detected at half resolution, so scale by 2
            for box in bbox:
                x1, y1, x2, y2 = [int(b * 2) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            out_path = os.path.join(DATASET_ROOT, "face_detection_sample.png")
            cv2.imwrite(out_path, frame)
            print(f"Saved to: {out_path}")
            break
    capture.release()


if __name__ == "__main__":
    visualize()

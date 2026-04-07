import argparse
import os

import cv2

def extract(video_path, num_frames=16):
    capture = cv2.VideoCapture(video_path)
    print(video_path)
    print(capture)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(total_frames)
    print(f"Total frames in video: {total_frames}")

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]

    out_dir = os.path.join(os.path.dirname(video_path), "jpegs")
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    for i, frame_idx in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            print(f"  Frame {frame_idx} failed to read")
            continue
        out_path = os.path.join(out_dir, f"test_{i:04d}.jpg")
        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
        print(f"  Saved {out_path}")
        saved += 1

    capture.release()
    print(f"\nDone. {saved}/{num_frames} frames saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="path to a single video file")
    parser.add_argument("--num-frames", type=int, default=16)
    args = parser.parse_args()
    extract(args.video, args.num_frames)

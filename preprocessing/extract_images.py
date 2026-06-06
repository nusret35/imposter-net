import argparse
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from functools import partial
from glob import glob
from multiprocessing.pool import Pool

import cv2
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
from tqdm import tqdm


def extract_video(video, root_dir, num_frames):
    capture = cv2.VideoCapture(video)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        capture.release()
        return

    if total_frames >= num_frames:
        indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    else:
        indices = list(range(total_frames))

    # Include parent dir to avoid collisions (e.g. Face2Face_944_032)
    parent = os.path.basename(os.path.dirname(video))
    basename = os.path.splitext(os.path.basename(video))[0]
    video_id = f"{parent}_{basename}"
    out_dir = os.path.join(root_dir, "jpegs", video_id)
    if os.path.exists(out_dir) and len(os.listdir(out_dir)) > 0:
        capture.release()
        return
    os.makedirs(out_dir, exist_ok=True)

    for i, frame_idx in enumerate(indices):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = capture.read()
        if not success:
            continue
        cv2.imwrite(os.path.join(out_dir, "{:04d}.jpg".format(i)), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])

    capture.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts evenly-spaced jpegs from video")
    parser.add_argument("--root-dir", help="root directory containing videos")
    parser.add_argument("--output-dir", help="directory to save extracted frames")
    parser.add_argument("--num-frames", type=int, default=16, help="number of frames to extract per video")

    args = parser.parse_args()
    output_dir = args.output_dir or args.root_dir
    os.makedirs(os.path.join(output_dir, "jpegs"), exist_ok=True)
    videos = [video_path for video_path in glob(os.path.join(args.root_dir, "**", "*.mp4"), recursive=True)]
    print(f"Found {len(videos)} videos in {args.root_dir}")
    with Pool(processes=4) as p:
        with tqdm(total=len(videos)) as pbar:
            for v in p.imap_unordered(partial(extract_video, root_dir=output_dir, num_frames=args.num_frames), videos):
                pbar.update()

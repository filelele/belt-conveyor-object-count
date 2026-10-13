import os
import sys
import argparse
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

import cv2


VIDEO_EXTENSIONS = {
    ".avi", ".mp4", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"
}


def find_video_files(source_dir: Path):
    """
    Recursively find all video files at any depth inside source_dir.
    """
    video_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in VIDEO_EXTENSIONS:
                video_files.append(Path(root) / file)
    return sorted(video_files)


def extract_frames_from_video(video_path: Path, output_dir: Path, force: bool = False, frame_format: str = "frame{:04d}.jpg"):
    """
    Extracts all frames from video_path and saves them into output_dir.
    Frame naming: frame0001.jpg, frame0002.jpg, ... (1-indexed).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing_frames = list(output_dir.glob("frame*.jpg"))
    if existing_frames and not force:
        print(f"Skipping {video_path} (already has {len(existing_frames)} frames in {output_dir})")
        return len(existing_frames)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}", file=sys.stderr)
        return 0

    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        frame_filename = frame_format.format(frame_count)
        frame_path = output_dir / frame_filename
        cv2.imwrite(str(frame_path), frame)

    cap.release()
    return frame_count


def convert_all_videos(source_dir: str, dest_dir: str, force: bool = False):
    """
    Finds all videos in source_dir and extracts frames into dest_dir,
    preserving relative directory structure plus video name.
    
    Example:
      source: raw_data/A/B/vid.avi
      dest:   frames/A/B/vid.avi/frame0001.jpg, ...
    """
    src_path = Path(source_dir).resolve()
    dst_path = Path(dest_dir).resolve()

    if not src_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_path}")

    videos = find_video_files(src_path)
    print(f"Found {len(videos)} video(s) in {src_path}")

    total_frames = 0
    for video in tqdm(videos, desc="Processing videos"):
        rel_path = video.relative_to(src_path)
        out_dir = dst_path / rel_path
        frames_saved = extract_frames_from_video(video, out_dir, force=force)
        total_frames += frames_saved

    print(f"\nDone! Extracted {total_frames} total frames across {len(videos)} videos into {dst_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert all videos in a source directory to frames in a destination directory."
    )
    # Default paths relative to repository structure
    default_src = Path(__file__).resolve().parent / "raw_data"
    default_dest = Path(__file__).resolve().parent / "frames"

    parser.add_argument(
        "--src",
        type=str,
        default=str(default_src),
        help=f"Path to source folder containing videos (default: {default_src})"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(default_dest),
        help=f"Path to destination folder for extracted frames (default: {default_dest})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite / re-extract frames if destination folder already exists"
    )

    args = parser.parse_args()
    convert_all_videos(source_dir=args.src, dest_dir=args.dest, force=args.force)


if __name__ == "__main__":
    main()

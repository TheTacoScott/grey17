#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Extracts a per-frame pHash sequence from the author's cooked output and writes
it into the recipe as cooked_phash_sequence.

This is the ground-truth fingerprint of what the final fan edit should look
like after conforming. Use verify-cook to compare any fan-rendered output
against this sequence.

Usage:
    python3 fingerprint_cook.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --video  /work/cooked/S01E01.mkv
"""
import argparse
import os
import sys

import yaml

from utils import ffprobe_source, extract_phashes_pipe


def parse_args():
    p = argparse.ArgumentParser(
        description="Fingerprint the author's cooked output and write cooked_phash_sequence into the recipe.",
    )
    p.add_argument("--recipe", required=True, help="Path to recipe.yaml (read/write)")
    p.add_argument("--video",  required=True, help="Path to the author's cooked output video")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    # Determine fps and expected frame count from the recipe scene/output block.
    fps = (recipe.get("output") or {}).get("fps") \
       or (recipe.get("scene") or {}).get("effective_fps") \
       or 24.0
    scene = recipe.get("scene") or {}
    frame_start  = scene.get("frame_start", 0)
    frame_end    = scene.get("frame_end", 0)
    n_frames     = frame_end - frame_start + 1
    duration_s   = scene.get("duration_seconds") or (n_frames / fps)

    video_info = ffprobe_source(args.video)
    print("Video:    {}x{}  {:.6f}fps  {:.3f}s  {}".format(
        video_info["resolution_x"], video_info["resolution_y"],
        video_info["fps"] or 0,
        video_info["duration_seconds"] or 0,
        video_info["video_codec"] or "?",
    ))
    print("Expected: {} frames at {:.6f}fps  ({:.3f}s)".format(n_frames, fps, duration_s))

    print("Extracting pHashes from cooked video...")

    def progress(n):
        tc = n / fps
        m  = int(tc // 60)
        s  = tc - m * 60
        print("  ... {} / {} frames  ({:02d}:{:05.2f})".format(n, n_frames, m, s),
              flush=True)

    hashes = extract_phashes_pipe(
        args.video, 0.0, fps,
        n_frames=n_frames,
        progress_callback=progress,
    )

    n_got = len(hashes)
    print("Extracted {} frames (expected {})".format(n_got, n_frames))

    if n_got == 0:
        print("ERROR: no frames extracted from cooked video", file=sys.stderr)
        sys.exit(1)

    if n_got < n_frames * 0.95:
        print("WARNING: got {:.1f}% of expected frames ({} / {})".format(
            100 * n_got / n_frames, n_got, n_frames))

    recipe["cooked_phash_sequence"] = hashes

    with open(args.recipe, "w") as f:
        yaml.dump(recipe, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=99999)

    print("\ncooked_phash_sequence ({} frames) written to: {}".format(
        n_got, args.recipe))


if __name__ == "__main__":
    main()

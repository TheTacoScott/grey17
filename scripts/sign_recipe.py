#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Generates fingerprint anchor points for all source files listed in a recipe.

Usage:
    python3 sign_recipe.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --source source_0=/work/sources/source_0/film.mkv
"""
import argparse
import os
import shutil
import subprocess
import sys

import yaml

from utils import compute_sha256, ffprobe_source, detect_crop, hash_frame

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fingerprint source videos and write anchor data into a recipe.",
        epilog=(
            "Captures dense pHash/aHash sequences for the first and last portion of each "
            "source at native fps. These start_anchors and end_anchors are used by "
            "'grey17 match' for sliding window endpoint alignment."
        ),
    )
    p.add_argument("--recipe", required=True, help="Path to recipe.yaml (read/write)")
    p.add_argument("--work-dir", default="/work/tmp", help="Writable scratch directory")
    p.add_argument("--source", action="append", default=[], metavar="slot_id=/path/to/file",
                   help="Map a source slot to a file (repeatable)")
    p.add_argument("--seq-fraction", type=float, default=0.10,
                   metavar="<0.0-1.0>",
                   help="Fraction of video to capture at each end (default: 0.10 = first/last 10%%).")
    p.add_argument("--seq-floor", type=int, default=1000,
                   metavar="<frames>",
                   help="Minimum frames per endpoint sequence (default: 1000).")
    p.add_argument("--seq-ceil", type=int, default=10000,
                   metavar="<frames>",
                   help="Maximum frames per endpoint sequence (default: 10000).")
    return p.parse_args()


def parse_source_args(source_args):
    """Return dict of {slot_id: filepath}."""
    mapping = {}
    for s in source_args:
        if "=" not in s:
            print("WARNING: ignoring malformed --source arg: {}".format(s), file=sys.stderr)
            continue
        slot_id, path = s.split("=", 1)
        mapping[slot_id.strip()] = path.strip()
    return mapping

# ---------------------------------------------------------------------------
# Endpoint sequence
# ---------------------------------------------------------------------------

SEQ_FRACTION_DEFAULT = 0.10
SEQ_FLOOR_DEFAULT    = 1000
SEQ_CEIL_DEFAULT     = 10000


def compute_sequence_window(duration_seconds, native_fps,
                            seq_fraction=SEQ_FRACTION_DEFAULT,
                            seq_floor=SEQ_FLOOR_DEFAULT,
                            seq_ceil=SEQ_CEIL_DEFAULT):
    """
    Compute how many seconds to capture for a start/end sequence and the frame count.
    Returns (window_seconds, frame_count).
    """
    target_frames = int(duration_seconds * seq_fraction * native_fps)
    target_frames = max(seq_floor, min(seq_ceil, target_frames))
    window_seconds = target_frames / native_fps
    return window_seconds, target_frames


def extract_endpoint_anchors(source_path, fps, start_tc, duration_secs, seq_dir):
    """
    Extract frames at native fps for a window starting at start_tc for duration_secs.
    Returns a list of {phash, ahash} dicts (one per frame), or None on failure.
    Frame 0 corresponds to start_tc; frame i corresponds to start_tc + i/fps.
    """
    os.makedirs(seq_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss", "{:.6f}".format(start_tc),
        "-i", source_path,
        "-t", "{:.6f}".format(duration_secs),
        "-vf", "scale=32:32:flags=lanczos,format=gray",
        "-vsync", "cfr",
        "-r", "{:.6f}".format(fps),
        "-f", "image2",
        os.path.join(seq_dir, "f_%08d.png"),
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: endpoint frame extraction failed", file=sys.stderr)
        return None

    anchors = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.startswith("f_") or not fname.endswith(".png"):
            continue
        ph, ah = hash_frame(os.path.join(seq_dir, fname))
        anchors.append({"phash": ph, "ahash": ah})

    if not anchors:
        return None

    return anchors

# ---------------------------------------------------------------------------
# Per-source signing
# ---------------------------------------------------------------------------

def sign_source(source, source_path, work_dir,
                seq_fraction=SEQ_FRACTION_DEFAULT,
                seq_floor=SEQ_FLOOR_DEFAULT,
                seq_ceil=SEQ_CEIL_DEFAULT):
    """
    Sign a single source file.
    Returns (source_metadata, start_anchors, end_anchors).
    start_anchors and end_anchors are lists of {phash, ahash} dicts.
    """
    source_id = source["id"]

    print("  Computing SHA256...", flush=True)
    sha256 = compute_sha256(source_path)
    print("  SHA256: {}".format(sha256), flush=True)

    print("  Running ffprobe...", flush=True)
    meta = ffprobe_source(source_path)
    meta["sha256"] = sha256
    duration = meta["duration_seconds"]
    if not duration:
        raise RuntimeError("Could not determine duration for {}".format(source_path))

    print("  Duration: {:.1f}s  {}x{}  {:.3f}fps  {}".format(
        duration,
        meta["resolution_x"] or 0, meta["resolution_y"] or 0,
        meta["fps"] or 0,
        meta["video_codec"] or "?",
    ), flush=True)

    print("  Detecting crop...", flush=True)
    crop = detect_crop(source_path, duration,
                       meta["resolution_x"] or 0, meta["resolution_y"] or 0)
    if crop:
        meta["detected_crop"] = crop
        print("  Black bars detected: content is {}x{} at offset ({},{})".format(
            crop["w"], crop["h"], crop["x"], crop["y"]), flush=True)
        print("  expect_full_frame will be set to FALSE for this source.", flush=True)
    else:
        meta["detected_crop"] = None
        print("  No black bars detected: source appears full-frame.", flush=True)

    native_fps = meta.get("fps") or 24.0
    frames_dir = os.path.join(work_dir, "{}_frames".format(source_id))
    window_secs, n_frames = compute_sequence_window(duration, native_fps,
                                                     seq_fraction, seq_floor, seq_ceil)

    print("  Extracting start anchors ({} frames, {:.1f}s at {:.3f}fps)...".format(
        n_frames, window_secs, native_fps), flush=True)
    start_anchors = extract_endpoint_anchors(
        source_path, native_fps,
        start_tc=0.0, duration_secs=window_secs,
        seq_dir=os.path.join(frames_dir, "start"),
    )

    end_start_tc = max(0.0, duration - window_secs)
    print("  Extracting end anchors ({} frames, {:.1f}s at {:.3f}fps)...".format(
        n_frames, window_secs, native_fps), flush=True)
    end_anchors = extract_endpoint_anchors(
        source_path, native_fps,
        start_tc=end_start_tc, duration_secs=window_secs,
        seq_dir=os.path.join(frames_dir, "end"),
    )

    return meta, start_anchors, end_anchors

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    source_map = parse_source_args(args.source)

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    for source in recipe.get("sources", []):
        slot_id = source["id"]
        source_path = source_map.get(slot_id)
        if not source_path:
            print("SKIP {}: no file provided".format(slot_id), flush=True)
            continue
        if not os.path.exists(source_path):
            print("SKIP {}: file not found: {}".format(slot_id, source_path), flush=True)
            continue

        print("\nSigning {} ({})".format(slot_id, os.path.basename(source_path)), flush=True)
        try:
            meta, start_anchors, end_anchors = sign_source(
                source, source_path, work_dir,
                seq_fraction=args.seq_fraction,
                seq_floor=args.seq_floor,
                seq_ceil=args.seq_ceil,
            )
        except Exception as e:
            print("ERROR signing {}: {}".format(slot_id, e), file=sys.stderr)
            continue

        source["expect_full_frame"] = meta["detected_crop"] is None
        if meta.get("detected_crop"):
            source.setdefault("original", {})
            source["original"]["detected_crop"] = meta["detected_crop"]

        orig = source.setdefault("original", {})
        orig["sha256"] = meta["sha256"]
        orig["resolution_x"] = meta["resolution_x"]
        orig["resolution_y"] = meta["resolution_y"]
        orig["fps"] = meta["fps"]
        orig["duration_seconds"] = meta["duration_seconds"]
        orig["video_codec"] = meta["video_codec"]
        orig["audio_codec"] = meta["audio_codec"]
        orig["audio_channels"] = meta["audio_channels"]
        orig["audio_sample_rate"] = meta["audio_sample_rate"]

        if start_anchors:
            source["start_anchors"] = start_anchors
            print("  start_anchors: {} frames".format(len(start_anchors)), flush=True)
        if end_anchors:
            source["end_anchors"] = end_anchors
            print("  end_anchors: {} frames".format(len(end_anchors)), flush=True)

        # Remove legacy fields if present from a previous sign run
        for key in ("anchors", "start_sequence", "end_sequence", "strip_timecodes"):
            source.pop(key, None)

        print("  Done: {} start anchors, {} end anchors".format(
            len(start_anchors) if start_anchors else 0,
            len(end_anchors) if end_anchors else 0,
        ), flush=True)

    recipe["signed"] = True

    shutil.rmtree(work_dir, ignore_errors=True)

    with open(args.recipe, "w") as f:
        yaml.dump(recipe, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=99999)

    print("\nRecipe signed and written to: {}".format(args.recipe), flush=True)


if __name__ == "__main__":
    main()

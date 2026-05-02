#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Generates fingerprint data for all source files listed in a recipe.

Signing produces:
  phash_sequence  - concatenated 16-char hex pHash strings for every frame of
                    the source, at native fps. One string covering the entire
                    video. Length = num_frames * 16 characters.
  audio_start_fp  - raw Chromaprint integers for the first AUDIO_WINDOW_SECS
                    of audio (list of ints).
  audio_end_fp    - raw Chromaprint integers for the last AUDIO_WINDOW_SECS.

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

from utils import compute_sha256, ffprobe_source, detect_crop, extract_phashes_pipe, run_fpcalc

# ---------------------------------------------------------------------------
# Audio fingerprint constants
# ---------------------------------------------------------------------------

# Seconds of audio to capture at each end of the source.
AUDIO_WINDOW_SECS = 300.0

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fingerprint source videos and write anchor data into a recipe.",
        epilog=(
            "Hashes every frame of each source at native fps (phash_sequence) and "
            "captures Chromaprint audio fingerprints for the first and last "
            "{:.0f}s. These are used by 'grey17 match' for sliding window "
            "endpoint alignment.".format(AUDIO_WINDOW_SECS)
        ),
    )
    p.add_argument("--recipe", required=True, help="Path to recipe.yaml (read/write)")
    p.add_argument("--work-dir", default="/work/tmp", help="Writable scratch directory (reserved)")
    p.add_argument("--source", action="append", default=[], metavar="slot_id=/path/to/file",
                   help="Map a source slot to a file (repeatable)")
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
# Per-source signing
# ---------------------------------------------------------------------------

def sign_source(source, source_path):
    """
    Sign a single source file.
    Returns (meta, phash_sequence, audio_start_fp, audio_end_fp).

    phash_sequence is a list of 16-char hex pHash strings, one per frame.
    audio_start_fp and audio_end_fp are lists of Chromaprint ints.
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
    n_expected = int(duration * native_fps)

    print("  Extracting pHash for all {} frames at {:.3f}fps...".format(
        n_expected, native_fps), flush=True)

    def progress(n):
        print("  ... {} / ~{} frames ({:.1f}s / {:.1f}s)...".format(
            n, n_expected, n / native_fps, duration), flush=True)

    hashes = extract_phashes_pipe(source_path, 0.0, native_fps,
                                  progress_callback=progress)
    if not hashes:
        raise RuntimeError("pHash extraction produced no frames for {}".format(source_path))

    phash_sequence = hashes  # list of 16-char hex pHash strings, one per frame
    print("  pHash complete: {} frames".format(len(phash_sequence)), flush=True)

    # Audio fingerprints
    print("  Extracting audio fingerprint (start {:.0f}s)...".format(
        AUDIO_WINDOW_SECS), flush=True)
    audio_start_fp = run_fpcalc(source_path, 0.0, AUDIO_WINDOW_SECS)
    if audio_start_fp:
        print("  audio_start_fp: {} ints ({:.1f}s)".format(
            len(audio_start_fp), len(audio_start_fp) / 86.0), flush=True)
    else:
        print("  WARNING: audio_start_fp extraction failed (fpcalc missing or audio absent?)",
              flush=True)

    end_audio_offset = max(0.0, duration - AUDIO_WINDOW_SECS)
    print("  Extracting audio fingerprint (end {:.0f}s from {:.1f}s)...".format(
        AUDIO_WINDOW_SECS, end_audio_offset), flush=True)
    audio_end_fp = run_fpcalc(source_path, end_audio_offset, AUDIO_WINDOW_SECS)
    if audio_end_fp:
        print("  audio_end_fp: {} ints ({:.1f}s)".format(
            len(audio_end_fp), len(audio_end_fp) / 86.0), flush=True)
    else:
        print("  WARNING: audio_end_fp extraction failed", flush=True)

    return meta, phash_sequence, audio_start_fp, audio_end_fp

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    source_map = parse_source_args(args.source)

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

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
            meta, phash_sequence, audio_start_fp, audio_end_fp = sign_source(source, source_path)
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

        source["phash_sequence"] = phash_sequence
        if audio_start_fp:
            source["audio_start_fp"] = audio_start_fp
        if audio_end_fp:
            source["audio_end_fp"] = audio_end_fp
        source.pop("breaks", None)

        # Remove fields from previous sign format
        for key in ("anchors", "start_anchors", "end_anchors",
                    "start_sequence", "end_sequence", "strip_timecodes"):
            source.pop(key, None)

        print("  Done: {} frames, {} audio start ints, {} audio end ints".format(
            len(phash_sequence),
            len(audio_start_fp) if audio_start_fp else 0,
            len(audio_end_fp) if audio_end_fp else 0,
        ), flush=True)

    recipe["signed"] = True

    with open(args.recipe, "w") as f:
        yaml.dump(recipe, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=99999)

    print("\nRecipe signed and written to: {}".format(args.recipe), flush=True)


if __name__ == "__main__":
    main()

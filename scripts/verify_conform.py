#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Verifies a conformed video against the recipe phash_sequence by comparing
pHashes frame-by-frame and reporting accuracy statistics.

Usage:
    python3 verify_conform.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --video /work/conformed/source_0.mkv \
        --slot source_0 \
        --sample-rate 1 \
        --output /work/out/verify_report.csv
"""
import argparse
import csv
import os
import sys

import yaml

from utils import ffprobe_source, extract_phashes_pipe, phash_distance


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare a conformed video frame-by-frame against the recipe phash_sequence.",
    )
    p.add_argument("--recipe",  required=True)
    p.add_argument("--video",   required=True, help="Path to the conformed video file")
    p.add_argument("--slot",    default="source_0",
                   help="Source slot to compare against (default: source_0)")
    p.add_argument("--sample-rate", type=int, default=1, metavar="<N>",
                   help="Compare every Nth frame (default: 1 = every frame). "
                        "Use 24 for ~1fps sampling.")
    p.add_argument("--output",  default=None, metavar="<report.csv>",
                   help="Path to write per-frame distance CSV (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    # Locate the requested slot
    source = None
    for s in recipe.get("sources", []):
        if s["id"] == args.slot:
            source = s
            break
    if source is None:
        print("ERROR: slot {} not found in recipe".format(args.slot), file=sys.stderr)
        sys.exit(1)

    phash_seq = source.get("phash_sequence")
    if not phash_seq:
        print("ERROR: recipe has no phash_sequence for slot {}. Re-sign the recipe.".format(
            args.slot), file=sys.stderr)
        sys.exit(1)

    orig = source.get("original") or {}
    fps = orig.get("fps") or 24.0
    author_duration = orig.get("duration_seconds") or 0.0
    n_recipe = len(phash_seq)

    print("Recipe:  {} frames  {:.6f}fps  {:.3f}s".format(n_recipe, fps, author_duration))

    video_info = ffprobe_source(args.video)
    print("Video:   {}x{}  {:.6f}fps  {:.3f}s  {}".format(
        video_info["resolution_x"], video_info["resolution_y"],
        video_info["fps"] or 0,
        video_info["duration_seconds"] or 0,
        video_info["video_codec"] or "?",
    ))

    sample_rate = max(1, args.sample_rate)
    n_expected = n_recipe // sample_rate

    # Extract at FULL fps (not downsampled fps) and subsample in Python.
    # Downsampling via -r fps/N in ffmpeg causes incorrect frame selection
    # when the conformed file uses B-frames (libx264 with bframes>0), because
    # B-frame reordering shifts stored PTS values by ~0.5 frames, causing the
    # cfr resampler to pick frames from the wrong position. Extracting at full
    # fps and taking every Nth hash avoids this entirely.
    n_to_extract = n_recipe

    print("Sampling every {} frame(s) (~{} comparisons)".format(sample_rate, n_expected))
    print("Extracting pHashes from conformed video at full fps ({:.6f})...".format(fps))

    def progress(n):
        tc = n / fps
        m = int(tc // 60)
        s = tc - m * 60
        print("  ... {} / ~{} frames  ({:02d}:{:05.2f} / {:.0f}s)".format(
            n, n_to_extract, m, s, author_duration), flush=True)

    all_hashes = extract_phashes_pipe(
        args.video, 0.0, fps,
        n_frames=n_to_extract,
        progress_callback=progress,
    )

    # Subsample: take every sample_rate-th hash
    conformed_hashes = all_hashes[::sample_rate]

    n_got = len(conformed_hashes)
    print("  Extracted {} frames, subsampled to {}".format(len(all_hashes), n_got))

    if n_got == 0:
        print("ERROR: no frames extracted from conformed video", file=sys.stderr)
        sys.exit(1)

    # Compare each extracted frame against the corresponding recipe entry
    print("Comparing to recipe phash_sequence...")
    distances = []
    rows = []  # (recipe_frame_index, author_tc, dist)

    for i, conf_hash in enumerate(conformed_hashes):
        recipe_idx = i * sample_rate
        if recipe_idx >= n_recipe:
            break
        dist = phash_distance(conf_hash, phash_seq[recipe_idx])
        author_tc = recipe_idx / fps
        distances.append(dist)
        rows.append((recipe_idx, author_tc, dist))

    n = len(distances)
    if n == 0:
        print("ERROR: no frames to compare", file=sys.stderr)
        sys.exit(1)

    # Summary statistics
    mean_dist   = sum(distances) / n
    sorted_d    = sorted(distances)
    median_dist = sorted_d[n // 2]
    p95_dist    = sorted_d[int(n * 0.95)]
    p99_dist    = sorted_d[int(n * 0.99)]
    max_dist    = max(distances)
    pct_under5  = sum(1 for d in distances if d < 5.0)  / n
    pct_under10 = sum(1 for d in distances if d < 10.0) / n

    # Verdict: based on mean and p95
    if mean_dist < 5.0 and p95_dist < 10.0:
        verdict = "PASS"
    elif mean_dist < 10.0 and p95_dist < 20.0:
        verdict = "WARN"
    else:
        verdict = "FAIL"

    print()
    print("=== Verify Results ({}) ===".format(verdict))
    print("Frames compared : {}".format(n))
    print("Mean distance   : {:.2f}".format(mean_dist))
    print("Median distance : {:.2f}".format(median_dist))
    print("p95 distance    : {:.2f}".format(p95_dist))
    print("p99 distance    : {:.2f}".format(p99_dist))
    print("Max distance    : {:.2f}".format(max_dist))
    print("Within 5.0      : {:.1%}  ({} frames)".format(pct_under5,  int(pct_under5  * n)))
    print("Within 10.0     : {:.1%}  ({} frames)".format(pct_under10, int(pct_under10 * n)))

    # Distance histogram (buckets: 0-2, 2-5, 5-10, 10-20, 20+)
    buckets = [0, 2, 5, 10, 20, 65]
    labels  = ["0-2", "2-5", "5-10", "10-20", "20+"]
    counts  = [0] * len(labels)
    for d in distances:
        for j in range(len(buckets) - 1):
            if d < buckets[j + 1]:
                counts[j] += 1
                break
    print()
    print("Distribution:")
    for label, count in zip(labels, counts):
        bar = "#" * int(40 * count / n)
        print("  {:>5}  {:5}  {}".format(label, count, bar))

    # Worst frames
    worst = sorted(rows, key=lambda r: r[2], reverse=True)[:20]
    print()
    print("Worst 20 frames (highest distance):")
    for frame_num, author_tc, dist in worst:
        m = int(author_tc // 60)
        s = author_tc - m * 60
        print("  frame {:6d}  {:02d}:{:06.3f}  dist={:.1f}".format(frame_num, m, s, dist))

    # Optional per-frame CSV report
    if args.output:
        with open(args.output, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["frame_number", "author_tc_secs", "dist"])
            for frame_num, author_tc, dist in rows:
                writer.writerow([frame_num, "{:.6f}".format(author_tc), dist])
        print()
        print("Per-frame report written to: {}".format(args.output))

    sys.exit(0 if verdict != "FAIL" else 1)


if __name__ == "__main__":
    main()

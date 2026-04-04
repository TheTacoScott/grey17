#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Matches viewer-provided source files against a signed recipe, detects
timing offset and speed drift, and writes a conform plan.

Usage:
    python3 match_recipe.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --output /work/out/edit.conform.yaml \
        --slot source_0=/work/candidates/source_0/film.mkv
"""
import argparse
import os
import shutil
import subprocess
import sys
import datetime

import yaml
from PIL import Image
import imagehash

from utils import compute_sha256, ffprobe_source, phash_distance, detect_crop

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Match viewer files against a signed recipe and produce a conform plan.",
        epilog=(
            "Matching uses a sliding window search over the first and last "
            "--search-fraction of the viewer file to locate the author's start and end "
            "anchor sequences. Offset and speed are derived from those two matched points."
        ),
    )
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", required=True, help="Path to write conform plan YAML")
    p.add_argument("--slot", action="append", default=[], metavar="slot_id=/path/to/file")
    p.add_argument("--threshold", type=float, default=0.65,
                   metavar="<0.0-1.0>",
                   help="Minimum match quality to consider a file suitable (default: 0.65). "
                        "Derived from start/end probe pHash distances.")
    p.add_argument("--search-fraction", type=float, default=0.25,
                   metavar="<0.0-1.0>",
                   help="Fraction of viewer duration to search at each end (default: 0.25). "
                        "Increase if the viewer has long pre-roll or extended credits.")
    p.add_argument("--probe-frames", type=int, default=120,
                   metavar="<frames>",
                   help="Sliding window probe size in frames (default: 120 = ~5s at 24fps).")
    p.add_argument("--work-dir", default="/work/tmp")
    return p.parse_args()


def parse_slot_args(slot_args):
    mapping = {}
    for s in slot_args:
        if "=" not in s:
            continue
        slot_id, path = s.split("=", 1)
        mapping[slot_id.strip()] = path.strip()
    return mapping

# ---------------------------------------------------------------------------
# Sliding window endpoint matching
# ---------------------------------------------------------------------------

VIEWER_SEARCH_FRACTION_DEFAULT = 0.25
PROBE_FRAMES_DEFAULT = 120
PROBE_SKIP_FRAMES = 24  # skip first ~1s to avoid fade-in black


def extract_viewer_endpoint_frames(viewer_path, start_tc, duration_secs, fps, seq_dir, crop=None):
    """
    Extract viewer frames at the given fps for a search window.
    If crop is a dict {w, h, x, y}, it is applied before scaling to strip black bars.
    Returns list of {phash, ahash} dicts (one per frame), or empty list on failure.
    """
    os.makedirs(seq_dir, exist_ok=True)
    if crop:
        vf = "crop={w}:{h}:{x}:{y},scale=32:32:flags=lanczos,format=gray".format(**crop)
    else:
        vf = "scale=32:32:flags=lanczos,format=gray"
    cmd = [
        "ffmpeg",
        "-ss", "{:.6f}".format(start_tc),
        "-i", viewer_path,
        "-t", "{:.6f}".format(duration_secs),
        "-vf", vf,
        "-vsync", "cfr",
        "-r", "{:.6f}".format(fps),
        "-f", "image2",
        os.path.join(seq_dir, "f_%08d.png"),
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  WARNING: viewer frame extraction failed", file=sys.stderr)
        return []

    anchors = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.startswith("f_") or not fname.endswith(".png"):
            continue
        try:
            img = Image.open(os.path.join(seq_dir, fname)).convert("L")
            anchors.append({
                "phash": str(imagehash.phash(img)),
                "ahash": str(imagehash.average_hash(img)),
            })
        except Exception:
            anchors.append({"phash": "0" * 16, "ahash": "0" * 16})
    return anchors


def sliding_window_search(probe, search):
    """
    Slide probe (list of {phash, ahash}) over search (list of {phash, ahash}).
    Score at each position = mean pHash Hamming distance (lower = better).
    Returns (best_position_index, best_mean_distance).
    best_position_index is the index into search where probe[0] best aligns.
    """
    n_probe = len(probe)
    n_search = len(search)

    if n_probe == 0 or n_search < n_probe:
        return 0, 64.0

    best_pos = 0
    best_score = 64.0

    for i in range(n_search - n_probe + 1):
        total = sum(phash_distance(probe[j]["phash"], search[i + j]["phash"])
                    for j in range(n_probe))
        mean_dist = total / n_probe
        if mean_dist < best_score:
            best_score = mean_dist
            best_pos = i

    return best_pos, best_score


def endpoint_match(source, viewer_path, viewer_duration, work_dir,
                   search_fraction=VIEWER_SEARCH_FRACTION_DEFAULT,
                   probe_frames=PROBE_FRAMES_DEFAULT,
                   viewer_crop=None):
    """
    Find where the author's start and end anchors appear in the viewer file.

    Uses fps from source["original"]["fps"] and derives anchor timecodes implicitly:
      start_anchors[0] corresponds to t=0.0 in the author's source
      end_anchors[0] corresponds to t=(duration - len(end_anchors)/fps)

    viewer_crop: optional {w, h, x, y} crop applied to viewer frames before hashing,
    to strip pillarbox/letterbox bars that would otherwise distort the phash comparison.

    Returns dict with start_match_tc, end_match_tc, start_score, end_score,
    author_start_tc, author_end_tc. Or None if anchors are missing.
    """
    start_anchors = source.get("start_anchors")
    end_anchors = source.get("end_anchors")
    if not start_anchors or not end_anchors:
        return None

    orig = source.get("original") or {}
    fps = orig.get("fps") or 24.0
    author_duration = orig.get("duration_seconds") or viewer_duration
    slot_id = source["id"]

    # Build probes, skipping initial frames to avoid fade-in black at the start
    skip = min(PROBE_SKIP_FRAMES, len(start_anchors) // 4)
    start_probe = start_anchors[skip: skip + probe_frames]

    # End probe: take probe_frames from near the end, leaving skip frames at the tail
    end_tail = max(0, len(end_anchors) - skip)
    end_probe_start = max(0, end_tail - probe_frames)
    end_probe = end_anchors[end_probe_start: end_tail]
    if not end_probe:
        end_probe = end_anchors[-probe_frames:]

    # Author timecodes for the probe windows
    author_start_tc = skip / fps

    # end_anchors starts at: author_duration - len(end_anchors)/fps
    end_anchors_start_tc = max(0.0, author_duration - len(end_anchors) / fps)
    author_end_tc = end_anchors_start_tc + end_probe_start / fps

    search_duration = viewer_duration * search_fraction

    # -- Start search: first search_fraction of viewer --
    print("  Extracting viewer start search region ({:.0f}s)...".format(
        search_duration), flush=True)
    start_search = extract_viewer_endpoint_frames(
        viewer_path, 0.0, search_duration, fps,
        os.path.join(work_dir, "{}_start_search".format(slot_id)),
        crop=viewer_crop,
    )
    if not start_search:
        return None

    print("  Sliding window search (start): {} probe vs {} viewer frames...".format(
        len(start_probe), len(start_search)), flush=True)
    start_pos, start_score = sliding_window_search(start_probe, start_search)
    viewer_start_tc = start_pos / fps
    print("  Start match: viewer_tc={:.3f}s  mean_dist={:.2f}".format(
        viewer_start_tc, start_score), flush=True)

    # -- End search: last search_fraction of viewer --
    end_search_start = max(0.0, viewer_duration - search_duration)
    print("  Extracting viewer end search region ({:.0f}s from {:.0f}s)...".format(
        search_duration, end_search_start), flush=True)
    end_search = extract_viewer_endpoint_frames(
        viewer_path, end_search_start, search_duration, fps,
        os.path.join(work_dir, "{}_end_search".format(slot_id)),
        crop=viewer_crop,
    )
    if not end_search:
        return None

    print("  Sliding window search (end): {} probe vs {} viewer frames...".format(
        len(end_probe), len(end_search)), flush=True)
    end_pos, end_score = sliding_window_search(end_probe, end_search)
    viewer_end_tc = end_search_start + end_pos / fps
    print("  End match: viewer_tc={:.3f}s  mean_dist={:.2f}".format(
        viewer_end_tc, end_score), flush=True)

    return {
        "start_match_tc": viewer_start_tc,
        "end_match_tc": viewer_end_tc,
        "start_score": round(start_score, 4),
        "end_score": round(end_score, 4),
        "author_start_tc": author_start_tc,
        "author_end_tc": author_end_tc,
    }

# ---------------------------------------------------------------------------
# Trim computation
# ---------------------------------------------------------------------------

def compute_trim_points(source, offset, speed_factor, viewer_duration):
    """
    Compute where to trim the viewer file.
    Author content spans from t=0 to t=original.duration_seconds.
    Maps those endpoints through the linear model: t_viewer = t_author * speed_factor + offset.
    """
    orig = source.get("original") or {}
    author_duration = orig.get("duration_seconds") or viewer_duration

    viewer_content_start = offset  # t_author=0
    viewer_content_end = author_duration * speed_factor + offset

    trim_start = max(0.0, viewer_content_start)
    trim_end = min(viewer_duration, viewer_content_end)
    trim_duration = max(0.0, trim_end - trim_start)

    return {
        "trim_start_seconds": round(trim_start, 6),
        "trim_duration_seconds": round(trim_duration, 6),
    }

# ---------------------------------------------------------------------------
# Main match loop
# ---------------------------------------------------------------------------

def match_slot(source, viewer_path, viewer_info, work_dir, threshold,
               search_fraction=VIEWER_SEARCH_FRACTION_DEFAULT,
               probe_frames=PROBE_FRAMES_DEFAULT):
    """
    Match a single viewer file against a recipe source slot.
    Returns a result dict for inclusion in the conform plan.
    """
    slot_id = source["id"]
    viewer_duration = viewer_info["duration_seconds"]
    orig = source.get("original", {})

    # SHA256 shortcut: exact file match bypasses all fingerprint work
    recipe_sha256 = orig.get("sha256")
    if recipe_sha256:
        print("  Computing SHA256...", flush=True)
        viewer_sha256 = compute_sha256(viewer_path)
        if viewer_sha256 == recipe_sha256:
            print("  SHA256 match - exact file, skipping fingerprint.", flush=True)
            trim = compute_trim_points(source, 0.0, 1.0, viewer_duration)
            transform = {
                "offset_seconds": 0.0,
                "speed_factor": 1.0,
                "trim_start_seconds": trim["trim_start_seconds"],
                "trim_duration_seconds": trim["trim_duration_seconds"],
                "fps_in": viewer_info["fps"],
                "fps_out": orig.get("fps"),
                "resolution_in": [viewer_info["resolution_x"], viewer_info["resolution_y"]],
                "resolution_out": [orig.get("resolution_x"), orig.get("resolution_y")],
            }
            return {
                "slot_id": slot_id,
                "slot_name": source.get("name", ""),
                "status": "suitable",
                "match_rate": 1.0,
                "match_method": "sha256",
                "input_file": viewer_path,
                "transform": transform,
                "output_filename": orig.get("filename", "{}_conformed.mkv".format(slot_id)),
            }
        print("  SHA256 mismatch - proceeding with fingerprint matching.", flush=True)

    if not source.get("start_anchors") or not source.get("end_anchors"):
        print("  ERROR: recipe has no start/end anchors for {}. Re-sign the recipe.".format(slot_id),
              file=sys.stderr)
        return {"slot_id": slot_id, "status": "no_match", "match_rate": 0.0}

    # Detect crop before matching so bars are stripped from viewer frames.
    # Only done when the recipe marks this source as full-frame: if the author's
    # signed anchors include bars, cropping the viewer would break the comparison.
    viewer_crop = None
    if source.get("expect_full_frame"):
        print("  Detecting viewer crop...", flush=True)
        viewer_crop = detect_crop(viewer_path, viewer_duration,
                                  viewer_info["resolution_x"], viewer_info["resolution_y"])
        if viewer_crop:
            print("  Crop detected: {}x{} at ({},{}) - applying during matching.".format(
                viewer_crop["w"], viewer_crop["h"], viewer_crop["x"], viewer_crop["y"]), flush=True)
        else:
            print("  No crop detected.", flush=True)

    ep = endpoint_match(source, viewer_path, viewer_duration, work_dir,
                        search_fraction=search_fraction,
                        probe_frames=probe_frames,
                        viewer_crop=viewer_crop)
    if ep is None:
        print("  ERROR: endpoint match returned no result", file=sys.stderr)
        return {"slot_id": slot_id, "status": "no_match", "match_rate": 0.0}

    author_span = ep["author_end_tc"] - ep["author_start_tc"]
    viewer_span = ep["end_match_tc"] - ep["start_match_tc"]

    if author_span > 0 and viewer_span > 0:
        final_speed = viewer_span / author_span
    else:
        final_speed = 1.0
    final_speed = max(0.90, min(1.10, final_speed))

    final_offset = ep["start_match_tc"] - ep["author_start_tc"] * final_speed

    start_quality = max(0.0, 1.0 - ep["start_score"] / 32.0)
    end_quality   = max(0.0, 1.0 - ep["end_score"]   / 32.0)
    match_rate = (start_quality + end_quality) / 2.0

    print("  Endpoint match: offset={:.3f}s  speed={:.6f}  match_rate={:.1%}".format(
        final_offset, final_speed, match_rate), flush=True)

    suitable = match_rate >= threshold
    trim = compute_trim_points(source, final_offset, final_speed, viewer_duration)

    transform = {
        "offset_seconds": round(final_offset, 6),
        "speed_factor": round(final_speed, 8),
        "trim_start_seconds": trim["trim_start_seconds"],
        "trim_duration_seconds": trim["trim_duration_seconds"],
        "fps_in": viewer_info["fps"],
        "fps_out": orig.get("fps"),
        "resolution_in": [viewer_info["resolution_x"], viewer_info["resolution_y"]],
        "resolution_out": [orig.get("resolution_x"), orig.get("resolution_y")],
    }

    if viewer_crop and suitable:
        transform["crop"] = viewer_crop

    return {
        "slot_id": slot_id,
        "slot_name": source.get("name", ""),
        "status": "suitable" if suitable else "unsuitable",
        "match_rate": round(match_rate, 4),
        "match_method": "endpoint",
        "input_file": viewer_path,
        "transform": transform,
        "output_filename": orig.get("filename", "{}_conformed.mkv".format(slot_id)),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    slot_map = parse_slot_args(args.slot)

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    results = []
    all_suitable = True

    for source in recipe.get("sources", []):
        slot_id = source["id"]
        viewer_path = slot_map.get(slot_id)

        if not viewer_path:
            print("\nSKIP {}: no candidate file provided".format(slot_id), flush=True)
            results.append({"slot_id": slot_id, "status": "missing"})
            all_suitable = False
            continue

        if not os.path.exists(viewer_path):
            print("\nSKIP {}: file not found: {}".format(slot_id, viewer_path), flush=True)
            results.append({"slot_id": slot_id, "status": "missing"})
            all_suitable = False
            continue

        print("\nMatching {} against {}".format(
            slot_id, os.path.basename(viewer_path)), flush=True)

        try:
            viewer_info = ffprobe_source(viewer_path)
            print("  Viewer: {}x{}  {:.3f}fps  {:.1f}s  {}".format(
                viewer_info["resolution_x"], viewer_info["resolution_y"],
                viewer_info["fps"] or 0,
                viewer_info["duration_seconds"] or 0,
                viewer_info["video_codec"] or "?",
            ), flush=True)

            result = match_slot(source, viewer_path, viewer_info, work_dir, args.threshold,
                                search_fraction=args.search_fraction,
                                probe_frames=args.probe_frames)
        except Exception as e:
            print("  ERROR: {}".format(e), file=sys.stderr)
            result = {"slot_id": slot_id, "status": "error", "error": str(e)}

        if result:
            results.append(result)
            status = result.get("status", "?")
            rate = result.get("match_rate", 0)
            print("  -> {} (match_rate={:.1%})".format(status, rate), flush=True)
            if status != "suitable":
                all_suitable = False
        else:
            results.append({"slot_id": slot_id, "status": "error"})
            all_suitable = False

    # Clean up work dir
    shutil.rmtree(work_dir, ignore_errors=True)

    # Write conform plan
    conform_plan = {
        "grey17_version": recipe.get("grey17_version", 1),
        "recipe_version": recipe.get("recipe_version", "1.0"),
        "matched_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "all_suitable": all_suitable,
        "scene": recipe.get("scene", {}),
        "output": recipe.get("output", {}),
        "sources": results,
    }

    with open(args.output, "w") as f:
        yaml.dump(conform_plan, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=99999)

    print("\nConform plan written to: {}".format(args.output), flush=True)
    print("Overall status: {}".format("SUITABLE" if all_suitable else "NOT SUITABLE"), flush=True)


if __name__ == "__main__":
    main()

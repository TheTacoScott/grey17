#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Matches viewer-provided source files against a signed recipe, detects
timing offset and speed drift, and writes a conform plan.

Usage:
    python3 match_recipe.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --output /work/out/edit.conform.yaml \
        --slot source_0=/work/candidates/source_0/film.mkv \
        --threshold 0.85
"""
import argparse
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import datetime

import yaml
from PIL import Image
import imagehash

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", required=True, help="Path to write conform plan YAML")
    p.add_argument("--slot", action="append", default=[], metavar="slot_id=/path/to/file")
    p.add_argument("--threshold", type=float, default=0.85,
                   help="Minimum match rate to consider a file suitable (default 0.85)")
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
# ffprobe
# ---------------------------------------------------------------------------

def ffprobe_source(path):
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed: {}".format(result.stderr[:200]))
    data = json.loads(result.stdout)

    info = {"duration_seconds": None, "fps": None,
            "resolution_x": None, "resolution_y": None,
            "video_codec": None, "audio_codec": None}

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info["video_codec"] is None:
            info["video_codec"] = stream.get("codec_name")
            info["resolution_x"] = stream.get("width")
            info["resolution_y"] = stream.get("height")
            r = stream.get("r_frame_rate", "0/1").split("/")
            try:
                info["fps"] = float(r[0]) / float(r[1])
            except (ValueError, ZeroDivisionError):
                pass
        elif stream.get("codec_type") == "audio" and info["audio_codec"] is None:
            info["audio_codec"] = stream.get("codec_name")

    fmt = data.get("format", {})
    if fmt.get("duration"):
        info["duration_seconds"] = float(fmt["duration"])

    return info

# ---------------------------------------------------------------------------
# Fingerprint comparison
# ---------------------------------------------------------------------------

def decode_chromaprint(encoded):
    """Decode a base64 Chromaprint fingerprint to a list of int32 values."""
    if not encoded:
        return []
    try:
        # Add padding and decode URL-safe base64
        pad = encoded + "=" * (4 - len(encoded) % 4)
        raw = base64.urlsafe_b64decode(pad)
        # First 4 bytes = algorithm (little-endian int32), rest = fingerprint ints
        n = (len(raw) - 4) // 4
        if n <= 0:
            return []
        return list(struct.unpack("<" + "I" * n, raw[4: 4 + n * 4]))
    except Exception:
        return []


def audio_ber(fp_encoded_a, fp_encoded_b):
    """Bit error rate between two base64 Chromaprint fingerprints. Lower = better match."""
    a = decode_chromaprint(fp_encoded_a)
    b = decode_chromaprint(fp_encoded_b)
    n = min(len(a), len(b))
    if n == 0:
        return 1.0
    bits = n * 32
    hamming = sum(bin(x ^ y).count("1") for x, y in zip(a[:n], b[:n]))
    return hamming / bits


def phash_distance(h1, h2):
    """Hamming distance between two 16-char hex pHash strings."""
    if not h1 or not h2:
        return 64
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return 64


def anchor_matches(recipe_anchor, viewer_audio_map, viewer_video_map,
                   viewer_tc, audio_threshold=0.35, phash_threshold=10):
    """
    Test whether a recipe anchor matches the viewer file at viewer_tc.
    Returns (matched: bool, confidence: str 'high'|'medium'|'none')
    """
    tc_int = int(round(viewer_tc))
    audio_match = False
    video_match = False

    # Audio comparison
    recipe_fp = (recipe_anchor.get("audio") or {}).get("fingerprint")
    viewer_fp = viewer_audio_map.get(tc_int)
    if recipe_fp and viewer_fp:
        ber = audio_ber(recipe_fp, viewer_fp)
        audio_match = ber < audio_threshold

    # Video comparison (skip low-entropy anchors)
    if not recipe_anchor.get("low_entropy"):
        recipe_ph = (recipe_anchor.get("video") or {}).get("phash")
        viewer_ph = (viewer_video_map.get(tc_int) or {}).get("phash")
        if recipe_ph and viewer_ph:
            dist = phash_distance(recipe_ph, viewer_ph)
            video_match = dist < phash_threshold

    if audio_match and video_match:
        return True, "high"
    if audio_match or video_match:
        return True, "medium"
    return False, "none"

# ---------------------------------------------------------------------------
# Viewer file fingerprinting
# ---------------------------------------------------------------------------

def fingerprint_viewer_audio(viewer_path, duration_seconds):
    """
    Run fpcalc -chunk 1 -overlap on the viewer file.
    Returns dict of {timecode_int -> fingerprint_str}.
    """
    print("  Extracting viewer audio fingerprints (fpcalc)...", flush=True)
    cmd = [
        "fpcalc", "-json", "-chunk", "1", "-overlap",
        "-length", str(int(duration_seconds) + 1),
        viewer_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    fp_map = {}
    if result.returncode != 0:
        print("  WARNING: fpcalc failed: {}".format(result.stderr[:200]), file=sys.stderr)
        return fp_map
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            tc = entry.get("timestamp", 0)
            fp = entry.get("fingerprint", "")
            if fp:
                fp_map[int(round(tc))] = fp
        except Exception:
            pass
    print("  Audio fingerprints: {}".format(len(fp_map)), flush=True)
    return fp_map


def fingerprint_viewer_video(viewer_path, duration_seconds, frames_dir):
    """
    Extract one frame per second from the viewer file, compute pHash/aHash.
    Returns dict of {timecode_int -> {phash, ahash}}.
    """
    print("  Extracting viewer video frames (1fps)...", flush=True)
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", viewer_path,
        "-vf", "fps=1,scale=32:32:flags=lanczos,format=gray",
        "-f", "image2",
        os.path.join(frames_dir, "frame_%06d.png"),
        "-hide_banner", "-loglevel", "error", "-y",
    ]
    subprocess.run(cmd, check=True)

    video_map = {}
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.startswith("frame_") or not fname.endswith(".png"):
            continue
        n = int(fname.replace("frame_", "").replace(".png", ""))
        tc_int = n - 1
        png_path = os.path.join(frames_dir, fname)
        try:
            img = Image.open(png_path).convert("L")
            video_map[tc_int] = {
                "phash": str(imagehash.phash(img)),
                "ahash": str(imagehash.average_hash(img)),
            }
        except Exception:
            pass

    print("  Video frames: {}".format(len(video_map)), flush=True)
    return video_map

# ---------------------------------------------------------------------------
# 3-phase matching
# ---------------------------------------------------------------------------

def score_offset(recipe_anchors, viewer_audio_map, viewer_video_map,
                 offset, speed_factor=1.0, sample_step=1):
    """
    Score an (offset, speed_factor) hypothesis against sampled anchors.
    Returns (match_count, total_tested).
    """
    matches = 0
    tested = 0
    for anchor in recipe_anchors[::sample_step]:
        t_r = anchor["timecode"]
        t_v = t_r * speed_factor + offset
        if t_v < 0:
            continue
        matched, _ = anchor_matches(anchor, viewer_audio_map, viewer_video_map, t_v)
        if matched:
            matches += 1
        tested += 1
    return matches, tested


def phase1_coarse_offset(recipe_anchors, viewer_audio_map, viewer_video_map,
                         viewer_duration, search_range=(-120, 300), step=1.0):
    """
    Phase 1: assume speed=1.0, slide offset to find coarse alignment.
    Use every 30th anchor (roughly every 30s) for speed.
    Returns best_offset, best_score.
    """
    print("  Phase 1: coarse offset search...", flush=True)
    best_offset = 0.0
    best_score = 0.0

    offsets = []
    o = float(search_range[0])
    while o <= search_range[1]:
        offsets.append(o)
        o += step

    for offset in offsets:
        matches, tested = score_offset(
            recipe_anchors, viewer_audio_map, viewer_video_map,
            offset, speed_factor=1.0, sample_step=30,
        )
        if tested == 0:
            continue
        score = matches / tested
        if score > best_score:
            best_score = score
            best_offset = offset

    print("  Phase 1 result: offset={:.2f}s  score={:.1%}".format(
        best_offset, best_score), flush=True)
    return best_offset, best_score


def phase2_drift_detection(recipe_anchors, viewer_audio_map, viewer_video_map, rough_offset):
    """
    Phase 2: detect speed drift using a linear regression.

    Strategy: for each anchor, test offset candidates near rough_offset +/- 5s.
    Find the best local offset for early anchors and late anchors separately.
    The difference gives us the slope (speed_factor).
    """
    print("  Phase 2: drift detection...", flush=True)

    # Find best local offset for anchors in the first and second halves
    if not recipe_anchors:
        return rough_offset, 1.0

    mid_tc = recipe_anchors[len(recipe_anchors) // 2]["timecode"]

    early = [a for a in recipe_anchors if a["timecode"] <= mid_tc]
    late = [a for a in recipe_anchors if a["timecode"] > mid_tc]

    def best_local_offset(anchors_subset, center, window=5.0, step=0.25):
        best = center
        best_s = 0.0
        o = center - window
        while o <= center + window:
            m, t = score_offset(anchors_subset, viewer_audio_map, viewer_video_map,
                                o, speed_factor=1.0, sample_step=5)
            s = m / t if t > 0 else 0.0
            if s > best_s:
                best_s = s
                best = o
            o += step
        return best

    early_offset = best_local_offset(early, rough_offset)
    late_offset = best_local_offset(late, rough_offset)

    # early anchors are centered around t = mid_tc/2
    # late anchors are centered around t = mid_tc * 1.5
    # speed_factor: how many viewer seconds per recipe second
    # t_viewer = t_recipe * speed_factor + base_offset
    # early: early_offset = base_offset + (mid_tc/2) * (speed_factor - 1)
    # late:  late_offset  = base_offset + (mid_tc*1.5) * (speed_factor - 1)
    # => late_offset - early_offset = mid_tc * (speed_factor - 1)
    # => speed_factor = 1 + (late_offset - early_offset) / mid_tc

    if mid_tc > 0:
        speed_factor = 1.0 + (late_offset - early_offset) / mid_tc
    else:
        speed_factor = 1.0

    # Recompute base offset using speed_factor and early anchors center
    early_tc_center = mid_tc / 2.0
    base_offset = early_offset - early_tc_center * (speed_factor - 1.0)

    # Clamp speed_factor to plausible range (0.95 to 1.05)
    speed_factor = max(0.95, min(1.05, speed_factor))

    print("  Phase 2 result: offset={:.3f}s  speed_factor={:.6f}".format(
        base_offset, speed_factor), flush=True)
    return base_offset, speed_factor


def phase3_refinement(recipe_anchors, viewer_audio_map, viewer_video_map,
                      rough_offset, rough_speed, window=2.0, step=0.1):
    """
    Phase 3: narrow 2D search around Phase 2 estimates using all anchors.
    Returns (best_offset, best_speed_factor, match_rate).
    """
    print("  Phase 3: refinement...", flush=True)
    best_offset = rough_offset
    best_speed = rough_speed
    best_score = 0.0

    offsets = []
    o = rough_offset - window
    while o <= rough_offset + window:
        offsets.append(round(o, 3))
        o += step

    speeds = []
    s = rough_speed - 0.005
    while s <= rough_speed + 0.005:
        speeds.append(round(s, 6))
        s += 0.001

    for offset in offsets:
        for speed in speeds:
            matches, tested = score_offset(
                recipe_anchors, viewer_audio_map, viewer_video_map,
                offset, speed_factor=speed, sample_step=3,
            )
            if tested == 0:
                continue
            score = matches / tested
            if score > best_score:
                best_score = score
                best_offset = offset
                best_speed = speed

    # Final full-density score at best params
    matches, tested = score_offset(
        recipe_anchors, viewer_audio_map, viewer_video_map,
        best_offset, speed_factor=best_speed, sample_step=1,
    )
    match_rate = matches / tested if tested > 0 else 0.0

    print("  Phase 3 result: offset={:.3f}s  speed={:.6f}  match_rate={:.1%}".format(
        best_offset, best_speed, match_rate), flush=True)
    return best_offset, best_speed, match_rate


def compute_trim_points(source, offset, speed_factor, viewer_duration):
    """
    Compute where to trim the viewer file.

    The author's content spans from the minimum strip_in timecode to the
    maximum strip_out timecode in the source.

    Using the linear model: t_viewer = t_author * speed_factor + offset
    We compute the viewer timecodes for the first and last used frames,
    then clamp to [0, viewer_duration].
    """
    strip_timecodes = source.get("strip_timecodes", [])
    if strip_timecodes:
        in_tcs = [e["timecode"] for e in strip_timecodes if e.get("role") == "strip_in"]
        out_tcs = [e["timecode"] for e in strip_timecodes if e.get("role") == "strip_out"]
        content_start_author = min(in_tcs) if in_tcs else 0.0
        content_end_author = max(out_tcs) if out_tcs else source.get(
            "original", {}).get("duration_seconds", 0.0)
    else:
        content_start_author = 0.0
        content_end_author = source.get("original", {}).get("duration_seconds", 0.0)

    # Map to viewer timecodes
    viewer_content_start = content_start_author * speed_factor + offset
    viewer_content_end = content_end_author * speed_factor + offset

    # Clamp to valid range
    trim_start = max(0.0, viewer_content_start)
    trim_end = min(viewer_duration, viewer_content_end)
    trim_duration = trim_end - trim_start

    return {
        "content_start_author_tc": round(content_start_author, 6),
        "content_end_author_tc": round(content_end_author, 6),
        "trim_start_seconds": round(trim_start, 6),
        "trim_duration_seconds": round(trim_duration, 6),
    }

# ---------------------------------------------------------------------------
# Main match loop
# ---------------------------------------------------------------------------

def match_slot(source, viewer_path, viewer_info, work_dir, threshold):
    """
    Match a single viewer file against a recipe source slot.
    Returns a result dict for inclusion in the conform plan.
    """
    slot_id = source["id"]
    viewer_duration = viewer_info["duration_seconds"]
    frames_dir = os.path.join(work_dir, "{}_frames".format(slot_id))

    # Fingerprint the viewer file
    viewer_audio_map = fingerprint_viewer_audio(viewer_path, viewer_duration)
    viewer_video_map = fingerprint_viewer_video(viewer_path, viewer_duration, frames_dir)

    recipe_anchors = source.get("anchors", [])
    if not recipe_anchors:
        print("  WARNING: no anchors in recipe for {}".format(slot_id), file=sys.stderr)
        return None

    # 3-phase alignment
    rough_offset, rough_score = phase1_coarse_offset(
        recipe_anchors, viewer_audio_map, viewer_video_map, viewer_duration)

    if rough_score < 0.1:
        print("  No alignment found (best score {:.1%}) - wrong file?".format(rough_score),
              flush=True)
        return {"slot_id": slot_id, "status": "no_match", "match_rate": rough_score}

    refined_offset, speed_factor = phase2_drift_detection(
        recipe_anchors, viewer_audio_map, viewer_video_map, rough_offset)

    final_offset, final_speed, match_rate = phase3_refinement(
        recipe_anchors, viewer_audio_map, viewer_video_map,
        refined_offset, speed_factor)

    suitable = match_rate >= threshold

    # Compute trim points
    trim = compute_trim_points(source, final_offset, final_speed, viewer_duration)

    orig = source.get("original", {})
    result = {
        "slot_id": slot_id,
        "slot_name": source.get("name", ""),
        "status": "suitable" if suitable else "unsuitable",
        "match_rate": round(match_rate, 4),
        "input_file": viewer_path,
        "transform": {
            "offset_seconds": round(final_offset, 6),
            "speed_factor": round(final_speed, 8),
            "trim_start_seconds": trim["trim_start_seconds"],
            "trim_duration_seconds": trim["trim_duration_seconds"],
            "fps_in": viewer_info["fps"],
            "fps_out": orig.get("fps"),
            "resolution_in": [viewer_info["resolution_x"], viewer_info["resolution_y"]],
            "resolution_out": [orig.get("resolution_x"), orig.get("resolution_y")],
        },
        "output_filename": orig.get("filename", "{}_conformed.mkv".format(slot_id)),
    }
    return result

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

            result = match_slot(source, viewer_path, viewer_info, work_dir, args.threshold)
            if result:
                results.append(result)
                if result["status"] != "suitable":
                    all_suitable = False
            else:
                results.append({"slot_id": slot_id, "status": "error"})
                all_suitable = False
        except Exception as e:
            print("  ERROR: {}".format(e), file=sys.stderr)
            import traceback; traceback.print_exc()
            results.append({"slot_id": slot_id, "status": "error", "error": str(e)})
            all_suitable = False

    # Print summary
    print("\n" + "=" * 60, flush=True)
    print("Match Summary", flush=True)
    print("=" * 60, flush=True)
    for r in results:
        status = r.get("status", "?")
        rate = r.get("match_rate")
        rate_str = "  ({:.1%})".format(rate) if rate is not None else ""
        t = r.get("transform", {})
        if t:
            print("  {}: {}{}  offset={:.2f}s  speed={:.6f}  trim={:.2f}s+{:.2f}s".format(
                r["slot_id"], status, rate_str,
                t.get("offset_seconds", 0),
                t.get("speed_factor", 1.0),
                t.get("trim_start_seconds", 0),
                t.get("trim_duration_seconds", 0),
            ), flush=True)
        else:
            print("  {}: {}".format(r["slot_id"], status), flush=True)
    print(flush=True)

    if not all_suitable:
        print("WARNING: not all slots are suitable. Conform plan written but may be incomplete.",
              flush=True)

    # Write conform plan
    conform_plan = {
        "grey17_version": "1",
        "conform_plan_version": "1.0",
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe": os.path.basename(args.recipe),
        "all_suitable": all_suitable,
        "sources": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(conform_plan, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print("Conform plan written to: {}".format(args.output), flush=True)


if __name__ == "__main__":
    main()

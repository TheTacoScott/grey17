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
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
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
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Minimum match rate to consider a file suitable (default 0.65)")
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
# SHA256
# ---------------------------------------------------------------------------

def compute_sha256(path):
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

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
    Tries both floor and ceil of viewer_tc so fractional offsets don't miss.
    Returns (matched: bool, confidence: str 'high'|'medium'|'none')
    """
    tc_floor = int(viewer_tc)
    tc_ceil = tc_floor + 1
    audio_match = False
    video_match = False

    # Audio comparison: try both floor and ceil, take the better BER
    recipe_fp = (recipe_anchor.get("audio") or {}).get("fingerprint")
    if recipe_fp:
        bers = []
        for tc_int in (tc_floor, tc_ceil):
            viewer_fp = viewer_audio_map.get(tc_int)
            if viewer_fp:
                bers.append(audio_ber(recipe_fp, viewer_fp))
        if bers:
            audio_match = min(bers) < audio_threshold

    # Video comparison: try both floor and ceil, take the lower pHash distance
    if not recipe_anchor.get("low_entropy"):
        recipe_ph = (recipe_anchor.get("video") or {}).get("phash")
        if recipe_ph:
            dists = []
            for tc_int in (tc_floor, tc_ceil):
                viewer_ph = (viewer_video_map.get(tc_int) or {}).get("phash")
                if viewer_ph:
                    dists.append(phash_distance(recipe_ph, viewer_ph))
            if dists:
                video_match = min(dists) < phash_threshold

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
# Phase 4: fine-pass frame alignment using cut sequences
# ---------------------------------------------------------------------------

def extract_frames_native_fps(viewer_path, start_tc, duration, fps, seq_dir):
    """
    Extract frames at the given fps for a window starting at start_tc.
    Returns list of pHash strings (one per frame).
    """
    os.makedirs(seq_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss", str(max(0.0, start_tc)),
        "-i", viewer_path,
        "-t", str(duration),
        "-vf", "scale=32:32:flags=lanczos,format=gray",
        "-vsync", "cfr",
        "-r", str(round(fps, 6)),
        "-f", "image2",
        os.path.join(seq_dir, "frame_%06d.png"),
        "-hide_banner", "-loglevel", "error", "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return []
    phashes = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.startswith("frame_") or not fname.endswith(".png"):
            continue
        try:
            img = Image.open(os.path.join(seq_dir, fname)).convert("L")
            phashes.append(str(imagehash.phash(img)))
        except Exception:
            phashes.append("0" * 16)
    return phashes


def sliding_window_best(recipe_phashes, viewer_phashes):
    """
    Slide recipe_phashes over viewer_phashes, return (best_pos, avg_dist).
    best_pos is the index into viewer_phashes where recipe_phashes[0] aligns best.
    avg_dist is average pHash Hamming distance at that alignment.
    """
    rn = len(recipe_phashes)
    vn = len(viewer_phashes)
    if rn == 0 or vn < rn:
        return None, float("inf")
    best_pos = 0
    best_dist = float("inf")
    for pos in range(vn - rn + 1):
        dist = sum(phash_distance(recipe_phashes[i], viewer_phashes[pos + i])
                   for i in range(rn))
        avg = dist / rn
        if avg < best_dist:
            best_dist = avg
            best_pos = pos
    return best_pos, best_dist


def phase4_fine_pass(recipe_anchors, viewer_path, rough_offset, rough_speed, work_dir):
    """
    Phase 4: frame-precise alignment using cut_sequence sliding window.

    For each cut anchor with a reliable cut_sequence:
    - Predict viewer timecode using rough_offset/rough_speed
    - Extract viewer frames at recipe native fps for a +/-4s window
    - Slide recipe cut_sequence.phashes over viewer frames to find frame-precise alignment
    - Collect (recipe_tc, precise_viewer_tc) pairs
    - Linear regression to get final (offset, speed_factor)

    Returns (offset, speed_factor), unchanged if insufficient data.
    """
    print("  Phase 4: fine-pass frame alignment...", flush=True)
    VIEWER_WINDOW = 4.0  # seconds on each side of predicted viewer position
    MAX_AVG_DIST = 15.0  # reject alignment if avg pHash distance exceeds this

    cut_anchors = [
        a for a in recipe_anchors
        if a.get("role") in ("strip_in", "strip_out")
        and isinstance(a.get("cut_sequence"), dict)
        and a["cut_sequence"].get("reliable", True)
        and a["cut_sequence"].get("phashes")
    ]

    if not cut_anchors:
        print("  Phase 4: no cut sequences available, skipping.", flush=True)
        return rough_offset, rough_speed

    print("  Phase 4: {} cut anchors with sequences".format(len(cut_anchors)), flush=True)

    pairs = []  # (recipe_tc, viewer_tc_precise)
    seq_base = os.path.join(work_dir, "fine_pass")

    for i, anchor in enumerate(cut_anchors):
        recipe_tc = anchor["timecode"]
        cut_seq = anchor["cut_sequence"]
        recipe_fps = cut_seq["fps"]
        recipe_phashes = cut_seq["phashes"]
        window_start = cut_seq["window_start_tc"]

        predicted_viewer_tc = recipe_tc * rough_speed + rough_offset
        viewer_start = max(0.0, predicted_viewer_tc - VIEWER_WINDOW)
        viewer_duration = VIEWER_WINDOW * 2.0

        seq_dir = os.path.join(seq_base, "anchor_{:04d}".format(i))
        viewer_phashes = extract_frames_native_fps(
            viewer_path, viewer_start, viewer_duration, recipe_fps, seq_dir)
        shutil.rmtree(seq_dir, ignore_errors=True)

        if len(viewer_phashes) < len(recipe_phashes):
            continue

        best_pos, avg_dist = sliding_window_best(recipe_phashes, viewer_phashes)
        if best_pos is None or avg_dist > MAX_AVG_DIST:
            continue

        # viewer_phashes[0] corresponds to viewer_start.
        # recipe_phashes[0] corresponds to window_start (which is recipe_tc - window).
        # best_pos tells us where recipe frame 0 aligns in the viewer frame list.
        # Viewer tc for recipe_phashes[0]: viewer_start + best_pos / recipe_fps
        # The cut point (recipe_tc) is (recipe_tc - window_start) seconds into the recipe sequence.
        frames_to_cut = (recipe_tc - window_start) * recipe_fps
        precise_viewer_cut_tc = viewer_start + (best_pos + frames_to_cut) / recipe_fps

        pairs.append((recipe_tc, precise_viewer_cut_tc))
        print("  anchor {:.3f}s -> viewer {:.6f}s  (avg dist {:.1f})".format(
            recipe_tc, precise_viewer_cut_tc, avg_dist), flush=True)

    shutil.rmtree(seq_base, ignore_errors=True)

    if len(pairs) < 2:
        print("  Phase 4: only {} pairs, need >=2. Keeping Phase 3 result.".format(
            len(pairs)), flush=True)
        return rough_offset, rough_speed

    # Linear regression: viewer_tc = recipe_tc * speed_factor + offset
    n = len(pairs)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        print("  Phase 4: degenerate regression, keeping Phase 3 result.", flush=True)
        return rough_offset, rough_speed

    speed_factor = (n * sum_xy - sum_x * sum_y) / denom
    offset = (sum_y - speed_factor * sum_x) / n

    speed_factor = max(0.95, min(1.05, speed_factor))

    print("  Phase 4 result: offset={:.6f}s  speed={:.8f}  ({} pairs)".format(
        offset, speed_factor, n), flush=True)
    return offset, speed_factor

# ---------------------------------------------------------------------------
# Crop detection
# ---------------------------------------------------------------------------

def detect_crop(viewer_path, duration, native_w, native_h):
    """
    Use ffmpeg cropdetect to find black bars in the viewer file.

    Samples frames from 5-85% of the video duration (avoids opening/ending
    credits and fade-to-black sequences). Runs cropdetect on ~300 evenly
    spaced frames and takes the mode (most common) crop value.

    Returns dict {w, h, x, y} if a significant crop is detected, else None.
    A crop is considered significant if it differs from the native resolution
    by more than THRESHOLD pixels on any side.
    """
    THRESHOLD = 8
    STABLE_FRACTION = 0.50  # crop value must appear in >=50% of frames

    start = duration * 0.05
    sample_duration = duration * 0.80
    target_samples = 300
    fps_sample = max(0.1, target_samples / sample_duration)

    cmd = [
        "ffmpeg",
        "-ss", "{:.3f}".format(start),
        "-i", viewer_path,
        "-t", "{:.3f}".format(sample_duration),
        "-vf", "fps={:.4f},cropdetect=limit=24:round=2:reset=0".format(fps_sample),
        "-an", "-f", "null", "-",
        "-hide_banner",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # cropdetect outputs to stderr: "crop=W:H:X:Y" at the end of each line
    crop_counts = {}
    for line in result.stderr.splitlines():
        if "crop=" not in line:
            continue
        idx = line.rfind("crop=")
        val = line[idx + 5:].split()[0] if line[idx + 5:].split() else ""
        parts = val.split(":")
        if len(parts) != 4:
            continue
        try:
            w, h, x, y = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            continue
        crop_counts[(w, h, x, y)] = crop_counts.get((w, h, x, y), 0) + 1

    if not crop_counts:
        return None

    total = sum(crop_counts.values())
    (w, h, x, y), count = max(crop_counts.items(), key=lambda kv: kv[1])

    if count / total < STABLE_FRACTION:
        return None

    # Check if crop is meaningfully different from native resolution
    cropped_right = (native_w - (x + w))
    cropped_bottom = (native_h - (y + h))
    if x <= THRESHOLD and y <= THRESHOLD and cropped_right <= THRESHOLD and cropped_bottom <= THRESHOLD:
        return None

    return {"w": w, "h": h, "x": x, "y": y}

# ---------------------------------------------------------------------------
# Cross-correlation matching engine
# ---------------------------------------------------------------------------

def phash_to_popcount(phash_hex):
    """Convert a 16-char hex pHash to its popcount (0..64)."""
    try:
        return bin(int(phash_hex, 16)).count("1")
    except (ValueError, TypeError):
        return 32  # neutral value on failure


def build_popcount_signal(phash_list):
    """Convert a list of pHash hex strings to a list of popcount ints."""
    return [phash_to_popcount(h) for h in phash_list]


def build_motion_signal(phash_list):
    """
    Build a frame-to-frame pHash distance signal (temporal gradient).

    signal[i] = hamming_distance(phash[i], phash[i+1])

    This is encode-agnostic: scene cuts produce identical spikes (distance ~30-60)
    regardless of codec, resolution, or color grading. Static shots produce low
    values (~0-5). The pattern is preserved across all versions of the same content,
    making it a far more discriminative cross-correlation signal than popcount.

    Returns a list of length len(phash_list)-1.
    """
    signal = []
    for i in range(len(phash_list) - 1):
        signal.append(phash_distance(phash_list[i], phash_list[i + 1]))
    return signal


def extract_viewer_phash_sequence(viewer_path, duration_seconds, dense_fps, seq_dir):
    """
    Extract frames from the viewer file at dense_fps and compute pHash for each.
    Returns dict: {interval, start, phashes} matching the recipe phash_sequence format.
    """
    os.makedirs(seq_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", viewer_path,
        "-vf", "fps={:.6f},scale=32:32:flags=lanczos,format=gray".format(dense_fps),
        "-f", "image2",
        os.path.join(seq_dir, "f_%08d.png"),
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  WARNING: dense viewer frame extraction failed", file=sys.stderr)
        return None

    phashes = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.startswith("f_") or not fname.endswith(".png"):
            continue
        try:
            img = Image.open(os.path.join(seq_dir, fname)).convert("L")
            phashes.append(str(imagehash.phash(img)))
        except Exception:
            phashes.append("0" * 16)

    if not phashes:
        return None

    interval = round(1.0 / dense_fps, 9)
    print("  Viewer dense frames: {} at {}fps".format(len(phashes), dense_fps), flush=True)
    return {"interval": interval, "start": 0.0, "phashes": phashes}


def crosscorr_scan(recipe_signal, viewer_signal, interval,
                   offset_min, offset_max, step, max_val=64):
    """
    Slide viewer_signal over recipe_signal within [offset_min, offset_max].
    offset is in seconds; step is in seconds.

    Score = normalized agreement: mean of (max_val - |r[i] - v[i+shift]|) over overlap.
    Works for both popcount signals (max_val=64) and motion signals (max_val=64).

    Normalizing by overlap count makes the score fair regardless of whether the
    viewer is longer or shorter than the recipe (extended credits, pre-roll, etc.).
    Requires at least 10% of recipe length to overlap before trusting a candidate.

    Returns (best_offset_seconds, best_normalized_score).
    """
    best_offset = 0.0
    best_score = -1.0
    min_overlap = max(10, len(recipe_signal) // 10)

    offset = offset_min
    while offset <= offset_max + 1e-9:
        shift = int(round(offset / interval))
        score = 0
        count = 0
        for ri, r_val in enumerate(recipe_signal):
            vi = ri + shift
            if vi < 0 or vi >= len(viewer_signal):
                continue
            score += max_val - abs(r_val - viewer_signal[vi])
            count += 1
        if count >= min_overlap:
            norm_score = score / count
            if norm_score > best_score:
                best_score = norm_score
                best_offset = offset
        offset += step

    return best_offset, best_score


def crosscorr_pass1(recipe_seq, viewer_seq, search_min=-600.0, search_max=600.0):
    """
    Coarse cross-correlation: downsample both signals to ~1fps, scan full range.
    Returns (coarse_offset_seconds, score).
    """
    print("  XCorr Pass 1: coarse scan ({:.0f}s to +{:.0f}s)...".format(
        search_min, search_max), flush=True)

    interval = recipe_seq["interval"]

    # Use full-density motion signal - do NOT downsample. Downsampling can drop
    # scene cuts that fall between sampled frames, killing the signal's peaks.
    # Instead scan in coarse 1s steps over the full-density signal.
    r_motion = build_motion_signal(recipe_seq["phashes"])
    v_motion = build_motion_signal(viewer_seq["phashes"])

    coarse_step = 1.0  # scan in 1s steps for speed; Pass 2 refines to 0.25s
    offset, score = crosscorr_scan(r_motion, v_motion, interval, search_min, search_max,
                                   coarse_step)

    print("  Pass 1 result: offset={:.2f}s  score={:.2f}".format(offset, score), flush=True)
    return offset, score


def crosscorr_pass2(recipe_seq, viewer_seq, coarse_offset, window=4.0):
    """
    Fine cross-correlation: full density signal, narrow window around coarse_offset.
    Returns (fine_offset_seconds, score).
    """
    print("  XCorr Pass 2: fine scan (offset {:.2f}s +/- {:.1f}s)...".format(
        coarse_offset, window), flush=True)

    interval = recipe_seq["interval"]
    r_motion = build_motion_signal(recipe_seq["phashes"])
    v_motion = build_motion_signal(viewer_seq["phashes"])

    search_min = coarse_offset - window
    search_max = coarse_offset + window

    offset, score = crosscorr_scan(r_motion, v_motion, interval, search_min, search_max, interval)

    print("  Pass 2 result: offset={:.4f}s  score={:.2f}".format(offset, score), flush=True)
    return offset, score


def crosscorr_pass3_speed(recipe_seq, viewer_seq, fine_offset, n_segments=5, window=2.0):
    """
    Speed detection: split recipe into N segments, find local offset for each
    via cross-correlation, fit a line to get speed_factor and base_offset.
    Returns (base_offset, speed_factor).
    """
    print("  XCorr Pass 3: speed detection ({} segments)...".format(n_segments), flush=True)

    interval = recipe_seq["interval"]
    r_hashes = recipe_seq["phashes"]
    v_hashes = viewer_seq["phashes"]
    r_len = len(r_hashes)

    seg_size = r_len // n_segments
    if seg_size < 10:
        print("  Pass 3: recipe too short for {} segments, skipping.".format(n_segments),
              flush=True)
        return fine_offset, 1.0

    # Score threshold: a segment whose best normalized xcorr score is below this
    # is assumed to have no corresponding content in the viewer (e.g. pre-roll or
    # credits that differ between versions). Exclude it from the speed regression.
    # 42/64 = ~65% agreement - well above random (which is ~50% for natural images).
    SEG_SCORE_THRESHOLD = 42.0

    pairs = []  # (recipe_midpoint_tc, local_offset)
    for i in range(n_segments):
        seg_start = i * seg_size
        seg_end = min(seg_start + seg_size, r_len)
        seg_mid_tc = (seg_start + (seg_end - seg_start) / 2.0) * interval

        r_seg_motion = build_motion_signal(r_hashes[seg_start:seg_end])

        # Determine viewer slice for this segment at the expected offset
        v_start_idx = int(round((seg_mid_tc - seg_size * interval / 2.0 + fine_offset) / interval))
        v_start_idx = max(0, v_start_idx)
        pad = int(round(window / interval))
        v_lo = max(0, v_start_idx - pad)
        v_hi = min(len(v_hashes), v_start_idx + seg_size + pad)
        if v_hi <= v_lo:
            print("  Pass 3: segment {} has no viewer content, skipping.".format(i), flush=True)
            continue
        v_seg_motion = build_motion_signal(v_hashes[v_lo:v_hi])

        seg_offset_min = fine_offset - window
        seg_offset_max = fine_offset + window
        local_off, local_score = crosscorr_scan(
            r_seg_motion, v_seg_motion, interval,
            seg_offset_min, seg_offset_max, interval,
        )
        if local_score < SEG_SCORE_THRESHOLD:
            print("  Pass 3: segment {} low score ({:.1f}), skipping (credits/pre-roll?).".format(
                i, local_score), flush=True)
            continue
        pairs.append((seg_mid_tc, local_off))

    if len(pairs) < 2:
        print("  Pass 3: only {} usable segments, keeping fine offset.".format(len(pairs)),
              flush=True)
        return fine_offset, 1.0

    # Linear regression: local_offset = base_offset + (speed_factor - 1) * tc
    # i.e. viewer_tc = recipe_tc * speed_factor + base_offset
    # local_offset[i] ~ base_offset + slope * seg_mid_tc[i]
    n = len(pairs)
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in xs)
    sum_xy = sum(x * y for x, y in zip(xs, ys))

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        print("  Pass 3: degenerate regression, keeping fine offset.", flush=True)
        return fine_offset, 1.0

    slope = (n * sum_xy - sum_x * sum_y) / denom
    base_offset = (sum_y - slope * sum_x) / n
    speed_factor = 1.0 + slope
    speed_factor = max(0.95, min(1.05, speed_factor))

    print("  Pass 3 result: base_offset={:.4f}s  speed={:.8f}".format(
        base_offset, speed_factor), flush=True)
    return base_offset, speed_factor


def xcorr_match(recipe_seq, viewer_seq):
    """
    Full cross-correlation pipeline: Pass 1 (coarse) -> Pass 2 (fine) -> Pass 3 (speed).
    Returns (offset_seconds, speed_factor).
    """
    coarse_offset, _ = crosscorr_pass1(recipe_seq, viewer_seq)
    fine_offset, _ = crosscorr_pass2(recipe_seq, viewer_seq, coarse_offset)
    base_offset, speed_factor = crosscorr_pass3_speed(recipe_seq, viewer_seq, fine_offset)
    return base_offset, speed_factor


# ---------------------------------------------------------------------------
# 3-phase matching (legacy - used when recipe has no phash_sequence)
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

    orig = source.get("original", {})

    # SHA256 shortcut: exact file match bypasses all fingerprint work
    recipe_sha256 = orig.get("sha256")
    if recipe_sha256:
        print("  Computing SHA256...", flush=True)
        viewer_sha256 = compute_sha256(viewer_path)
        if viewer_sha256 == recipe_sha256:
            print("  SHA256 match - exact file, skipping fingerprint.", flush=True)
            transform = {
                "offset_seconds": 0.0,
                "speed_factor": 1.0,
                "trim_start_seconds": 0.0,
                "trim_duration_seconds": round(viewer_duration, 6),
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

    frames_dir = os.path.join(work_dir, "{}_frames".format(slot_id))
    recipe_anchors = source.get("anchors", [])
    recipe_seq = source.get("phash_sequence")

    if recipe_seq:
        # Cross-correlation path (new recipes signed at 4fps)
        dense_fps = round(1.0 / recipe_seq["interval"])
        print("  Extracting viewer dense frames ({}fps)...".format(dense_fps), flush=True)
        viewer_seq = extract_viewer_phash_sequence(
            viewer_path, viewer_duration, dense_fps,
            os.path.join(frames_dir, "dense"),
        )
        if viewer_seq is None:
            print("  ERROR: could not extract viewer frames", file=sys.stderr)
            return {"slot_id": slot_id, "status": "no_match", "match_rate": 0.0}

        final_offset, final_speed = xcorr_match(recipe_seq, viewer_seq)

        # Compute match_rate directly from dense sequences at the computed offset.
        # For each recipe frame, find the expected viewer frame index and compare pHash.
        # This avoids the 1fps integer-snapping problem entirely.
        r_hashes = recipe_seq["phashes"]
        v_hashes = viewer_seq["phashes"]
        r_interval = recipe_seq["interval"]
        v_interval = viewer_seq["interval"]
        PHASH_MATCH_THRESHOLD = 15  # relaxed for cross-encode matching (DVDRip, Xvid etc.)

        matches = 0
        tested = 0
        for i, r_hash in enumerate(r_hashes):
            t_r = i * r_interval
            t_v = t_r * final_speed + final_offset
            if t_v < 0:
                continue
            j = int(round(t_v / v_interval))
            if j < 0 or j >= len(v_hashes):
                continue
            tested += 1
            if phash_distance(r_hash, v_hashes[j]) < PHASH_MATCH_THRESHOLD:
                matches += 1

        match_rate = matches / tested if tested > 0 else 0.0
        print("  Match rate (dense xcorr): {:.1%}  ({}/{} frames)".format(
            match_rate, matches, tested), flush=True)

        # Still need audio+video maps for Phase 4 (cut sequence fine pass)
        viewer_audio_map = fingerprint_viewer_audio(viewer_path, viewer_duration)
        viewer_video_map = fingerprint_viewer_video(viewer_path, viewer_duration,
                                                    os.path.join(frames_dir, "1fps"))
        match_method = "xcorr"
    else:
        # Legacy 3-phase path (old recipes signed at 1fps without phash_sequence)
        print("  WARNING: recipe has no phash_sequence (signed at low density). "
              "Re-sign for better accuracy.", file=sys.stderr)
        viewer_audio_map = fingerprint_viewer_audio(viewer_path, viewer_duration)
        viewer_video_map = fingerprint_viewer_video(viewer_path, viewer_duration, frames_dir)

        if not recipe_anchors:
            print("  WARNING: no anchors in recipe for {}".format(slot_id), file=sys.stderr)
            return None

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
        match_method = "fingerprint"

    # Phase 4: frame-precise alignment using cut sequences
    # Guard: discard Phase 4 result if it deviates more than 2s from current offset
    p4_offset, p4_speed = phase4_fine_pass(
        recipe_anchors, viewer_path, final_offset, final_speed, work_dir)
    if abs(p4_offset - final_offset) <= 2.0:
        final_offset, final_speed = p4_offset, p4_speed
    else:
        print("  Phase 4 discarded: offset changed {:.2f}s (> 2s), keeping Pass 3 result.".format(
            abs(p4_offset - final_offset)), flush=True)

    suitable = match_rate >= threshold

    # Compute trim points
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

    # Crop detection: only when the author flagged the source as full-frame
    # and the match is suitable (no point detecting crop on a wrong file)
    if source.get("expect_full_frame") and suitable:
        print("  Detecting crop (expect_full_frame=true)...", flush=True)
        crop = detect_crop(
            viewer_path,
            viewer_duration,
            viewer_info["resolution_x"],
            viewer_info["resolution_y"],
        )
        if crop:
            transform["crop"] = crop
            print("  Crop detected: {}x{} offset {}x{}  (removing {}px left, {}px top, "
                  "{}px right, {}px bottom)".format(
                      crop["w"], crop["h"], crop["x"], crop["y"],
                      crop["x"], crop["y"],
                      viewer_info["resolution_x"] - (crop["x"] + crop["w"]),
                      viewer_info["resolution_y"] - (crop["y"] + crop["h"]),
                  ), flush=True)
        else:
            print("  No significant crop detected.", flush=True)

    result = {
        "slot_id": slot_id,
        "slot_name": source.get("name", ""),
        "status": "suitable" if suitable else "unsuitable",
        "match_rate": round(match_rate, 4),
        "match_method": match_method,
        "input_file": viewer_path,
        "transform": transform,
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

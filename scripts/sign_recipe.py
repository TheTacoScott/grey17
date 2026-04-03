#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Generates fingerprint anchor points for all source files listed in a recipe.

Usage:
    python3 sign_recipe.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --manifest /work/manifest.json \
        --source source_0=/work/sources/source_0/film.mkv \
        --anchor-interval 1.0
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys

import imagehash
import yaml
from PIL import Image

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--recipe", required=True, help="Path to recipe.yaml (read/write)")
    p.add_argument("--work-dir", default="/work/tmp", help="Writable scratch directory")
    p.add_argument("--source", action="append", default=[], metavar="slot_id=/path/to/file",
                   help="Map a source slot to a file (repeatable)")
    p.add_argument("--anchor-interval", type=float, default=1.0,
                   help="Seconds between interval anchors (default: 1.0)")
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
# ffprobe
# ---------------------------------------------------------------------------

def ffprobe_source(path):
    """Return dict with fps, duration, resolution, codec info."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed for {}: {}".format(path, result.stderr))
    data = json.loads(result.stdout)

    info = {
        "duration_seconds": None,
        "fps": None,
        "resolution_x": None,
        "resolution_y": None,
        "video_codec": None,
        "audio_codec": None,
        "audio_channels": None,
        "audio_sample_rate": None,
    }

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and info["video_codec"] is None:
            info["video_codec"] = stream.get("codec_name")
            info["resolution_x"] = stream.get("width")
            info["resolution_y"] = stream.get("height")
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            try:
                num, den = r_frame_rate.split("/")
                info["fps"] = float(num) / float(den)
            except (ValueError, ZeroDivisionError):
                info["fps"] = None
        elif stream.get("codec_type") == "audio" and info["audio_codec"] is None:
            info["audio_codec"] = stream.get("codec_name")
            info["audio_channels"] = stream.get("channels")
            info["audio_sample_rate"] = stream.get("sample_rate")

    fmt = data.get("format", {})
    if fmt.get("duration"):
        info["duration_seconds"] = float(fmt["duration"])

    return info

# ---------------------------------------------------------------------------
# Anchor timecode computation
# ---------------------------------------------------------------------------

def get_strip_timecodes(source):
    """
    Return list of (timecode, role, strip_name) from recipe source's
    strip_timecodes field (written by generate-recipe).
    """
    points = []
    for entry in source.get("strip_timecodes", []):
        points.append((entry["timecode"], entry["role"], entry.get("strip", "")))
    return points


def compute_anchor_timecodes(source, duration_seconds, anchor_interval):
    """
    Return sorted list of dicts: {timecode, role, strip_refs}.
    Interval anchors are merged with strip in/out anchors when within 0.5s.
    """
    # Build interval set
    interval_tcs = set()
    t = 0.0
    while t <= duration_seconds + 0.01:
        interval_tcs.add(round(t, 6))
        t += anchor_interval

    # Build strip in/out points
    strip_points = {}  # timecode -> {role, refs}
    for (tc, role, strip_name) in get_strip_timecodes(source):
        tc = round(tc, 6)
        if tc < 0 or tc > duration_seconds + 1.0:
            continue
        if tc not in strip_points:
            strip_points[tc] = {"role": role, "refs": []}
        strip_points[tc]["refs"].append(strip_name)

    # Merge: for each strip point, check if an interval anchor is within 0.5s
    # If so, promote the interval anchor to also carry the strip role
    merged = {}  # timecode -> {role, strip_refs}

    for tc in sorted(interval_tcs):
        merged[tc] = {"role": "interval", "strip_refs": []}

    for tc, sp in strip_points.items():
        # Find nearest interval anchor
        nearest = round(round(tc / anchor_interval) * anchor_interval, 6)
        if abs(nearest - tc) <= 0.5 and nearest in merged:
            # Promote nearest interval anchor
            merged[nearest]["role"] = sp["role"]
            merged[nearest]["strip_refs"].extend(sp["refs"])
        else:
            # Insert as its own anchor
            merged[tc] = {"role": sp["role"], "strip_refs": sp["refs"]}

    return [{"timecode": tc, **v} for tc, v in sorted(merged.items())]

# ---------------------------------------------------------------------------
# Video frame extraction
# ---------------------------------------------------------------------------

def extract_frames_batch(source_path, timecodes_1s, frames_dir):
    """
    Extract one frame per second (at integer timecodes) using a single ffmpeg pass.
    Output: frames_dir/interval_NNNNNN.png (numbered from 0)
    Returns dict: {timecode_int -> png_path}
    """
    os.makedirs(frames_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", source_path,
        "-vf", "fps=1,scale=32:32:flags=lanczos,format=gray",
        "-f", "image2",
        os.path.join(frames_dir, "interval_%06d.png"),
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("ffmpeg frame batch extraction failed for {}".format(source_path))

    # Map frame number -> timecode (frame 1 = t=0, frame 2 = t=1, ...)
    mapping = {}
    for fname in sorted(os.listdir(frames_dir)):
        if not fname.startswith("interval_") or not fname.endswith(".png"):
            continue
        n = int(fname.replace("interval_", "").replace(".png", ""))
        tc_int = n - 1  # 0-indexed timecode
        mapping[tc_int] = os.path.join(frames_dir, fname)
    return mapping


DENSE_FPS = 4.0  # frames per second for phash_sequence (cross-correlation signal)


def extract_phash_sequence(source_path, duration_seconds, dense_fps, frames_dir):
    """
    Extract frames at dense_fps and compute pHash for each.
    Returns dict with keys: interval, start, phashes (list of hex strings).
    The list index maps to timecode: tc = start + index * (1/dense_fps).
    """
    seq_dir = os.path.join(frames_dir, "dense")
    os.makedirs(seq_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", source_path,
        "-vf", "fps={:.6f},scale=32:32:flags=lanczos,format=gray".format(dense_fps),
        "-f", "image2",
        os.path.join(seq_dir, "f_%08d.png"),
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: dense frame extraction failed, skipping phash_sequence", file=sys.stderr)
        return None

    interval = round(1.0 / dense_fps, 9)
    phashes = []
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.startswith("f_") or not fname.endswith(".png"):
            continue
        ph, _ah, _le = compute_hashes(os.path.join(seq_dir, fname))
        phashes.append(ph if ph else "0" * 16)

    if not phashes:
        return None

    print("  Dense phash sequence: {} frames at {}fps".format(len(phashes), dense_fps),
          flush=True)
    return {"interval": interval, "start": 0.0, "phashes": phashes}


def extract_frame_seek(source_path, timecode, out_path):
    """Extract a single frame at a specific timecode using accurate seek."""
    cmd = [
        "ffmpeg",
        "-ss", str(timecode),
        "-i", source_path,
        "-vframes", "1",
        "-vf", "scale=32:32:flags=lanczos,format=gray",
        out_path,
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    return result.returncode == 0


def compute_hashes(png_path):
    """Return (phash_str, ahash_str) for a 32x32 grayscale PNG."""
    try:
        img = Image.open(png_path).convert("L")
        ph = str(imagehash.phash(img))
        ah = str(imagehash.average_hash(img))
        # Detect low-entropy (solid black/white)
        mean_val = sum(img.getdata()) / (img.width * img.height)
        low_entropy = mean_val < 5 or mean_val > 250
        return ph, ah, low_entropy
    except Exception as e:
        print("WARNING: hash failed for {}: {}".format(png_path, e), file=sys.stderr)
        return None, None, True

# ---------------------------------------------------------------------------
# Crop detection
# ---------------------------------------------------------------------------

CROP_THRESHOLD = 8  # pixels - less than this on all sides = full frame


def detect_crop(source_path, duration, native_w, native_h):
    """
    Use ffmpeg cropdetect to check if the author's source has black bars.
    Samples frames from 5-85% of video, takes the mode crop value.
    Returns dict {w, h, x, y} if bars detected, or None if full frame.
    """
    STABLE_FRACTION = 0.50

    start = duration * 0.05
    sample_duration = duration * 0.80
    fps_sample = max(0.1, 300.0 / sample_duration)

    cmd = [
        "ffmpeg",
        "-ss", "{:.3f}".format(start),
        "-i", source_path,
        "-t", "{:.3f}".format(sample_duration),
        "-vf", "fps={:.4f},cropdetect=limit=24:round=2:reset=0".format(fps_sample),
        "-an", "-f", "null", "-",
        "-hide_banner",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

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

    cropped_right = native_w - (x + w)
    cropped_bottom = native_h - (y + h)
    if x <= CROP_THRESHOLD and y <= CROP_THRESHOLD and \
            cropped_right <= CROP_THRESHOLD and cropped_bottom <= CROP_THRESHOLD:
        return None

    return {"w": w, "h": h, "x": x, "y": y}

# ---------------------------------------------------------------------------
# Native-fps cut sequence extraction
# ---------------------------------------------------------------------------

CUT_SEQUENCE_WINDOW = 2.0  # seconds on each side of the cut point


def extract_cut_sequence(source_path, cut_tc, fps, seq_dir):
    """
    Extract frames at native fps for a window of +/- CUT_SEQUENCE_WINDOW seconds
    around cut_tc. Compute pHash for each frame.
    Returns dict with window_start_tc, fps, phashes, reliable - or None on failure.
    """
    start = max(0.0, cut_tc - CUT_SEQUENCE_WINDOW)
    duration = CUT_SEQUENCE_WINDOW * 2.0

    os.makedirs(seq_dir, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(duration),
        "-vf", "scale=32:32:flags=lanczos,format=gray",
        "-vsync", "cfr",
        "-r", str(round(fps, 6)),
        "-f", "image2",
        os.path.join(seq_dir, "frame_%06d.png"),
        "-hide_banner", "-loglevel", "error",
        "-y",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        return None

    phashes = []
    low_entropy_count = 0
    for fname in sorted(os.listdir(seq_dir)):
        if not fname.startswith("frame_") or not fname.endswith(".png"):
            continue
        ph, _ah, low_entropy = compute_hashes(os.path.join(seq_dir, fname))
        phashes.append(ph if ph else "0" * 16)
        if low_entropy:
            low_entropy_count += 1

    if not phashes:
        return None

    reliable = (low_entropy_count / len(phashes)) < 0.5

    return {
        "window_start_tc": round(start, 6),
        "fps": round(fps, 6),
        "phashes": phashes,
        "reliable": reliable,
    }

# ---------------------------------------------------------------------------
# Audio fingerprinting via fpcalc
# ---------------------------------------------------------------------------

FINGERPRINT_CHUNK_SECONDS = 3


def build_audio_fingerprint_map(source_path, duration_seconds):
    """
    Run fpcalc once with -chunk 1 -overlap to get per-second fingerprints
    for the whole file. Returns dict of {timecode_int -> fingerprint_str}.

    fpcalc -chunk 1 produces one fingerprint per second chunk.
    With -overlap each chunk slightly overlaps the next, reducing edge effects.
    """
    cmd = [
        "fpcalc",
        "-json",
        "-chunk", "1",
        "-overlap",
        "-length", str(int(duration_seconds) + 1),
        source_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("WARNING: fpcalc batch failed: {}".format(result.stderr[:200]), file=sys.stderr)
        return {}

    fp_map = {}
    try:
        # fpcalc -chunk outputs one JSON object per line (newline-delimited)
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            tc = entry.get("timestamp", 0)
            fp = entry.get("fingerprint", "")
            if fp:
                fp_map[int(round(tc))] = fp
    except Exception as e:
        print("WARNING: fpcalc output parse error: {}".format(e), file=sys.stderr)

    return fp_map


def fingerprint_at_offset(source_path, timecode):
    """
    Run fpcalc for a single 3-second window at an exact timecode.
    Used for strip in/out anchors that don't fall on integer seconds.
    Returns fingerprint string or None.
    """
    start = max(0.0, timecode - FINGERPRINT_CHUNK_SECONDS / 2.0)
    cmd = [
        "fpcalc",
        "-json",
        "-format", "matroska",
        "-length", str(FINGERPRINT_CHUNK_SECONDS),
        source_path,
    ]
    # Use ffmpeg pipe to seek to offset, pipe into fpcalc via stdin
    seek_cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-i", source_path,
        "-t", str(FINGERPRINT_CHUNK_SECONDS),
        "-ac", "1", "-ar", "16000",
        "-f", "wav", "-",
        "-hide_banner", "-loglevel", "error",
    ]
    fpcalc_cmd = ["fpcalc", "-json", "-format", "wav", "-"]
    try:
        ffmpeg_proc = subprocess.Popen(seek_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        fpcalc_result = subprocess.run(fpcalc_cmd, stdin=ffmpeg_proc.stdout,
                                       capture_output=True, text=True)
        ffmpeg_proc.wait()
        if fpcalc_result.returncode == 0:
            data = json.loads(fpcalc_result.stdout)
            return data.get("fingerprint")
    except Exception as e:
        print("WARNING: fpcalc seek failed at t={}: {}".format(timecode, e), file=sys.stderr)
    return None

# ---------------------------------------------------------------------------
# Per-source signing
# ---------------------------------------------------------------------------

def sign_source(source, source_path, anchor_interval, work_dir):
    """
    Sign a single source file. Returns (source_metadata, anchors_list).
    source: the source dict from the recipe.
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

    # Detect whether the author's source is full-frame or has black bars
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

    print("  Computing anchor timecodes...", flush=True)
    anchor_list = compute_anchor_timecodes(source, duration, anchor_interval)
    total = len(anchor_list)
    print("  Anchors: {}  ({} interval, {} strip in/out)".format(
        total,
        sum(1 for a in anchor_list if a["role"] == "interval"),
        sum(1 for a in anchor_list if a["role"] in ("strip_in", "strip_out")),
    ), flush=True)

    frames_dir = os.path.join(work_dir, "{}_frames".format(source_id))
    seek_frame_dir = os.path.join(work_dir, "{}_seek_frames".format(source_id))
    os.makedirs(seek_frame_dir, exist_ok=True)

    # --- Video: batch extract 1fps frames (used for per-anchor hashes) ---
    print("  Extracting video frames (1fps pass)...", flush=True)
    interval_frame_map = extract_frames_batch(source_path, None, frames_dir)
    print("  Extracted {} interval frames".format(len(interval_frame_map)), flush=True)

    # --- Dense phash sequence at DENSE_FPS for cross-correlation matching ---
    print("  Extracting dense phash sequence ({}fps)...".format(DENSE_FPS), flush=True)
    phash_sequence = extract_phash_sequence(source_path, duration, DENSE_FPS, frames_dir)

    # --- Audio: batch fpcalc per-second fingerprints ---
    print("  Computing audio fingerprints (fpcalc batch)...", flush=True)
    audio_map = build_audio_fingerprint_map(source_path, duration)
    print("  Audio fingerprints: {}".format(len(audio_map)), flush=True)

    # --- Compute anchors ---
    print("  Computing anchors...", flush=True)
    anchors = []
    seek_count = 0
    fpcalc_seek_count = 0

    for i, anchor in enumerate(anchor_list):
        tc = anchor["timecode"]
        tc_int = int(round(tc))

        # --- Video frame ---
        if tc_int in interval_frame_map and abs(tc - tc_int) < 0.5:
            png_path = interval_frame_map[tc_int]
        else:
            seek_png = os.path.join(seek_frame_dir, "seek_{:.3f}.png".format(tc))
            if not os.path.exists(seek_png):
                extract_frame_seek(source_path, tc, seek_png)
                seek_count += 1
            png_path = seek_png if os.path.exists(seek_png) else None

        phash, ahash, low_entropy = (None, None, True)
        if png_path and os.path.exists(png_path):
            phash, ahash, low_entropy = compute_hashes(png_path)

        # --- Audio fingerprint ---
        if tc_int in audio_map and abs(tc - tc_int) < 0.5:
            audio_fp = audio_map[tc_int]
        else:
            # Strip in/out point not on an integer second - targeted seek
            audio_fp = fingerprint_at_offset(source_path, tc)
            fpcalc_seek_count += 1

        entry = {
            "timecode": round(tc, 6),
            "role": anchor["role"],
        }
        if anchor["strip_refs"]:
            entry["strip_refs"] = anchor["strip_refs"]
        if low_entropy:
            entry["low_entropy"] = True
        entry["video"] = {"phash": phash, "ahash": ahash}
        entry["audio"] = {"fingerprint": audio_fp}

        anchors.append(entry)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print("  [{}/{}] ({} video seeks, {} audio seeks)".format(
                i + 1, total, seek_count, fpcalc_seek_count), flush=True)

    # Extract native-fps cut sequences for strip_in/strip_out anchors
    native_fps = meta.get("fps") or 24.0
    cut_anchors = [a for a in anchors if a["role"] in ("strip_in", "strip_out")]
    if cut_anchors:
        print("  Extracting cut sequences for {} cut anchors (fps={:.3f})...".format(
            len(cut_anchors), native_fps), flush=True)
        cut_seq_base = os.path.join(work_dir, "{}_cut_seqs".format(source_id))
        for anchor in cut_anchors:
            tc = anchor["timecode"]
            seq_dir = os.path.join(cut_seq_base, "tc_{:.6f}".format(tc).replace(".", "_"))
            cut_seq = extract_cut_sequence(source_path, tc, native_fps, seq_dir)
            if cut_seq:
                anchor["cut_sequence"] = cut_seq
            shutil.rmtree(seq_dir, ignore_errors=True)
        n_seq = sum(1 for a in cut_anchors if "cut_sequence" in a)
        print("  Cut sequences extracted: {}".format(n_seq), flush=True)

    return meta, anchors, phash_sequence

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
            meta, anchors, phash_sequence = sign_source(
                source, source_path,
                args.anchor_interval, work_dir,
            )
        except Exception as e:
            print("ERROR signing {}: {}".format(slot_id, e), file=sys.stderr)
            continue

        # Write back into recipe
        # expect_full_frame: true if no black bars detected in the author's source.
        # Viewers with pillarboxed/letterboxed files will be cropped to match.
        # The author can override this manually if the auto-detection is wrong.
        if "expect_full_frame" not in source:
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
        source["anchors"] = anchors
        if phash_sequence:
            source["phash_sequence"] = phash_sequence
            print("  {} dense phash frames written for {}".format(
                len(phash_sequence["phashes"]), slot_id), flush=True)
        print("  {} anchors written for {}".format(len(anchors), slot_id), flush=True)

    recipe["signed"] = True

    # Clean up work dir
    shutil.rmtree(work_dir, ignore_errors=True)

    # Write updated recipe
    with open(args.recipe, "w") as f:
        yaml.dump(recipe, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False, width=99999)

    print("\nRecipe signed and written to: {}".format(args.recipe), flush=True)


if __name__ == "__main__":
    main()

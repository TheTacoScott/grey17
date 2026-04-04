#!/usr/bin/env python3
"""
Shared utilities for grey17 Docker scripts.
Imported by sign_recipe.py and match_recipe.py.
"""
import hashlib
import json
import subprocess
import sys

import imagehash
from PIL import Image

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
    """
    Return dict with technical metadata for a video file.
    Keys: duration_seconds, fps, resolution_x, resolution_y,
          video_codec, audio_codec, audio_channels, audio_sample_rate.
    All values may be None if the stream is absent or unreadable.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError("ffprobe failed for {}: {}".format(path, result.stderr[:300]))
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
                pass
        elif stream.get("codec_type") == "audio" and info["audio_codec"] is None:
            info["audio_codec"] = stream.get("codec_name")
            info["audio_channels"] = stream.get("channels")
            info["audio_sample_rate"] = stream.get("sample_rate")

    fmt = data.get("format", {})
    if fmt.get("duration"):
        info["duration_seconds"] = float(fmt["duration"])

    return info

# ---------------------------------------------------------------------------
# pHash
# ---------------------------------------------------------------------------

def phash_distance(h1, h2):
    """Hamming distance between two 16-char hex pHash strings. Returns 0-64."""
    if not h1 or not h2:
        return 64
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count("1")
    except ValueError:
        return 64


def hash_frame(png_path):
    """
    Compute perceptual hashes for a 32x32 grayscale PNG.
    Returns (phash_str, ahash_str), or ("0"*16, "0"*16) on failure.
    """
    try:
        img = Image.open(png_path).convert("L")
        return str(imagehash.phash(img)), str(imagehash.average_hash(img))
    except Exception as e:
        print("WARNING: hash failed for {}: {}".format(png_path, e), file=sys.stderr)
        return "0" * 16, "0" * 16

# ---------------------------------------------------------------------------
# Crop detection
# ---------------------------------------------------------------------------

CROP_THRESHOLD = 8  # pixels - less than this on all sides = full frame


def detect_crop(source_path, duration, native_w, native_h):
    """
    Use ffmpeg cropdetect to check for black bars.
    Samples ~300 frames from 5-85% of the video duration, takes the mode crop value.
    Returns dict {w, h, x, y} if bars are detected, or None if the frame is full.
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

#!/usr/bin/env python3
"""
Runs inside Docker container.
Applies ffmpeg transforms from a conform plan to produce conformed output files.

Two conform modes are supported:

Simple conform (no commercial breaks):
    A single ffmpeg pass applies offset, trim, crop, fps normalisation, and
    scale. Speed is always 1.0 so no setpts or atempo filter is applied.
    Used when source["dtw"]["segments"] is absent or empty.

Segmented conform (commercial-break-aware, frame-perfect):
    The video is cut into content segments and black-frame break segments.
    Content segments are extracted at 1:1 speed with exactly author_frames
    output frames (-frames:v). Break segments are synthesised as exact-length
    black video with silence. All segments are encoded to a lossless
    intermediate (FFV1 + PCM), then joined with ffmpeg concat demuxer + copy,
    and finally re-encoded to the delivery format (libx264 CRF 16 + AAC 320k).

    Used when source["dtw"]["segments"] is a non-empty list, written by
    match_recipe.py when DTW detects commercial break length differences.

Usage:
    python3 conform_sources.py \
        --plan /work/plan/edit.conform.yaml \
        --work-dir /work/output \
        --slot source_0=/work/sources/source_0/film.mkv
"""
import argparse
import os
import subprocess
import sys

import yaml

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="Path to conform plan YAML")
    p.add_argument("--work-dir", default="/work/output",
                   help="Output directory for conformed files")
    p.add_argument("--slot", action="append", default=[],
                   metavar="slot_id=/path/to/file",
                   help="Override input file path for a slot (repeatable)")
    return p.parse_args()


def parse_slot_args(slot_args):
    mapping = {}
    for s in slot_args:
        if "=" not in s:
            print("WARNING: ignoring malformed --slot arg: {}".format(s),
                  file=sys.stderr)
            continue
        slot_id, path = s.split("=", 1)
        mapping[slot_id.strip()] = path.strip()
    return mapping


# ---------------------------------------------------------------------------
# Output parameter helpers
# ---------------------------------------------------------------------------

def _output_params(source):
    """
    Extract output fps, resolution, and crop from a conform plan source entry.
    Returns (fps_out, res_w, res_h, crop_spec).
    fps_out may be None; res_w/res_h may be None.
    """
    t = source.get("transform") or {}
    fps_out = t.get("fps_out") or t.get("fps_in")
    fps_out = float(fps_out) if fps_out else None
    res_out = t.get("resolution_out") or [None, None]
    res_w = int(res_out[0]) if res_out[0] else None
    res_h = int(res_out[1]) if res_out[1] else None
    crop_spec = t.get("crop")
    if crop_spec and not (crop_spec.get("w") and crop_spec.get("h")):
        crop_spec = None
    return fps_out, res_w, res_h, crop_spec


# ---------------------------------------------------------------------------
# Simple (single-pass) conform
# ---------------------------------------------------------------------------

def _simple_conform(source, input_file, work_dir):
    """
    Apply the global transform from the conform plan in a single ffmpeg pass.
    Covers offset, trim, crop, speed correction, fps, and scale.
    """
    slot_id = source.get("slot_id", "?")
    output_filename = source.get("output_filename") or "{}_conformed.mkv".format(slot_id)
    output_path = os.path.join(work_dir, output_filename)

    t = source.get("transform") or {}
    offset        = float(t.get("offset_seconds") or 0.0)
    trim_duration = t.get("trim_duration_seconds")
    fps_in        = t.get("fps_in")
    fps_out       = t.get("fps_out")
    res_in        = t.get("resolution_in") or [None, None]
    res_out       = t.get("resolution_out") or [None, None]
    crop_spec     = t.get("crop")

    apply_crop  = bool(crop_spec and crop_spec.get("w") and crop_spec.get("h"))
    apply_fps   = fps_out and fps_in and abs(float(fps_out) - float(fps_in)) > 0.001

    if apply_crop:
        eff_w, eff_h = int(crop_spec["w"]), int(crop_spec["h"])
    else:
        eff_w, eff_h = res_in[0], res_in[1]
    apply_scale = (
        res_out[0] and res_out[1] and eff_w and eff_h and
        (int(res_out[0]) != int(eff_w) or int(res_out[1]) != int(eff_h))
    )

    print("  Input:    {}".format(os.path.basename(input_file)), flush=True)
    print("  Offset:   {:.6f}s".format(offset), flush=True)
    print("  Duration: {:.3f}s".format(float(trim_duration) if trim_duration else 0.0),
          flush=True)
    if apply_crop:
        print("  Crop:     {}x{} from ({},{})".format(
            crop_spec["w"], crop_spec["h"], crop_spec["x"], crop_spec["y"]), flush=True)
    if apply_fps:
        print("  FPS:      {:.6f} -> {:.6f}".format(float(fps_in), float(fps_out)),
              flush=True)
    if apply_scale:
        print("  Scale:    {}x{} -> {}x{}".format(eff_w, eff_h, res_out[0], res_out[1]),
              flush=True)
    print("  Output:   {}".format(output_path), flush=True)

    cmd = ["ffmpeg", "-y"]
    if offset > 0.001:
        cmd += ["-ss", "{:.6f}".format(offset)]
    cmd += ["-i", input_file]
    if trim_duration:
        cmd += ["-t", "{:.6f}".format(float(trim_duration))]

    vf_parts = []
    if apply_crop:
        vf_parts.append("crop={}:{}:{}:{}".format(
            int(crop_spec["w"]), int(crop_spec["h"]),
            int(crop_spec["x"]), int(crop_spec["y"])))
    if apply_fps:
        vf_parts.append("fps={:.6f}".format(float(fps_out)))
    if apply_scale:
        vf_parts.append("scale={}:{}:flags=lanczos".format(
            int(res_out[0]), int(res_out[1])))
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "16"]
    cmd += ["-c:a", "aac", "-b:a", "320k"]
    cmd += ["-hide_banner", output_path]

    print("  Running ffmpeg...", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("  ERROR: ffmpeg failed (exit {})".format(r.returncode), file=sys.stderr)
        return False

    if not os.path.exists(output_path):
        print("  ERROR: output file not created", file=sys.stderr)
        return False

    print("  Written: {:.1f}MB".format(os.path.getsize(output_path) / 1e6), flush=True)
    return True


# ---------------------------------------------------------------------------
# Segmented conform helpers
# ---------------------------------------------------------------------------

# Lossless intermediate codec used for all per-segment files.
# FFV1 + PCM ensures no generation loss before the final concat re-encode.
_SEG_VCODEC = ["-c:v", "ffv1", "-level", "3", "-threads", "4"]
_SEG_ACODEC = ["-c:a", "pcm_s16le"]

# Delivery codec for the final concatenated output.
_OUT_VCODEC = ["-c:v", "libx264", "-preset", "fast", "-crf", "16"]
_OUT_ACODEC = ["-c:a", "aac", "-b:a", "320k"]


def _vf_for_segment(fps_out, res_w, res_h, crop_spec):
    """
    Build the video filter string for a content segment.
    Applies PTS normalisation, crop, fps normalisation, and scale so all
    intermediates are identical in format (required for copy-concat).
    setpts=PTS-STARTPTS resets the output PTS to start from 0 regardless of
    where in the source file the seek landed. This prevents the concat demuxer
    from accumulating audio/video seek-offset drift across segments.
    """
    parts = []
    if crop_spec:
        parts.append("crop={}:{}:{}:{}".format(
            int(crop_spec["w"]), int(crop_spec["h"]),
            int(crop_spec["x"]), int(crop_spec["y"])))
    if fps_out:
        parts.append("fps={:.6f}".format(fps_out))
    if res_w and res_h:
        parts.append("scale={}:{}:flags=lanczos".format(res_w, res_h))
    # Reset PTS to 0 after all other filters so the FFV1 intermediate always
    # starts at PTS=0. This must come AFTER fps so the fps filter uses the
    # original source PTS for correct frame selection at the seek position.
    parts.append("setpts=PTS-STARTPTS")
    return ",".join(parts)


def _probe_has_audio(input_file):
    """Return True if the file has at least one audio stream."""
    import json
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", input_file],
        capture_output=True, text=True)
    if r.returncode != 0:
        return False
    try:
        streams = json.loads(r.stdout).get("streams", [])
        return any(s.get("codec_type") == "audio" for s in streams)
    except Exception:
        return False


def _extract_content_segment(input_file, seg, seg_path,
                              fps_out, res_w, res_h, crop_spec, has_audio):
    """
    Extract a content segment from input_file at 1:1 speed with fps
    normalisation, crop, and scale. Outputs exactly seg["author_frames"]
    frames to a lossless intermediate (FFV1 + PCM when has_audio, FFV1-only
    otherwise).
    """
    viewer_start_tc     = float(seg.get("viewer_start_tc") or 0.0)
    viewer_duration_sec = float(seg.get("viewer_duration_secs") or 0.0)
    author_frames       = int(seg.get("author_frames") or 0)

    print("  seg[content] seek={:.3f}s extract={:.3f}s -> {}fr".format(
        viewer_start_tc, viewer_duration_sec, author_frames), flush=True)

    cmd = ["ffmpeg", "-y"]
    if viewer_start_tc > 0.001:
        cmd += ["-ss", "{:.6f}".format(viewer_start_tc)]
    cmd += ["-i", input_file]
    # Soft duration limit: +1s headroom keeps us from running into the next
    # segment. -frames:v enforces the exact output frame count.
    if viewer_duration_sec > 0.0:
        cmd += ["-t", "{:.6f}".format(viewer_duration_sec + 1.0)]

    vf = _vf_for_segment(fps_out, res_w, res_h, crop_spec)
    if vf:
        cmd += ["-vf", vf]

    if has_audio:
        # Normalize to 48000 Hz stereo so all segments are concat-compatible.
        # asetpts=PTS-STARTPTS resets audio PTS to 0 to match the video PTS
        # normalisation done by setpts=PTS-STARTPTS in the video filter chain.
        cmd += ["-af", "asetpts=PTS-STARTPTS", "-ar", "48000", "-ac", "2"]

    if author_frames > 0:
        cmd += ["-frames:v", str(author_frames)]

    cmd += _SEG_VCODEC
    if has_audio:
        cmd += _SEG_ACODEC
    else:
        cmd += ["-an"]
    # -shortest ensures audio is truncated when video ends (prevents the +1s
    # headroom on -t from adding extra audio duration per segment).
    cmd += ["-shortest", "-hide_banner", seg_path]

    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("  ERROR: content segment ffmpeg failed", file=sys.stderr)
        return False
    if not os.path.exists(seg_path):
        print("  ERROR: segment file not created: {}".format(seg_path), file=sys.stderr)
        return False
    return True


def _generate_break_segment(seg, seg_path, fps_out, res_w, res_h, has_audio):
    """
    Synthesise a break segment as exact-length black video (and silence when
    has_audio). Produces a lossless intermediate (FFV1 + optional PCM) at
    the same stream layout as content segments so copy-concat works cleanly.
    """
    author_frames = int(seg.get("author_frames") or 0)
    if author_frames <= 0:
        open(seg_path, "wb").close()   # zero-length placeholder, skipped later
        return True

    w   = res_w  if res_w  else 1920
    h   = res_h  if res_h  else 1080
    fps = fps_out if fps_out else 24.0

    print("  seg[break]   {}fr  {}x{}  {:.4f}fps".format(
        author_frames, w, h, fps), flush=True)

    # Duration passed to lavfi must be slightly longer than frames/fps so
    # ffmpeg produces at least author_frames before -frames:v cuts it.
    lavfi_dur = (author_frames + 2) / fps

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s={}x{}:r={:.6f}:d={:.6f}".format(
            w, h, fps, lavfi_dur),
    ]
    if has_audio:
        cmd += [
            "-f", "lavfi",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ]
    cmd += ["-frames:v", str(author_frames)]
    if has_audio:
        cmd += ["-ar", "48000", "-ac", "2"]
    cmd += _SEG_VCODEC
    cmd += _SEG_ACODEC if has_audio else ["-an"]
    # -shortest ensures audio is truncated when video ends.
    cmd += ["-shortest", "-hide_banner", seg_path]

    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("  ERROR: break segment ffmpeg failed", file=sys.stderr)
        return False
    return True


def _segmented_conform(source, input_file, work_dir):
    """
    Frame-perfect segmented conform.

    1. Extract each content segment at 1:1 speed with exact -frames:v
       output count (lossless FFV1 intermediate).
    2. Synthesise each break as exact-length black + silence (FFV1).
    3. Concat all segments with ffmpeg concat demuxer + -c copy.
    4. Re-encode the joined lossless stream to the delivery format
       (libx264 CRF 16 + AAC 320k).

    The two-pass approach (lossless intermediate -> delivery encode) avoids
    generation loss that would occur if each segment were pre-encoded to
    the lossy delivery format before concat-re-encode.
    """
    slot_id  = source.get("slot_id", "?")
    segments = source["dtw"]["segments"]

    fps_out, res_w, res_h, crop_spec = _output_params(source)

    output_filename = source.get("output_filename") or "{}_conformed.mkv".format(slot_id)
    output_path = os.path.join(work_dir, output_filename)

    seg_dir = os.path.join(work_dir, "{}_segtmp".format(slot_id))
    os.makedirs(seg_dir, exist_ok=True)

    has_audio = _probe_has_audio(input_file)

    n_content = sum(1 for s in segments if s["type"] == "content")
    n_break   = sum(1 for s in segments if s["type"] == "break")
    print("  Segmented conform: {} content + {} break segments  audio={}".format(
        n_content, n_break, has_audio), flush=True)
    if fps_out:
        print("  Target: {}x{}  {:.4f}fps".format(res_w, res_h, fps_out), flush=True)

    seg_paths = []
    for idx, seg in enumerate(segments):
        seg_path = os.path.join(seg_dir, "seg_{:04d}.mkv".format(idx))

        if seg["type"] == "content":
            ok = _extract_content_segment(
                input_file, seg, seg_path,
                fps_out, res_w, res_h, crop_spec, has_audio)
        else:
            ok = _generate_break_segment(
                seg, seg_path, fps_out, res_w, res_h, has_audio)

        if not ok:
            _cleanup_seg_dir(seg_dir, seg_paths)
            return False

        # Skip zero-length break files
        if os.path.getsize(seg_path) > 0:
            seg_paths.append(seg_path)
        else:
            os.unlink(seg_path)

    if not seg_paths:
        print("  ERROR: no segment files produced", file=sys.stderr)
        return False

    # Write concat list
    concat_list = os.path.join(seg_dir, "concat.txt")
    with open(concat_list, "w") as f:
        for sp in seg_paths:
            # Use absolute paths; safe=0 is required for that
            f.write("file '{}'\n".format(sp))

    # Join all lossless segments and re-encode to delivery format.
    # Two ffmpegs chained via pipe avoids writing a large lossless joined file.
    # concat demuxer -> pipe -> libx264 encoder.
    joined_path = os.path.join(seg_dir, "joined.mkv")
    print("  Concatenating {} segments -> lossless join...".format(
        len(seg_paths)), flush=True)
    join_cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", concat_list,
        "-c", "copy",
        "-hide_banner",
        joined_path,
    ]
    r = subprocess.run(join_cmd)
    if r.returncode != 0:
        print("  ERROR: concat join failed", file=sys.stderr)
        _cleanup_seg_dir(seg_dir, seg_paths + [concat_list])
        return False

    print("  Encoding joined lossless -> delivery format...", flush=True)
    encode_cmd = [
        "ffmpeg", "-y",
        "-i", joined_path,
    ] + _OUT_VCODEC + _OUT_ACODEC + [
        "-hide_banner",
        output_path,
    ]
    r = subprocess.run(encode_cmd)
    if r.returncode != 0:
        print("  ERROR: delivery encode failed", file=sys.stderr)
        _cleanup_seg_dir(seg_dir, seg_paths + [concat_list, joined_path])
        return False

    # Clean up all intermediate files
    _cleanup_seg_dir(seg_dir, seg_paths + [concat_list, joined_path])

    if not os.path.exists(output_path):
        print("  ERROR: output file not created", file=sys.stderr)
        return False

    print("  Written: {:.1f}MB  ({})".format(
        os.path.getsize(output_path) / 1e6, output_path), flush=True)
    return True


def _cleanup_seg_dir(seg_dir, files):
    for f in files:
        try:
            os.unlink(f)
        except OSError:
            pass
    try:
        os.rmdir(seg_dir)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def conform_source(source, input_file, work_dir):
    """
    Apply timing transforms from a conform plan source entry to input_file.

    Dispatch:
        segmented conform  - when dtw.segments is non-empty. Uses FFV1 lossless
                             intermediates per segment. Used when DTW detected
                             break length differences between author and viewer.
        simple conform     - single-pass trim, fps, crop, scale. No speed warp.
                             Used when no DTW segments are present.

    Returns True on success.
    """
    print("  Input: {}".format(os.path.basename(input_file)), flush=True)

    dtw      = source.get("dtw") or {}
    segments = dtw.get("segments") or []

    if segments:
        print("  Mode: segmented ({} content + {} break segments)".format(
            sum(1 for s in segments if s["type"] == "content"),
            sum(1 for s in segments if s["type"] == "break")), flush=True)
        return _segmented_conform(source, input_file, work_dir)
    else:
        print("  Mode: simple", flush=True)
        return _simple_conform(source, input_file, work_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    slot_map = parse_slot_args(args.slot)

    with open(args.plan) as f:
        plan = yaml.safe_load(f)

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    sources = plan.get("sources", [])
    ok_count = skip_count = fail_count = 0

    for source in sources:
        slot_id = source.get("slot_id", "?")
        status  = source.get("status", "")

        print("\nConforming {}...".format(slot_id), flush=True)

        if status != "suitable":
            print("  SKIP: status={}".format(status), flush=True)
            skip_count += 1
            continue

        input_file = slot_map.get(slot_id) or source.get("input_file")
        if not input_file:
            print("  ERROR: no input file for {}".format(slot_id), file=sys.stderr)
            fail_count += 1
            continue
        if not os.path.exists(input_file):
            print("  ERROR: input file not found: {}".format(input_file), file=sys.stderr)
            fail_count += 1
            continue

        ok = conform_source(source, input_file, work_dir)
        if ok:
            ok_count += 1
        else:
            fail_count += 1

    print("\n" + "=" * 50, flush=True)
    print("Conform complete: {} ok, {} skipped, {} failed".format(
        ok_count, skip_count, fail_count), flush=True)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

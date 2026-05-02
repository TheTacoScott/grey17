#!/usr/bin/env python3
"""
Runs inside Docker container.
Applies ffmpeg transforms from a conform plan to produce conformed output files.

Two conform modes are supported:

Simple conform:
    A single ffmpeg pass applies offset, trim, crop, fps normalisation, and
    scale. Speed is always 1.0. Used when source["dtw"]["segments"] is absent
    or empty (i.e. all detected black breaks have frame_delta == 0).

Segmented conform (break-length-corrected, frame-perfect):
    Used when DTW detects one or more black breaks where the viewer has more or
    fewer frames than the author (frame_delta != 0). A single ffmpeg invocation
    with a filter_complex graph handles all segments in one pass:
      - Content segments: per-segment -ss/-t/-i seek inputs; fps, crop, scale,
        trim=end_frame=AUTHOR_FRAMES, setpts=PTS-STARTPTS.
      - Break segments: lavfi color/anullsrc sources synthesised to exactly
        author_frames black frames + matching silence.
      - All segments joined by a concat filter, producing uniform PTS with no
        drift accumulation.
    Output encoded directly to libx264 CRF 16 + AAC 320k in a single pass.

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


def _segmented_conform(source, input_file, work_dir):
    """
    Frame-perfect segmented conform using a single-pass filter_complex.

    One ffmpeg invocation handles all segments:
      - Content segments: each gets its own -ss/-t/-i seek input. The per-input
        stream is piped through fps, crop, scale, trim=end_frame=AUTHOR_FRAMES,
        and setpts=PTS-STARTPTS inside the filtergraph.
      - Break segments: synthesised inline as lavfi color (black) and
        anullsrc sources, trimmed to exactly author_frames.
      - A concat filter joins all video and audio streams in order, producing
        a continuous output with uniform PTS from the very start. No intermediate
        files are written so no PTS drift accumulates across segments.

    Output: libx264 CRF 16 + AAC 320k directly from the filtergraph.
    """
    slot_id  = source.get("slot_id", "?")
    segments = source["dtw"]["segments"]

    fps_out, res_w, res_h, crop_spec = _output_params(source)

    output_filename = source.get("output_filename") or "{}_conformed.mkv".format(slot_id)
    output_path = os.path.join(work_dir, output_filename)

    has_audio = _probe_has_audio(input_file)

    fps = fps_out if fps_out else 24.0
    w   = res_w   if res_w   else 1920
    h   = res_h   if res_h   else 1080

    n_content = sum(1 for s in segments if s["type"] == "content")
    n_break   = sum(1 for s in segments if s["type"] == "break")
    n_total   = n_content + n_break

    print("  Segmented conform (filtergraph): {} content + {} break segments  audio={}".format(
        n_content, n_break, has_audio), flush=True)
    print("  Target: {}x{}  {:.6f}fps".format(w, h, fps), flush=True)

    # -- Build command inputs (one -ss/-t/-i per content segment) -----------
    cmd = ["ffmpeg", "-y"]

    # Map each content segment to its ffmpeg input index.
    input_idx = 0
    seg_input_idx = {}  # seg_list_index -> ffmpeg input index
    for seg_idx, seg in enumerate(segments):
        if seg["type"] != "content":
            continue
        author_frames = int(seg.get("author_frames") or 0)
        if author_frames <= 0:
            continue
        viewer_start_tc      = float(seg.get("viewer_start_tc")      or 0.0)
        viewer_duration_secs = float(seg.get("viewer_duration_secs") or 0.0)
        if viewer_start_tc > 0.001:
            cmd += ["-ss", "{:.6f}".format(viewer_start_tc)]
        # +1s headroom on the read window so ffmpeg decodes enough frames
        # before trim=end_frame cuts at the exact count.
        cmd += ["-t", "{:.6f}".format(viewer_duration_secs + 1.0)]
        cmd += ["-i", input_file]
        seg_input_idx[seg_idx] = input_idx
        input_idx += 1

    if input_idx == 0:
        print("  ERROR: no content segments with frames > 0", file=sys.stderr)
        return False

    # -- Build filter_complex -----------------------------------------------
    # Each segment contributes one video pad and (if has_audio) one audio pad.
    # collect_v / collect_a are the named output pads in concat order.
    filter_parts = []
    collect_v = []
    collect_a = []

    for seg_idx, seg in enumerate(segments):
        if seg["type"] == "content":
            author_frames = int(seg.get("author_frames") or 0)
            if author_frames <= 0:
                continue
            in_idx = seg_input_idx[seg_idx]
            content_dur = author_frames / fps

            # Video: crop (optional) -> fps -> scale (optional) ->
            #        trim=end_frame=N (exact frame count) ->
            #        setpts=PTS-STARTPTS (reset PTS to 0) -> yuv420p
            vchain = []
            if crop_spec:
                vchain.append("crop={}:{}:{}:{}".format(
                    int(crop_spec["w"]), int(crop_spec["h"]),
                    int(crop_spec["x"]), int(crop_spec["y"])))
            vchain.append("fps={:.6f}".format(fps))
            if res_w and res_h:
                vchain.append("scale={}:{}:flags=lanczos".format(w, h))
            vchain.append("trim=end_frame={}".format(author_frames))
            vchain.append("setpts=PTS-STARTPTS")
            vchain.append("format=yuv420p")
            v_label = "c{}v".format(seg_idx)
            filter_parts.append("[{}:v]{}[{}]".format(
                in_idx, ",".join(vchain), v_label))
            collect_v.append("[{}]".format(v_label))

            if has_audio:
                a_label = "c{}a".format(seg_idx)
                filter_parts.append(
                    "[{}:a]aresample=48000,atrim=duration={:.6f},"
                    "asetpts=PTS-STARTPTS[{}]".format(
                        in_idx, content_dur, a_label))
                collect_a.append("[{}]".format(a_label))

        else:  # break
            author_frames = int(seg.get("author_frames") or 0)
            if author_frames <= 0:
                # Zero-frame break: skip (viewer and author agree, no synthesis needed)
                continue
            break_dur = author_frames / fps
            # Overshoot by 2 frames so the color source definitely produces
            # enough frames before trim=end_frame cuts exactly.
            lavfi_dur = break_dur + 2.0 / fps

            v_raw   = "b{}vr".format(seg_idx)
            v_label = "b{}v".format(seg_idx)
            filter_parts.append(
                "color=c=black:s={}x{}:r={:.6f}:d={:.6f}[{}]".format(
                    w, h, fps, lavfi_dur, v_raw))
            filter_parts.append(
                "[{}]trim=end_frame={},setpts=PTS-STARTPTS,format=yuv420p[{}]".format(
                    v_raw, author_frames, v_label))
            collect_v.append("[{}]".format(v_label))

            if has_audio:
                a_raw   = "b{}ar".format(seg_idx)
                a_label = "b{}a".format(seg_idx)
                filter_parts.append(
                    "anullsrc=r=48000:cl=stereo[{}]".format(a_raw))
                filter_parts.append(
                    "[{}]atrim=duration={:.6f},asetpts=PTS-STARTPTS[{}]".format(
                        a_raw, break_dur, a_label))
                collect_a.append("[{}]".format(a_label))

    # Count actual pads (zero-frame segments may have been skipped)
    n_pads = len(collect_v)
    if n_pads == 0:
        print("  ERROR: filtergraph has no segments", file=sys.stderr)
        return False

    filter_parts.append(
        "{}concat=n={}:v=1:a=0[outv]".format("".join(collect_v), n_pads))
    if has_audio:
        filter_parts.append(
            "{}concat=n={}:v=0:a=1[outa]".format("".join(collect_a), n_pads))

    cmd += ["-filter_complex", ";".join(filter_parts)]
    cmd += ["-map", "[outv]"]
    if has_audio:
        cmd += ["-map", "[outa]"]
    else:
        cmd += ["-an"]

    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "16"]
    if has_audio:
        cmd += ["-c:a", "aac", "-b:a", "320k"]
    cmd += ["-hide_banner", output_path]

    print("  Running filtergraph conform ({} pads)...".format(n_pads), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("  ERROR: filtergraph conform failed (exit {})".format(r.returncode),
              file=sys.stderr)
        return False

    if not os.path.exists(output_path):
        print("  ERROR: output file not created", file=sys.stderr)
        return False

    print("  Written: {:.1f}MB  ({})".format(
        os.path.getsize(output_path) / 1e6, output_path), flush=True)
    return True


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def conform_source(source, input_file, work_dir):
    """
    Apply timing transforms from a conform plan source entry to input_file.

    Dispatch:
        segmented conform  - when dtw.segments is non-empty. Single-pass
                             filter_complex with per-segment seeks and lavfi
                             black synthesisers. Used when DTW detected breaks
                             where the viewer has more or fewer frames than the
                             author (frame_delta != 0).
        simple conform     - single-pass trim, fps, crop, scale.
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

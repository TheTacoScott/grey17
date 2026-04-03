#!/usr/bin/env python3
"""
Runs inside Docker container.
Applies ffmpeg transforms from a conform plan to produce conformed output files.

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
    p.add_argument("--work-dir", default="/work/output", help="Output directory for conformed files")
    p.add_argument("--slot", action="append", default=[], metavar="slot_id=/path/to/file",
                   help="Override input file path for a slot (repeatable)")
    return p.parse_args()


def parse_slot_args(slot_args):
    mapping = {}
    for s in slot_args:
        if "=" not in s:
            print("WARNING: ignoring malformed --slot arg: {}".format(s), file=sys.stderr)
            continue
        slot_id, path = s.split("=", 1)
        mapping[slot_id.strip()] = path.strip()
    return mapping

# ---------------------------------------------------------------------------
# atempo filter chain
# ---------------------------------------------------------------------------

def build_atempo_chain(speed):
    """
    Build an atempo filter chain. ffmpeg's atempo only accepts values in
    [0.5, 2.0], so chain multiple filters for larger corrections.
    Returns list of filter strings (may be empty if speed ~= 1.0).
    """
    if abs(speed - 1.0) < 1e-6:
        return []
    filters = []
    remaining = speed
    while remaining > 2.0 + 1e-9:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5 - 1e-9:
        filters.append("atempo=0.5")
        remaining *= 2.0
    filters.append("atempo={:.10f}".format(remaining))
    return filters

# ---------------------------------------------------------------------------
# Per-source conform
# ---------------------------------------------------------------------------

def conform_source(source, input_file, work_dir):
    """
    Apply the transform from a conform plan source entry to input_file.
    Writes the result to work_dir/output_filename.
    Returns True on success.
    """
    slot_id = source.get("slot_id", "?")

    output_filename = source.get("output_filename") or "{}_conformed.mkv".format(slot_id)
    output_path = os.path.join(work_dir, output_filename)

    t = source.get("transform") or {}
    offset = float(t.get("offset_seconds") or 0.0)
    trim_duration = t.get("trim_duration_seconds")
    speed = float(t.get("speed_factor") or 1.0)
    fps_in = t.get("fps_in")
    fps_out = t.get("fps_out")
    res_in = t.get("resolution_in") or [None, None]
    res_out = t.get("resolution_out") or [None, None]

    apply_speed = abs(speed - 1.0) > 0.0001
    apply_fps = fps_out and fps_in and abs(float(fps_out) - float(fps_in)) > 0.001
    apply_scale = (
        res_out[0] and res_out[1] and res_in[0] and res_in[1] and
        (int(res_out[0]) != int(res_in[0]) or int(res_out[1]) != int(res_in[1]))
    )

    # Print transform summary
    print("  Input:    {}".format(os.path.basename(input_file)), flush=True)
    print("  Offset:   {:.6f}s".format(offset), flush=True)
    print("  Duration: {:.3f}s".format(float(trim_duration) if trim_duration else 0.0), flush=True)
    if apply_speed:
        print("  Speed:    {:.8f}x".format(speed), flush=True)
    if apply_fps:
        print("  FPS:      {:.6f} -> {:.6f}".format(float(fps_in), float(fps_out)), flush=True)
    if apply_scale:
        print("  Scale:    {}x{} -> {}x{}".format(
            res_in[0], res_in[1], res_out[0], res_out[1]), flush=True)
    print("  Output:   {}".format(output_path), flush=True)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y"]

    # Fast input seek - placed before -i for efficient trimming
    if offset > 0.001:
        cmd += ["-ss", "{:.6f}".format(offset)]

    cmd += ["-i", input_file]

    if trim_duration:
        cmd += ["-t", "{:.6f}".format(float(trim_duration))]

    # Video filter chain
    vf_parts = []
    if apply_speed:
        # setpts scales the presentation timestamps: slower = higher PTS multiplier
        vf_parts.append("setpts={:.10f}*PTS".format(1.0 / speed))
    if apply_fps:
        vf_parts.append("fps={:.6f}".format(float(fps_out)))
    if apply_scale:
        vf_parts.append("scale={}:{}:flags=lanczos".format(int(res_out[0]), int(res_out[1])))
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]

    # Audio filter chain
    if apply_speed:
        af = build_atempo_chain(speed)
        if af:
            cmd += ["-af", ",".join(af)]

    # Output codec: high quality intermediate (will be re-encoded by cook/Blender)
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "16"]
    cmd += ["-c:a", "aac", "-b:a", "320k"]
    cmd += ["-hide_banner", output_path]

    print("  Running ffmpeg...", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("  ERROR: ffmpeg failed (exit {})".format(result.returncode), file=sys.stderr)
        return False

    # Confirm output exists and has reasonable size
    if not os.path.exists(output_path):
        print("  ERROR: output file not created".format(), file=sys.stderr)
        return False

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print("  Written: {:.1f} MB".format(size_mb), flush=True)
    return True

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
    ok_count = 0
    skip_count = 0
    fail_count = 0

    for source in sources:
        slot_id = source.get("slot_id", "?")
        status = source.get("status", "")

        print("\nConforming {}...".format(slot_id), flush=True)

        if status != "suitable":
            print("  SKIP: status={}".format(status), flush=True)
            skip_count += 1
            continue

        # Resolve input file: --slot arg overrides plan input_file
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

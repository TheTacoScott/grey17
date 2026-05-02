#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Matches viewer-provided source files against a signed recipe and writes a
conform plan.

Matching uses full-frame DTW only. No anchor/sliding-window pre-search is
performed. The DTW band (default 10000 frames, ~417s at 24fps) is centered on
the diagonal (speed=1.0, offset=0.0), which is wide enough to accommodate any
realistic initial offset between a viewer file and the author source. The
initial viewer offset is read directly from path[0] after traceback.

Usage:
    python3 match_recipe.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --output /work/out/edit.conform.yaml \
        --slot source_0=/work/candidates/source_0/film.mkv
"""
import argparse
import datetime
import os
import sys

import yaml

from utils import compute_sha256, ffprobe_source, detect_crop
import dtw_align

# DTW parameters.
# Band of 10000 frames covers ~417s at 24fps - enough for any realistic
# initial offset between a viewer file and the author source.
DTW_BAND_FRAMES = 10000


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Match viewer files against a signed recipe and produce a conform plan.",
        epilog=(
            "Runs full-frame Sakoe-Chiba banded DTW over the entire pHash sequence. "
            "The band is centered on the diagonal (speed=1.0, offset=0.0). "
            "Initial viewer offset and per-break frame deltas are read from the DTW path."
        ),
    )
    p.add_argument("--recipe", required=True)
    p.add_argument("--output", required=True, help="Path to write conform plan YAML")
    p.add_argument("--slot", action="append", default=[], metavar="slot_id=/path/to/file")
    p.add_argument("--dtw-band", type=int, default=DTW_BAND_FRAMES,
                   metavar="<frames>",
                   help="Sakoe-Chiba half-band for DTW in frame units "
                        "(default: {} = ~{:.0f}s at 24fps).".format(
                            DTW_BAND_FRAMES, DTW_BAND_FRAMES / 24.0))
    p.add_argument("--dtw-max-mem", type=int, default=256,
                   metavar="<MB>",
                   help="RAM budget in MB for DTW backpointer stripe cache "
                        "before spilling to disk (default: 256).")
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
# Trim computation
# ---------------------------------------------------------------------------

def compute_trim_points(source, offset, viewer_duration):
    """
    Compute where to trim the viewer file (speed is always 1.0).
    Author content spans t=0 to t=author_duration.
    Viewer content starts at t=offset and runs for author_duration seconds.
    """
    orig = source.get("original") or {}
    author_duration = orig.get("duration_seconds") or viewer_duration

    trim_start    = max(0.0, offset)
    trim_end      = min(viewer_duration, offset + author_duration)
    trim_duration = max(0.0, trim_end - trim_start)

    return {
        "trim_start_seconds":    round(trim_start, 6),
        "trim_duration_seconds": round(trim_duration, 6),
    }


# ---------------------------------------------------------------------------
# Main match loop
# ---------------------------------------------------------------------------

def match_slot(source, viewer_path, viewer_info,
               dtw_band=DTW_BAND_FRAMES, dtw_max_mem_mb=256, work_dir=None):
    """
    Match a single viewer file against a recipe source slot via DTW.
    Returns a result dict for inclusion in the conform plan.
    """
    slot_id = source["id"]
    viewer_duration = viewer_info["duration_seconds"]
    orig = source.get("original", {})
    fps = orig.get("fps") or 24.0

    # SHA256 shortcut: exact file match bypasses all fingerprint work.
    recipe_sha256 = orig.get("sha256")
    if recipe_sha256:
        print("  Computing SHA256...", flush=True)
        viewer_sha256 = compute_sha256(viewer_path)
        if viewer_sha256 == recipe_sha256:
            print("  SHA256 match - exact file, skipping DTW.", flush=True)
            trim = compute_trim_points(source, 0.0, viewer_duration)
            transform = {
                "offset_seconds":        0.0,
                "speed_factor":          1.0,
                "trim_start_seconds":    trim["trim_start_seconds"],
                "trim_duration_seconds": trim["trim_duration_seconds"],
                "fps_in":                viewer_info["fps"],
                "fps_out":               orig.get("fps"),
                "resolution_in":  [viewer_info["resolution_x"], viewer_info["resolution_y"]],
                "resolution_out": [orig.get("resolution_x"), orig.get("resolution_y")],
            }
            return {
                "slot_id":        slot_id,
                "slot_name":      source.get("name", ""),
                "status":         "suitable",
                "match_method":   "sha256",
                "input_file":     viewer_path,
                "transform":      transform,
                "output_filename": orig.get("filename",
                                           "{}_conformed.mkv".format(slot_id)),
            }
        print("  SHA256 mismatch - proceeding with DTW.", flush=True)

    phash_seq = source.get("phash_sequence")
    if not phash_seq:
        print("  ERROR: recipe has no phash_sequence for {}. "
              "Re-sign the recipe.".format(slot_id), file=sys.stderr)
        return {"slot_id": slot_id, "status": "no_match"}

    # Detect crop before DTW so black bars are stripped from viewer frames.
    # Only done when the recipe marks this source as full-frame.
    viewer_crop = None
    if source.get("expect_full_frame"):
        print("  Detecting viewer crop...", flush=True)
        viewer_crop = detect_crop(viewer_path, viewer_duration,
                                  viewer_info["resolution_x"], viewer_info["resolution_y"])
        if viewer_crop:
            print("  Crop detected: {}x{} at ({},{}) - applying during DTW.".format(
                viewer_crop["w"], viewer_crop["h"],
                viewer_crop["x"], viewer_crop["y"]), flush=True)
        else:
            print("  No crop detected.", flush=True)

    # Full-frame DTW. Band centered on diagonal (speed=1.0, offset=0.0).
    # The band of 10000 frames (~417s) accommodates any realistic initial
    # file-start offset. Initial viewer offset is read from path[0] after
    # traceback; per-break frame deltas come from path deviation around each
    # detected black segment.
    print("  Running DTW (full-frame, band={} frames = {:.0f}s)...".format(
        dtw_band, dtw_band / fps), flush=True)
    try:
        dtw_result = dtw_align.run_dtw(
            phash_seq, viewer_path, fps,
            crop=viewer_crop,
            band_frames=dtw_band,
            max_mem_mb=dtw_max_mem_mb,
            tmp_dir=work_dir,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print("  ERROR: DTW failed: {}".format(exc), file=sys.stderr)
        return {"slot_id": slot_id, "status": "error"}

    if dtw_result is None:
        print("  ERROR: DTW returned no result.", file=sys.stderr)
        return {"slot_id": slot_id, "status": "no_match"}

    initial_offset = dtw_result["initial_offset_seconds"]

    print("  DTW done: initial_offset={:.6f}s  diag_offset={:.6f}s  "
          "rms={:.2f}fr  max={:.2f}fr  breaks={}  segments={}".format(
              initial_offset,
              dtw_result["diag_offset"],
              dtw_result["rms_frames"],
              dtw_result["max_frames"],
              len(dtw_result["author_breaks"]),
              len(dtw_result["segments"])), flush=True)

    trim = compute_trim_points(source, initial_offset, viewer_duration)

    transform = {
        "offset_seconds":        round(initial_offset, 6),
        "speed_factor":          1.0,
        "trim_start_seconds":    trim["trim_start_seconds"],
        "trim_duration_seconds": trim["trim_duration_seconds"],
        "fps_in":                viewer_info["fps"],
        "fps_out":               orig.get("fps"),
        "resolution_in":  [viewer_info["resolution_x"], viewer_info["resolution_y"]],
        "resolution_out": [orig.get("resolution_x"), orig.get("resolution_y")],
    }

    if viewer_crop:
        transform["crop"] = viewer_crop

    return {
        "slot_id":        slot_id,
        "slot_name":      source.get("name", ""),
        "status":         "suitable",
        "match_method":   "dtw",
        "input_file":     viewer_path,
        "transform":      transform,
        "output_filename": orig.get("filename",
                                   "{}_conformed.mkv".format(slot_id)),
        "dtw": {
            "initial_offset_seconds": round(initial_offset, 6),
            "diag_offset":   dtw_result["diag_offset"],
            "rms_frames":    dtw_result["rms_frames"],
            "max_frames":    dtw_result["max_frames"],
            "n_author":      dtw_result["n_author"],
            "n_viewer":      dtw_result["n_viewer"],
            "path_length":   dtw_result["path_length"],
            "author_breaks": dtw_result["author_breaks"],
            "segments":      dtw_result["segments"],
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    slot_map = parse_slot_args(args.slot)

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

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

            result = match_slot(source, viewer_path, viewer_info,
                                dtw_band=args.dtw_band,
                                dtw_max_mem_mb=args.dtw_max_mem,
                                work_dir=args.work_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("  ERROR: {}".format(e), file=sys.stderr)
            result = {"slot_id": slot_id, "status": "error", "error": str(e)}

        results.append(result)
        status = result.get("status", "?")
        print("  -> {}".format(status), flush=True)
        if status != "suitable":
            all_suitable = False

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
    print("Overall status: {}".format(
        "SUITABLE" if all_suitable else "NOT SUITABLE"), flush=True)


if __name__ == "__main__":
    main()

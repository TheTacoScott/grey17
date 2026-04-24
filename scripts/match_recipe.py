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
import subprocess
import sys
import datetime

import yaml

from utils import compute_sha256, ffprobe_source, phash_distance, detect_crop
from utils import extract_phashes_pipe
import dtw_align

_POPCOUNT16 = [bin(i).count("1") for i in range(65536)]

# DTW parameters
# Full-frame (no downsampling). Band of 10000 frames covers ~417s at 24fps,
# which is larger than any realistic commercial break length difference.
DTW_BAND_FRAMES = 10000


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Match viewer files against a signed recipe and produce a conform plan.",
        epilog=(
            "Matching uses a sliding window search over the first and last "
            "--search-fraction of the viewer file to locate the author's start and end "
            "anchor sequences. Offset and speed are derived from those two matched points. "
            "A full-video DTW pass then refines the speed+offset using all frames."
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
    p.add_argument("--probe-frames", type=int, default=500,
                   metavar="<frames>",
                   help="Sliding window probe size in frames (default: 500 = ~21s at 24fps).")
    p.add_argument("--no-dtw", action="store_true",
                   help="Skip the DTW refinement pass and use only the anchor-based LSQ fit.")
    p.add_argument("--dtw-band", type=int, default=DTW_BAND_FRAMES,
                   metavar="<frames>",
                   help="Sakoe-Chiba half-band for DTW in original frame units "
                        "(default: {} = ~{:.0f}s at 24fps). Increase to handle "
                        "longer commercial break differences.".format(
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
# Constants
# ---------------------------------------------------------------------------

VIEWER_SEARCH_FRACTION_DEFAULT = 0.25
PROBE_FRAMES_DEFAULT = 500
# Only exit the probe window retry early when the match is nearly perfect.
PROBE_EARLY_EXIT_DIST = 0.5
# Maximum number of probe windows per endpoint. Always uses stride=probe_frames//2
# (half-overlap). Stops after this many windows regardless of sequence length.
PROBE_MAX_WINDOWS = 48

# Sub-frame refinement: try +-SUBFRAME_STEPS * (1/fps / SUBFRAME_STEPS) offsets.
# With SUBFRAME_STEPS=10, this is +-1 frame in 0.1-frame increments (21 candidates).
SUBFRAME_STEPS = 10
SUBFRAME_COMPARE_FRAMES = 50  # frames compared per candidate (fast)

# ---------------------------------------------------------------------------
# Viewer frame extraction
# ---------------------------------------------------------------------------

def extract_viewer_endpoint_frames(viewer_path, start_tc, duration_secs, fps, crop=None):
    """
    Extract viewer frames at the given fps for a search window.
    If crop is a dict {w, h, x, y}, it is applied before scaling to strip black bars.
    Returns list of pHash hex strings (one per frame), or empty list on failure.

    Uses extract_phashes_pipe from utils for rawvideo pipe extraction.
    """
    n_frames = int(duration_secs * fps) + 2  # slight over-request; pipe stops at EOF anyway
    hashes = extract_phashes_pipe(viewer_path, start_tc, fps, crop=crop, n_frames=n_frames)
    if not hashes:
        print("  WARNING: viewer frame extraction produced no frames", file=sys.stderr)
    return hashes

# ---------------------------------------------------------------------------
# Sliding window search
# ---------------------------------------------------------------------------

def sliding_window_search(probe, search):
    """
    Slide probe (list of pHash hex strings) over search (list of pHash hex strings).
    Score at each position = mean pHash Hamming distance (lower = better).
    Returns (best_position_index, best_mean_distance).

    Preconverts all hex strings to ints once to avoid repeated int() calls.
    """
    n_probe = len(probe)
    n_search = len(search)

    if n_probe == 0 or n_search < n_probe:
        return 0, 64.0

    _PC = _POPCOUNT16  # local reference for speed
    probe_ints  = [int(h, 16) for h in probe]
    search_ints = [int(h, 16) for h in search]

    best_pos = 0
    best_score = 64.0

    for i in range(n_search - n_probe + 1):
        total = 0
        for j in range(n_probe):
            x = probe_ints[j] ^ search_ints[i + j]
            total += (_PC[x & 0xFFFF] + _PC[(x >> 16) & 0xFFFF] +
                      _PC[(x >> 32) & 0xFFFF] + _PC[(x >> 48) & 0xFFFF])
        mean_dist = total / n_probe
        if mean_dist < best_score:
            best_score = mean_dist
            best_pos = i

    return best_pos, best_score


def _search_probe_windows(anchors, search_frames, fps, search_start_tc, anchor_base_tc,
                          probe_frames, from_end, label):
    """
    Exhaustively try all overlapping probe windows from the anchor sequence.
    anchors and search_frames are lists of pHash hex strings.
    Each window is probe_frames wide; windows are spaced probe_frames//2 apart.

    from_end=False: windows start at 0, stride forward (start anchors).
    from_end=True:  windows start at the tail, stride backward (end anchors).

    anchor_base_tc is the author-source timecode of anchors[0].
    search_start_tc is the viewer timecode of search_frames[0].

    Returns dict {viewer_tc, author_tc, score, probe} for the best window found.
    probe is the list of pHash strings for the winning window (used for sub-frame
    refinement).
    """
    n = len(anchors)
    stride = max(1, probe_frames // 2)
    best = {
        "viewer_tc": search_start_tc,
        "author_tc": anchor_base_tc,
        "score": 64.0,
        "probe": [],
    }

    attempt = 0
    while True:
        if from_end:
            win_end = n - attempt * stride
            win_start = win_end - probe_frames
        else:
            win_start = attempt * stride
            win_end = win_start + probe_frames

        if win_start < 0 or win_end > n or attempt >= PROBE_MAX_WINDOWS:
            break

        probe = anchors[win_start:win_end]
        author_tc = anchor_base_tc + win_start / fps

        pos, score = sliding_window_search(probe, search_frames)
        viewer_tc = search_start_tc + pos / fps

        is_best = score < best["score"]
        if is_best:
            best = {
                "viewer_tc": viewer_tc,
                "author_tc": author_tc,
                "score": score,
                "probe": probe,
            }

        print("  {} window {}: author={:.1f}s  viewer={:.3f}s  dist={:.2f}{}".format(
            label, attempt, author_tc, viewer_tc, score, " *" if is_best else ""),
            flush=True)

        if best["score"] <= PROBE_EARLY_EXIT_DIST:
            break

        attempt += 1

    print("  {} best: viewer={:.3f}s  author={:.1f}s  dist={:.2f}".format(
        label, best["viewer_tc"], best["author_tc"], best["score"]), flush=True)
    return best

# ---------------------------------------------------------------------------
# Sub-frame refinement
# ---------------------------------------------------------------------------

def subframe_refine(viewer_path, coarse_viewer_tc, probe_hashes, fps, crop=None):
    """
    Refine coarse_viewer_tc by trying +-1 frame in 0.1-frame steps (21 candidates).
    Uses SUBFRAME_COMPARE_FRAMES frames per candidate for speed.
    probe_hashes: list of pHash strings from the winning coarse probe window.
    Returns (refined_viewer_tc, refined_score).
    """
    n = min(SUBFRAME_COMPARE_FRAMES, len(probe_hashes))
    probe = probe_hashes[:n]

    step_secs = 1.0 / (fps * SUBFRAME_STEPS)  # 0.1-frame step in seconds
    best_tc = coarse_viewer_tc
    best_score = 64.0

    for i in range(-SUBFRAME_STEPS, SUBFRAME_STEPS + 1):
        candidate_tc = max(0.0, coarse_viewer_tc + i * step_secs)
        frames = extract_phashes_pipe(viewer_path, candidate_tc, fps, crop=crop, n_frames=n)
        if len(frames) < n:
            continue
        total = sum(phash_distance(probe[j], frames[j]) for j in range(n))
        score = total / n
        if score < best_score:
            best_score = score
            best_tc = candidate_tc

    return best_tc, best_score

# ---------------------------------------------------------------------------
# Endpoint matching (visual pHash)
# ---------------------------------------------------------------------------

def endpoint_match(source, viewer_path, viewer_duration,
                   search_fraction=VIEWER_SEARCH_FRACTION_DEFAULT,
                   probe_frames=PROBE_FRAMES_DEFAULT,
                   viewer_crop=None):
    """
    Find where the author's start and end anchors appear in the viewer file.

    Supports two recipe formats:
      phash_sequence  (new): list of pHash hex strings covering the entire video.
                             Split at midpoint: first half -> start search,
                             second half -> end search.
      start_anchors / end_anchors (legacy): separate lists of {phash, ahash} dicts
                             or plain strings.

    Returns dict with start_match_tc, end_match_tc, start_score, end_score,
    author_start_tc, author_end_tc. Or None if no anchor data is present.
    """
    orig = source.get("original") or {}
    fps = orig.get("fps") or 24.0
    author_duration = orig.get("duration_seconds") or viewer_duration
    search_duration = viewer_duration * search_fraction

    # Resolve anchor lists from recipe (plain string lists)
    phash_seq = source.get("phash_sequence")
    if phash_seq:
        # New format: list of pHash strings for entire video
        n_total = len(phash_seq)
        half = n_total // 2
        start_anchors = phash_seq[:half]
        end_anchors = phash_seq[half:]
        end_anchors_base_tc = half / fps
    else:
        # Legacy format: {phash, ahash} dicts or plain strings
        raw_start = source.get("start_anchors") or []
        raw_end   = source.get("end_anchors") or []
        if not raw_start or not raw_end:
            return None
        # Normalize to plain strings
        start_anchors = [a["phash"] if isinstance(a, dict) else a for a in raw_start]
        end_anchors   = [a["phash"] if isinstance(a, dict) else a for a in raw_end]
        end_anchors_base_tc = max(0.0, author_duration - len(end_anchors) / fps)

    # -- Start: extract viewer search frames, then probe windows forward --
    print("  Extracting viewer start search region ({:.0f}s)...".format(
        search_duration), flush=True)
    start_search = extract_viewer_endpoint_frames(
        viewer_path, 0.0, search_duration, fps, crop=viewer_crop)
    if not start_search:
        return None

    best_start = _search_probe_windows(
        start_anchors, start_search, fps,
        search_start_tc=0.0,
        anchor_base_tc=0.0,
        probe_frames=probe_frames,
        from_end=False,
        label="start",
    )

    # Sub-frame refinement for start endpoint (TC only - score stays the coarse 500-frame value)
    if best_start["probe"] and best_start["score"] < 5.0:
        print("  Sub-frame refining start...", flush=True)
        refined_tc, refined_score = subframe_refine(
            viewer_path, best_start["viewer_tc"], best_start["probe"], fps, crop=viewer_crop)
        if refined_tc != best_start["viewer_tc"]:
            print("  start sub-frame: {:.6f}s -> {:.6f}s  (50-frame dist {:.2f})".format(
                best_start["viewer_tc"], refined_tc, refined_score),
                flush=True)
            best_start["viewer_tc"] = refined_tc

    # -- End: extract viewer search frames, then probe windows backward from tail --
    end_search_start = max(0.0, viewer_duration - search_duration)
    print("  Extracting viewer end search region ({:.0f}s from {:.0f}s)...".format(
        search_duration, end_search_start), flush=True)
    end_search = extract_viewer_endpoint_frames(
        viewer_path, end_search_start, search_duration, fps, crop=viewer_crop)
    if not end_search:
        return None

    best_end = _search_probe_windows(
        end_anchors, end_search, fps,
        search_start_tc=end_search_start,
        anchor_base_tc=end_anchors_base_tc,
        probe_frames=probe_frames,
        from_end=True,
        label="end",
    )

    # Sub-frame refinement for end endpoint (TC only - score stays the coarse 500-frame value)
    if best_end["probe"] and best_end["score"] < 5.0:
        print("  Sub-frame refining end...", flush=True)
        refined_tc, refined_score = subframe_refine(
            viewer_path, best_end["viewer_tc"], best_end["probe"], fps, crop=viewer_crop)
        if refined_tc != best_end["viewer_tc"]:
            print("  end sub-frame: {:.6f}s -> {:.6f}s  (50-frame dist {:.2f})".format(
                best_end["viewer_tc"], refined_tc, refined_score),
                flush=True)
            best_end["viewer_tc"] = refined_tc

    # -- Mid-point anchors for a better-than-two-point linear fit --
    # Use the rough (start+end) model to predict where to look, then refine.
    anchor_list = [best_start, best_end]

    if phash_seq:
        orig = source.get("original") or {}
        author_duration = orig.get("duration_seconds") or viewer_duration

        rough_author_span = best_end["author_tc"] - best_start["author_tc"]
        rough_viewer_span = best_end["viewer_tc"]  - best_start["viewer_tc"]
        if rough_author_span > 0:
            rough_speed  = rough_viewer_span / rough_author_span
            rough_offset = best_start["viewer_tc"] - best_start["author_tc"] * rough_speed
        else:
            rough_speed, rough_offset = 1.0, 0.0

        for frac in MID_ANCHOR_FRACTIONS:
            mid_author_tc = author_duration * frac
            # Skip fractions that fall within the start/end search regions
            # (already covered by those anchor probes).
            if mid_author_tc < best_start["author_tc"] + probe_frames / fps:
                continue
            if mid_author_tc > best_end["author_tc"] - probe_frames / fps:
                continue

            mid = _find_mid_anchor(
                viewer_path, mid_author_tc, phash_seq, fps,
                rough_speed, rough_offset, probe_frames, crop=viewer_crop)
            if mid:
                anchor_list.append(mid)
                print("  mid anchor {:.0f}s: viewer={:.6f}s  dist={:.2f}".format(
                    mid_author_tc, mid["viewer_tc"], mid["score"]), flush=True)
            else:
                print("  mid anchor {:.0f}s: no match".format(mid_author_tc), flush=True)

    # Mean offset (speed=1.0) across all good anchors.
    anchor_list.sort(key=lambda x: x["author_tc"])
    fitted_offset = _offset_from_anchors(anchor_list)
    print("  LSQ fit over {} anchors: speed=1.0  offset={:.6f}s".format(
        len(anchor_list), fitted_offset), flush=True)

    return {
        "start_match_tc":  best_start["viewer_tc"],
        "end_match_tc":    best_end["viewer_tc"],
        "start_score":     round(best_start["score"], 4),
        "end_score":       round(best_end["score"],   4),
        "author_start_tc": best_start["author_tc"],
        "author_end_tc":   best_end["author_tc"],
        "fitted_offset":   fitted_offset,
        "n_anchors":       len(anchor_list),
        "anchors":         anchor_list,
    }

# ---------------------------------------------------------------------------
# Mid-anchor search (for multi-point linear fit)
# ---------------------------------------------------------------------------

# Number of mid-point anchors to add between start and end.
# At fractions [0.25, 0.50, 0.75] of author duration.
# More anchors -> better speed accuracy -> closer to frame-perfect.
MID_ANCHOR_FRACTIONS = [0.25, 0.50, 0.75]

# Local search slack around the predicted viewer position (seconds).
# Covers up to ~10x the expected rough-model error.
MID_ANCHOR_SLACK_SECS = 3.0

# Only keep mid-anchors whose match score is below this (same as subframe gate).
MID_ANCHOR_MAX_DIST = 5.0


def _find_mid_anchor(viewer_path, author_tc, phash_seq, fps,
                     rough_speed, rough_offset, probe_frames, crop=None):
    """
    Locate the viewer frame corresponding to author_tc using a local sliding
    window search around the rough linear model prediction.

    probe_frames frames of phash_seq starting at author_tc are used as the probe.
    The viewer search region is MID_ANCHOR_SLACK_SECS on each side of the prediction.

    Returns dict {viewer_tc, author_tc, score} on success, or None on failure.
    """
    probe_start_frame = int(author_tc * fps)
    probe_end_frame = probe_start_frame + probe_frames
    if probe_end_frame > len(phash_seq):
        return None
    probe = phash_seq[probe_start_frame:probe_end_frame]
    if len(probe) < probe_frames // 2:
        return None

    predicted_viewer_tc = author_tc * rough_speed + rough_offset
    search_start = max(0.0, predicted_viewer_tc - MID_ANCHOR_SLACK_SECS)
    search_duration = 2.0 * MID_ANCHOR_SLACK_SECS + probe_frames / fps

    viewer_frames = extract_viewer_endpoint_frames(
        viewer_path, search_start, search_duration, fps, crop=crop)
    if len(viewer_frames) < len(probe):
        return None

    pos, score = sliding_window_search(probe, viewer_frames)
    if score > MID_ANCHOR_MAX_DIST:
        return None

    viewer_tc = search_start + pos / fps

    # Sub-frame refine
    refined_tc, _ = subframe_refine(viewer_path, viewer_tc, probe, fps, crop=crop)

    return {"viewer_tc": refined_tc, "author_tc": author_tc, "score": score}


def _offset_from_anchors(anchors):
    """
    Compute the timing offset between viewer and author with speed fixed at 1.0.
    Returns the mean of (viewer_tc - author_tc) across all anchors.
    """
    if not anchors:
        return 0.0
    return sum(x["viewer_tc"] - x["author_tc"] for x in anchors) / len(anchors)


# ---------------------------------------------------------------------------
# Audio fingerprint matching
# ---------------------------------------------------------------------------

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

def match_slot(source, viewer_path, viewer_info, threshold,
               search_fraction=VIEWER_SEARCH_FRACTION_DEFAULT,
               probe_frames=PROBE_FRAMES_DEFAULT,
               use_dtw=True, dtw_band=DTW_BAND_FRAMES,
               dtw_max_mem_mb=256, work_dir=None):
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

    has_phash_seq = bool(source.get("phash_sequence"))
    has_legacy_anchors = bool(source.get("start_anchors") and source.get("end_anchors"))
    if not has_phash_seq and not has_legacy_anchors:
        print("  ERROR: recipe has no phash_sequence or start/end anchors for {}. "
              "Re-sign the recipe.".format(slot_id), file=sys.stderr)
        return {"slot_id": slot_id, "status": "no_match", "match_rate": 0.0}

    # Detect crop before matching so bars are stripped from viewer frames.
    # Only done when the recipe marks this source as full-frame.
    viewer_crop = None
    if source.get("expect_full_frame"):
        print("  Detecting viewer crop...", flush=True)
        viewer_crop = detect_crop(viewer_path, viewer_duration,
                                  viewer_info["resolution_x"], viewer_info["resolution_y"])
        if viewer_crop:
            print("  Crop detected: {}x{} at ({},{}) - applying during matching.".format(
                viewer_crop["w"], viewer_crop["h"], viewer_crop["x"], viewer_crop["y"]),
                flush=True)
        else:
            print("  No crop detected.", flush=True)

    # -- Visual endpoint match --
    ep = endpoint_match(source, viewer_path, viewer_duration,
                        search_fraction=search_fraction,
                        probe_frames=probe_frames,
                        viewer_crop=viewer_crop)
    if ep is None:
        print("  ERROR: endpoint match returned no result", file=sys.stderr)
        return {"slot_id": slot_id, "status": "no_match", "match_rate": 0.0}

    # Compute timing offset (speed=1.0) from anchor matches.
    # Use the full anchor list from endpoint_match when available, otherwise
    # fall back to the two raw endpoint TCs.
    anchor_list = ep.get("anchors") or [
        {"author_tc": ep.get("author_start_tc", 0.0),
         "viewer_tc": ep.get("start_match_tc",  0.0)},
        {"author_tc": ep.get("author_end_tc",   0.0),
         "viewer_tc": ep.get("end_match_tc",    0.0)},
    ]
    visual_offset = ep.get("fitted_offset") or _offset_from_anchors(anchor_list)

    start_quality = max(0.0, 1.0 - ep["start_score"] / 32.0)
    end_quality   = max(0.0, 1.0 - ep["end_score"]   / 32.0)
    visual_match_rate = (start_quality + end_quality) / 2.0

    print("  Endpoint match: offset={:.3f}s  match_rate={:.1%}  anchors={}".format(
        visual_offset, visual_match_rate,
        ep.get("n_anchors", 2)), flush=True)

    suitable = visual_match_rate >= threshold
    match_method = "visual" if has_phash_seq else "endpoint"

    # -- DTW refinement --
    # Uses the anchor offset to center the Sakoe-Chiba band. The DTW path
    # then measures per-break frame deltas (including fade transitions that
    # fall outside the detected black interval) via path deviation before/after
    # each break, producing accurate viewer_start_tc / viewer_end_tc values
    # for the segmented conform.
    dtw_result = None
    fps = orig.get("fps") or 24.0
    phash_seq = source.get("phash_sequence")
    if use_dtw and has_phash_seq and phash_seq and suitable:
        print("  Running DTW refinement (full-frame, band={})...".format(
            dtw_band), flush=True)
        try:
            dtw_result = dtw_align.run_dtw(
                phash_seq, viewer_path,
                1.0, visual_offset, fps,
                crop=viewer_crop,
                band_frames=dtw_band,
                max_mem_mb=dtw_max_mem_mb,
                tmp_dir=work_dir,
            )
        except Exception as exc:
            print("  WARNING: DTW failed: {}".format(exc), file=sys.stderr)
            dtw_result = None

    # Offset for the transform and simple conform: use the start-anchor TC
    # difference (viewer_tc - author_tc at t=0). This is unaffected by any
    # break length differences that accumulate later in the video.
    start_anchor_offset = (ep.get("start_match_tc", 0.0)
                           - ep.get("author_start_tc", 0.0))
    final_offset = start_anchor_offset

    if dtw_result:
        print("  DTW done: offset={:.6f}s  rms={:.2f}fr  breaks={}  segments={}".format(
            dtw_result["diag_offset"], dtw_result["rms_frames"],
            len(dtw_result["author_breaks"]), len(dtw_result["segments"])), flush=True)

    trim = compute_trim_points(source, final_offset, viewer_duration)

    transform = {
        "offset_seconds":        round(final_offset, 6),
        "speed_factor":          1.0,
        "trim_start_seconds":    trim["trim_start_seconds"],
        "trim_duration_seconds": trim["trim_duration_seconds"],
        "fps_in":                viewer_info["fps"],
        "fps_out":               orig.get("fps"),
        "resolution_in":  [viewer_info["resolution_x"], viewer_info["resolution_y"]],
        "resolution_out": [orig.get("resolution_x"), orig.get("resolution_y")],
    }

    if viewer_crop and suitable:
        transform["crop"] = viewer_crop

    result = {
        "slot_id":       slot_id,
        "slot_name":     source.get("name", ""),
        "status":        "suitable" if suitable else "unsuitable",
        "match_rate":    round(visual_match_rate, 4),
        "match_method":  match_method,
        "start_score":   ep["start_score"],
        "end_score":     ep["end_score"],
        "anchor_offset": round(visual_offset, 6),
        "input_file":    viewer_path,
        "transform":     transform,
        "output_filename": orig.get("filename", "{}_conformed.mkv".format(slot_id)),
    }

    if dtw_result:
        result["dtw"] = {
            "diag_offset":   dtw_result["diag_offset"],
            "rms_frames":    dtw_result["rms_frames"],
            "max_frames":    dtw_result["max_frames"],
            "n_author":      dtw_result["n_author"],
            "n_viewer":      dtw_result["n_viewer"],
            "path_length":   dtw_result["path_length"],
            "author_breaks": dtw_result["author_breaks"],
            "segments":      dtw_result["segments"],
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

            result = match_slot(source, viewer_path, viewer_info, args.threshold,
                                search_fraction=args.search_fraction,
                                probe_frames=args.probe_frames,
                                use_dtw=not args.no_dtw,
                                dtw_band=args.dtw_band,
                                dtw_max_mem_mb=args.dtw_max_mem,
                                work_dir=args.work_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
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

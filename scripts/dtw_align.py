#!/usr/bin/env python3
"""
Full-frame DTW alignment for match_recipe.py.

Runs a Sakoe-Chiba banded DTW at native fps with NO downsampling. Every frame
of both sequences is compared. The band is centered on the linear model
prediction from the anchor search (not the true diagonal), so it tracks
correctly for non-zero offsets.

Grid size: for a 43-minute episode at 23.976fps with band=10000 frames, the
band covers 20001 cells per row, totalling ~1.24 billion cell operations.
Runtime is roughly 5-15 minutes in pure Python. Disk usage for backpointer
storage is approximately n_rows * 2*band bytes, e.g. ~1.2GB for the above.

Backpointer storage: the DP table is processed row by row (only prev/curr
rows are live at any time: O(m) memory). Backpointers (one byte per cell in
the band) are accumulated in horizontal stripes of stripe_height rows each.
When total in-memory stripe data exceeds max_mem_mb, the oldest stripes are
flushed to a temp file on disk and reloaded during traceback. This bounds RAM
usage regardless of sequence length or band width.

Path objective: minimize total accumulated Hamming distance. This IS standard
DTW: each edge (i-1,j-1)->(i,j), (i-1,j)->(i,j), or (i,j-1)->(i,j) carries
cost hamming(author[i], viewer[j]) at the destination, and the DP computes
the minimum cumulative cost from (0,0) to each cell. Traceback recovers the
cheapest (lowest total Hamming) path, not the shortest (fewest steps) path.

Band centering: the band is always centered on the diagonal. The default
half-band of 10000 frames (~417s at 24fps) accommodates any realistic
initial offset between a viewer file and the author source without a
pre-search step.

Black segment detection: the author pHash sequence is scanned for runs of
near-black frames. Each detected break is mapped to its corresponding viewer
timecode range using the DTW path, giving the viewer break duration. The
delta (viewer_frames - author_frames) per break quantifies how many frames
the consumer has extra or fewer, which a downstream segmented conform can use
to achieve frame-perfect alignment. Break boundaries are then refined to
half-frame precision via a targeted local search.
"""

import os
import struct
import tempfile
from utils import extract_phashes_pipe

# bisect is stdlib - used in _map_breaks_via_path
import bisect

_PC = [bin(i).count("1") for i in range(65536)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred_j(i, viewer_fps, fps, m):
    """
    Predicted viewer frame index for author frame i.
    Band is always centered on the diagonal (speed=1.0, offset=0.0).
    viewer_frame = int(i * viewer_fps / fps)
    """
    j = int(i * viewer_fps / fps)
    return max(0, min(m - 1, j))


def _lsq_offset(path, fps):
    """
    Least-squares offset with speed fixed at 1.0:
        viewer_tc = author_tc + offset
    Returns offset in seconds (mean of viewer_tc - author_tc).
    Used only for diagnostics; conform always uses deviation at path start.
    """
    if not path:
        return 0.0
    return sum(j / fps - i / fps for i, j in path) / len(path)


# ---------------------------------------------------------------------------
# Black segment detection
# ---------------------------------------------------------------------------

def detect_black_segments(phash_sequence, fps, min_frames=12, black_threshold=10):
    """
    Scan a pHash sequence for runs of near-black frames.

    A frame is "black" when its Hamming distance from 0x0000000000000000
    (the pHash of a uniformly black 32x32 image) is <= black_threshold.
    Threshold 10 accommodates small noise, logos, or subtle gradients that
    survive the 32x32 downscale.

    Runs shorter than min_frames are ignored to avoid spurious detections on
    scene-cut flashes or single-frame dips.

    Returns list of dicts with keys:
        start_frame, end_frame     -- inclusive indices in phash_sequence
        start_tc, end_tc           -- timecodes in seconds
        duration_frames            -- frame count
        duration_secs              -- duration in seconds
    """
    segments = []
    in_black = False
    run_start = 0

    for i, h in enumerate(phash_sequence):
        x = int(h, 16)
        dist = (_PC[x & 0xFFFF] + _PC[(x >> 16) & 0xFFFF] +
                _PC[(x >> 32) & 0xFFFF] + _PC[(x >> 48) & 0xFFFF])
        is_black = dist <= black_threshold

        if is_black and not in_black:
            in_black = True
            run_start = i
        elif not is_black and in_black:
            in_black = False
            dur = i - run_start
            if dur >= min_frames:
                # end_frame is exclusive (one past last black frame), matching
                # Python slice convention. end_tc is the TC of the first frame
                # after the black run. _compute_segment_transforms relies on
                # this convention to start content segments at the correct frame.
                segments.append({
                    "start_frame":    run_start,
                    "end_frame":      i,
                    "start_tc":       run_start / fps,
                    "end_tc":         i / fps,
                    "duration_frames": dur,
                    "duration_secs":  dur / fps,
                })

    if in_black:
        dur = len(phash_sequence) - run_start
        if dur >= min_frames:
            end_ex = len(phash_sequence)
            segments.append({
                "start_frame":    run_start,
                "end_frame":      end_ex,
                "start_tc":       run_start / fps,
                "end_tc":         end_ex / fps,
                "duration_frames": dur,
                "duration_secs":  dur / fps,
            })

    return segments


def _map_breaks_via_path(author_breaks, path, fps):
    """
    Map each author black-segment break to its viewer timecodes by measuring
    the DTW path deviation just before and just after the break interval.

    Why deviation-based rather than within-interval frame counting:
    The detected black interval captures only the pure-black core of a break.
    Fade-to-black and fade-from-black transitions around the core are near-black
    but above the detection threshold, so they are NOT included in the detected
    interval. If HMAX has longer fades than the author, those extra frames sit
    outside the detected interval and the within-interval frame count shows
    delta=0 even though the true break is longer.

    Measuring the path deviation (viewer_frame - author_frame) in a stable
    content window before the break and again after gives the true accumulated
    frame delta, including the fade frames on both sides.

    Returns a list of dicts per break:
        author_start_frame, author_end_frame
        author_start_tc, author_end_tc, author_duration_frames
        viewer_start_tc   -- viewer TC at author break start (seek target before break)
        viewer_end_tc     -- viewer TC at author break end (seek target after break)
        frame_delta       -- total viewer - author frames across the full break region
                             positive = viewer has MORE frames (longer black/fades)
                             negative = viewer has FEWER
    """
    if not author_breaks or not path:
        return []

    # Build sorted arrays for binary search over the path.
    path_a = [p[0] for p in path]
    path_d = [p[1] - p[0] for p in path]  # deviation = viewer_frame - author_frame
    n_path  = len(path_a)

    # 2-second window of content frames for stable deviation measurement.
    WINDOW = max(12, int(2.0 * fps))

    def median_deviation(lo_frame, hi_frame):
        """Median path deviation over author frames in [lo_frame, hi_frame]."""
        lo = bisect.bisect_left(path_a, lo_frame)
        hi = bisect.bisect_right(path_a, hi_frame)
        seg = path_d[lo:hi]
        if not seg:
            # No path points in range; use nearest available point.
            k = max(0, min(bisect.bisect_left(path_a, lo_frame), n_path - 1))
            return path_d[k]
        s = sorted(seg)
        return s[len(s) // 2]

    result = []
    for b in author_breaks:
        # Measure deviation in content just before the break starts.
        d_before = median_deviation(b["start_frame"] - WINDOW, b["start_frame"])
        # Measure deviation in content just after the break ends.
        d_after  = median_deviation(b["end_frame"],  b["end_frame"] + WINDOW)

        # frame_delta: how many MORE viewer frames span the full break region
        # (including fades outside the detected pure-black interval).
        frame_delta = int(round(d_after - d_before))

        # Viewer TC where the break region starts (= where content extraction
        # before this break should stop seeking, i.e. exclude viewer's fades).
        # Using author_start_frame + d_before positions us at the equivalent
        # frame in the viewer just before the break.
        viewer_start_frame = b["start_frame"] + d_before
        # Viewer TC where content resumes after the break (skipping viewer's
        # fades on the far side).
        viewer_end_frame   = b["end_frame"] + d_after

        result.append({
            "author_start_frame":     b["start_frame"],
            "author_end_frame":       b["end_frame"],
            "author_start_tc":        round(b["start_tc"], 4),
            "author_end_tc":          round(b["end_tc"], 4),
            "author_duration_frames": b["duration_frames"],
            "viewer_start_tc":        round(viewer_start_frame / fps, 6),
            "viewer_end_tc":          round(viewer_end_frame   / fps, 6),
            "frame_delta":            frame_delta,
        })

    return result


# ---------------------------------------------------------------------------
# Chunked stripe backpointer storage
# ---------------------------------------------------------------------------

class _StripeStore:
    """
    Manages DTW backpointer data in horizontal stripes to bound RAM usage.

    Each stripe covers stripe_height rows. A stripe's data is a list of
    (j_lo: int, dirs: bytearray), one entry per row in the stripe.

    Forward pass: stripes accumulate in self._mem. When len(self._mem) exceeds
    max_mem_stripes, the oldest (lowest-index) stripe is serialized to a
    shared temp file and evicted from RAM.

    Traceback: accesses rows in strictly decreasing index order. Stripes are
    loaded from disk on demand. Each stripe is released from RAM via release()
    once the traceback has moved past it, so at most ~2 stripes are live.

    Direction byte codes: 0=diagonal(i-1,j-1)  1=left(i,j-1)  2=up(i-1,j)
    """

    def __init__(self, stripe_height, max_mem_stripes, tmp_dir=None):
        self._sh = stripe_height
        self._max_mem = max(2, max_mem_stripes)
        self._tmp_dir = tmp_dir

        self._mem = {}       # stripe_idx -> list[(j_lo, bytearray)]
        self._disk = {}      # stripe_idx -> (file_offset, byte_count)
        self._disk_fh = None
        self._disk_path = None

        self._cur_idx = None
        self._cur_rows = None

    # -- Forward pass API --

    def push_row(self, i, j_lo, dirs):
        """Append backpointer data for row i."""
        sidx = i // self._sh
        if sidx != self._cur_idx:
            self._seal()
            self._cur_idx = sidx
            self._cur_rows = []
        self._cur_rows.append((j_lo, dirs))

    def finish(self):
        """Call after all rows have been pushed."""
        self._seal()

    def _seal(self):
        if self._cur_idx is None or not self._cur_rows:
            return
        self._mem[self._cur_idx] = self._cur_rows
        self._cur_idx = None
        self._cur_rows = None
        self._evict()

    def _evict(self):
        while len(self._mem) > self._max_mem:
            oldest = min(self._mem.keys())
            self._flush_to_disk(oldest)
            del self._mem[oldest]

    def _open_disk(self):
        if self._disk_fh is None:
            fd, path = tempfile.mkstemp(
                suffix=".dtwback",
                dir=self._tmp_dir or tempfile.gettempdir(),
            )
            os.close(fd)
            self._disk_fh = open(path, "w+b")
            self._disk_path = path

    def _flush_to_disk(self, sidx):
        self._open_disk()
        f = self._disk_fh
        f.seek(0, 2)
        start = f.tell()
        for j_lo, dirs in self._mem[sidx]:
            f.write(struct.pack(">iI", j_lo, len(dirs)))
            f.write(bytes(dirs))
        end = f.tell()
        self._disk[sidx] = (start, end - start)
        print("  [DTW] stripe {} -> disk ({:.0f}MB at offset {:.0f}MB)".format(
            sidx, (end - start) / 1e6, start / 1e6), flush=True)

    def _load_from_disk(self, sidx):
        start, nbytes = self._disk[sidx]
        self._disk_fh.seek(start)
        data = self._disk_fh.read(nbytes)
        rows = []
        pos = 0
        while pos < len(data):
            j_lo, dlen = struct.unpack(">iI", data[pos:pos + 8])
            pos += 8
            rows.append((j_lo, bytearray(data[pos:pos + dlen])))
            pos += dlen
        return rows

    # -- Traceback API --

    def get_row(self, i):
        """Return (j_lo, dirs) for row i. Loads stripe from disk if needed."""
        sidx = i // self._sh
        if sidx not in self._mem:
            if sidx not in self._disk:
                return None
            self._mem[sidx] = self._load_from_disk(sidx)
        rows = self._mem[sidx]
        k = i - sidx * self._sh
        return rows[k] if 0 <= k < len(rows) else None

    def release(self, sidx):
        """Free stripe sidx from RAM. Call when traceback moves past it."""
        self._mem.pop(sidx, None)

    def cleanup(self):
        """Delete temp file. Call after traceback completes."""
        if self._disk_fh is not None:
            self._disk_fh.close()
            self._disk_fh = None
            try:
                os.unlink(self._disk_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Forward DP pass
# ---------------------------------------------------------------------------

def _forward_pass(author_ints, viewer_ints, fps, viewer_fps, band, store):
    """
    Fill the DTW DP table row by row.

    Row 0 is initialized with cumulative-left costs (no up or diagonal
    predecessors exist for the first row). Rows 1..n-1 use the standard
    three-predecessor recurrence.

    The first cell in each row i>=1 (k=0) has no left predecessor, so it
    is handled separately to avoid a branch in the hot inner loop.

    Working state: prev (the last completed row) and curr (being filled).
    Both are full-width float lists; only band cells are written, the rest
    remain INF. List allocation per row is fast in Python and the GC
    handles memory churn.

    Backpointer bytes are pushed to store as each row completes.
    Returns the final accumulated-cost row (length m, INF outside band).
    """
    n = len(author_ints)
    m = len(viewer_ints)
    INF = float("inf")
    _P = _PC  # local ref avoids global lookup in hot loop
    vi = viewer_ints

    # ------------------------------------------------------------------
    # Row 0: seed with cumulative left costs.
    # Standard DTW: D[0,j] = cost(0,j) + D[0,j-1], D[0,j_lo] = cost(0,j_lo).
    # All up/diagonal predecessors are non-existent so treated as INF.
    # ------------------------------------------------------------------
    jc0 = _pred_j(0, viewer_fps, fps, m)
    j_lo_0 = max(0, jc0 - band)
    j_hi_0 = min(m - 1, jc0 + band)
    width_0 = j_hi_0 - j_lo_0 + 1

    prev = [INF] * m
    dirs_0 = bytearray(width_0)
    ai_0 = author_ints[0]

    j = j_lo_0
    x = ai_0 ^ vi[j]
    prev[j] = _P[x & 0xFFFF] + _P[(x >> 16) & 0xFFFF] + _P[(x >> 32) & 0xFFFF] + _P[(x >> 48) & 0xFFFF]
    dirs_0[0] = 0  # origin marker (direction unused at start of traceback)

    for k in range(1, width_0):
        j = j_lo_0 + k
        x = ai_0 ^ vi[j]
        d = _P[x & 0xFFFF] + _P[(x >> 16) & 0xFFFF] + _P[(x >> 32) & 0xFFFF] + _P[(x >> 48) & 0xFFFF]
        prev[j] = prev[j - 1] + d
        dirs_0[k] = 1  # came from left

    store.push_row(0, j_lo_0, dirs_0)

    # ------------------------------------------------------------------
    # Rows 1..n-1: standard three-predecessor recurrence.
    # ------------------------------------------------------------------
    for i in range(1, n):
        jc = _pred_j(i, viewer_fps, fps, m)
        j_lo = max(0, jc - band)
        j_hi = min(m - 1, jc + band)
        width = j_hi - j_lo + 1

        curr = [INF] * m
        dirs = bytearray(width)
        ai = author_ints[i]

        # k=0: no left predecessor
        j = j_lo
        x = ai ^ vi[j]
        d = _P[x & 0xFFFF] + _P[(x >> 16) & 0xFFFF] + _P[(x >> 32) & 0xFFFF] + _P[(x >> 48) & 0xFFFF]
        c_diag = prev[j - 1] if j > 0 else INF
        c_up   = prev[j]
        best = c_diag if c_diag <= c_up else c_up
        curr[j] = best + d
        dirs[0] = 0 if best == c_diag else 2

        # k=1..width-1: all three predecessors available (left always in-band)
        for k in range(1, width):
            j = j_lo + k
            x = ai ^ vi[j]
            d = _P[x & 0xFFFF] + _P[(x >> 16) & 0xFFFF] + _P[(x >> 32) & 0xFFFF] + _P[(x >> 48) & 0xFFFF]
            c_diag = prev[j - 1]
            c_left = curr[j - 1]
            c_up   = prev[j]
            best = min(c_diag, c_left, c_up)
            curr[j] = best + d
            if best == c_diag:
                dirs[k] = 0
            elif best == c_left:
                dirs[k] = 1
            else:
                dirs[k] = 2

        store.push_row(i, j_lo, dirs)
        prev = curr

        if i % 5000 == 0:
            print("  [DTW] forward {}/{}".format(i, n), flush=True)

    store.finish()
    return prev


# ---------------------------------------------------------------------------
# Traceback
# ---------------------------------------------------------------------------

def _traceback(n, m, final_row, fps, viewer_fps, band, store):
    """
    Trace the minimum-cost path from the best endpoint in the final row back
    to (0, 0). Releases stripes from RAM as traceback moves to older rows.
    Returns path as list of (author_idx, viewer_idx) in ascending order.
    """
    INF = float("inf")
    sh = store._sh

    # Best endpoint: minimum cost within band on final row
    jc = _pred_j(n - 1, viewer_fps, fps, m)
    j_lo_last = max(0, jc - band)
    j_hi_last = min(m - 1, jc + band)
    best_j, best_cost = j_lo_last, INF
    for j in range(j_lo_last, j_hi_last + 1):
        c = final_row[j]
        if c < best_cost:
            best_cost = c
            best_j = j

    path = []
    i, j = n - 1, best_j
    cur_stripe = i // sh

    while i > 0 or j > 0:
        path.append((i, j))

        if i == 0:
            j -= 1
            continue
        if j == 0:
            i -= 1
            new_s = i // sh
            if new_s < cur_stripe:
                store.release(cur_stripe)
                cur_stripe = new_s
            continue

        row_data = store.get_row(i)
        if row_data is None:
            # Band miss: step up as fallback
            i -= 1
            new_s = i // sh
            if new_s < cur_stripe:
                store.release(cur_stripe)
                cur_stripe = new_s
            continue

        j_lo_i, dirs_i = row_data
        k = j - j_lo_i
        if k < 0 or k >= len(dirs_i):
            # Outside stored band: step up as fallback
            i -= 1
            new_s = i // sh
            if new_s < cur_stripe:
                store.release(cur_stripe)
                cur_stripe = new_s
            continue

        d = dirs_i[k]
        if d == 0:
            i -= 1
            j -= 1
        elif d == 1:
            j -= 1
        else:
            i -= 1

        new_s = i // sh
        if new_s < cur_stripe:
            store.release(cur_stripe)
            cur_stripe = new_s

    path.append((0, 0))
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# Sub-frame refinement of break boundary TCs
# ---------------------------------------------------------------------------

def _hamming(a, b):
    x = a ^ b
    return (_PC[x & 0xFFFF] + _PC[(x >> 16) & 0xFFFF] +
            _PC[(x >> 32) & 0xFFFF] + _PC[(x >> 48) & 0xFFFF])


def _refine_break_boundary(viewer_path, author_ints, b, fps,
                            probe_frames=60, half_frame_steps=4):
    """
    Refine viewer_end_tc for a single non-zero-delta break by testing
    sub-frame candidate positions around the DTW prediction.

    The DTW path gives viewer_end_tc accurate to +-1 frame (the median
    deviation is rounded to the nearest integer frame). This function
    tests positions at half-frame increments across a window of
    +-half_frame_steps frames, comparing probe_frames of author content
    starting at b["author_end_frame"] against the viewer at each candidate
    seek point.

    Returns the refined viewer_end_tc (float, seconds).
    """
    author_end_fr = b["author_end_frame"]
    probe_ints = author_ints[author_end_fr:author_end_fr + probe_frames]
    n_probe = len(probe_ints)
    if n_probe == 0:
        return b["viewer_end_tc"]

    base_tc = b["viewer_end_tc"]
    half_step = 0.5 / fps  # half-frame increment

    best_tc    = base_tc
    best_score = float("inf")

    n_candidates = half_frame_steps * 2 * 2 + 1  # +-half_frame_steps frames in 0.5-fr steps
    for i in range(-(half_frame_steps * 2), half_frame_steps * 2 + 1):
        candidate_tc = max(0.0, base_tc + i * half_step)
        viewer_frames = extract_phashes_pipe(
            viewer_path, candidate_tc, fps, n_frames=n_probe)
        if len(viewer_frames) < n_probe:
            continue
        viewer_ints_c = [int(h, 16) for h in viewer_frames]
        total = sum(_hamming(probe_ints[j], viewer_ints_c[j]) for j in range(n_probe))
        score = total / n_probe
        if score < best_score:
            best_score = score
            best_tc    = candidate_tc

    return best_tc


# ---------------------------------------------------------------------------
# Segment transform computation
# ---------------------------------------------------------------------------

def _compute_segment_transforms(path, author_breaks, fps, n_author):
    """
    Build a segment list for the segmented conform from the DTW path and
    detected black-segment breaks.

    Only breaks where frame_delta != 0 require a split: the viewer has more or
    fewer frames in the break region than the author, so we synthesize exactly
    author_frames black frames and skip the viewer's break entirely. Zero-delta
    breaks are transparent to the conform (viewer frames are black and match the
    author count exactly) so they are absorbed into the surrounding content
    segments without creating a split.

    The output is a list of alternating content and break dicts:
        [content, break, content, break, ..., content]

    Content segment keys:
        type                  "content"
        author_start_tc       author timecode at segment start
        author_end_tc         author timecode at segment end
        author_frames         exact frame count (-frames:v in ffmpeg)
        viewer_start_tc       seek point in the viewer file for this segment
        viewer_duration_secs  soft extraction limit (+1s headroom added by conform)

    Break segment keys:
        type                  "break"
        author_start_tc       author timecode at break start
        author_end_tc         author timecode at break end
        author_frames         exact number of black frames to synthesize

    viewer_start_tc for each content segment is derived from the break's
    viewer_end_tc (computed via path deviation in _map_breaks_via_path), so
    the full break region including fades is correctly skipped in the viewer.

    Returns an empty list when no non-zero-delta breaks exist (caller uses
    simple single-pass conform instead).
    """
    if not path:
        return []

    # Only split at breaks where the viewer break length differs from the author.
    adjustments = [b for b in sorted(author_breaks, key=lambda b: b["author_start_tc"])
                   if b["frame_delta"] != 0]

    if not adjustments:
        return []

    segments = []

    # Viewer TC at author t=0: path[0][1] is the viewer frame index at the
    # first path point; dividing by fps gives the initial viewer offset.
    initial_viewer_tc = path[0][1] / fps

    prev_author_end_tc = 0.0
    prev_viewer_end_tc = initial_viewer_tc

    for b in adjustments:
        c_a_start_fr    = int(round(prev_author_end_tc * fps))
        c_a_end_fr      = int(round(b["author_start_tc"] * fps))
        c_author_frames = max(0, c_a_end_fr - c_a_start_fr)

        if c_author_frames > 0:
            # viewer_duration_secs: with speed=1.0 this equals the author
            # content duration. +1s headroom is added by the conform step.
            segments.append({
                "type":               "content",
                "author_start_tc":    round(prev_author_end_tc, 6),
                "author_end_tc":      round(b["author_start_tc"], 6),
                "author_frames":      c_author_frames,
                "viewer_start_tc":    round(prev_viewer_end_tc, 6),
                "viewer_duration_secs": round(c_author_frames / fps, 6),
            })

        segments.append({
            "type":            "break",
            "author_start_tc": round(b["author_start_tc"], 6),
            "author_end_tc":   round(b["author_end_tc"],   6),
            "author_frames":   b["author_duration_frames"],
        })

        prev_author_end_tc = b["author_end_tc"]
        # viewer_end_tc accounts for the full break region including fades
        # (computed via path deviation in _map_breaks_via_path).
        prev_viewer_end_tc = b["viewer_end_tc"]

    # Final content segment after the last adjustment break.
    c_a_start_fr    = int(round(prev_author_end_tc * fps))
    c_author_frames = max(0, n_author - c_a_start_fr)

    if c_author_frames > 0:
        segments.append({
            "type":               "content",
            "author_start_tc":    round(prev_author_end_tc, 6),
            "author_end_tc":      round(n_author / fps, 6),
            "author_frames":      c_author_frames,
            "viewer_start_tc":    round(prev_viewer_end_tc, 6),
            "viewer_duration_secs": round(c_author_frames / fps, 6),
        })

    return segments


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dtw(author_hashes, viewer_path, fps,
            crop=None, band_frames=10000, sub_frame_factor=1,
            stripe_height=2000, max_mem_mb=256, tmp_dir=None):
    """
    Full-frame DTW alignment. No downsampling. Band centered on diagonal.

    author_hashes:    full per-frame pHash list from recipe phash_sequence
    viewer_path:      path to viewer video file (inside Docker)
    fps:              author native fps; viewer is also extracted at this rate
    crop:             optional {w,h,x,y} dict applied before hashing
    band_frames:      Sakoe-Chiba half-band in ORIGINAL frame units.
                      Default 10000 (~417s at 24fps) covers any realistic
                      commercial break length difference.
    sub_frame_factor: extract viewer at fps*N (default 1 = native fps).
                      N=2 gives half-frame precision; band and viewer sequence
                      length are scaled by N.
    stripe_height:    rows per backpointer stripe. Higher values reduce disk
                      seeks at the cost of larger individual stripes.
    max_mem_mb:       RAM budget for in-memory stripes before disk spill.
    tmp_dir:          directory for backpointer spill files (default: system tmp).

    Returns dict with keys:
        diag_offset    -- mean (viewer_tc - author_tc) over path (diagnostic only)
        rms_frames     -- RMS deviation from uniform offset
        max_frames     -- max deviation from uniform offset
        author_breaks  -- black segments in author sequence with viewer_start_tc,
                          viewer_end_tc, and frame_delta (deviation-based)
        segments       -- alternating content/break list for segmented conform
        n_author, n_viewer, path_length
    Returns None if extraction fails.
    """
    n = len(author_hashes)
    if n < 10:
        print("  [DTW] author sequence too short ({} frames), skipping.".format(n),
              flush=True)
        return None

    author_ints = [int(h, 16) for h in author_hashes]

    viewer_fps = fps * sub_frame_factor
    print("  [DTW] extracting viewer at {:.4f}fps...".format(viewer_fps), flush=True)
    viewer_hashes = extract_phashes_pipe(
        viewer_path, 0.0, viewer_fps, crop=crop,
        progress_callback=lambda c: print(
            "  [DTW] ... {} viewer frames".format(c), flush=True),
    )
    m = len(viewer_hashes)
    if m < 10:
        print("  [DTW] viewer extraction failed ({} frames).".format(m), flush=True)
        return None

    viewer_ints = [int(h, 16) for h in viewer_hashes]
    band = band_frames * sub_frame_factor

    # Estimate memory and disk requirements
    n_stripes = (n + stripe_height - 1) // stripe_height
    bytes_per_stripe = stripe_height * (2 * band + 1)
    total_back_bytes = n_stripes * bytes_per_stripe
    max_mem_stripes = max(2, int(max_mem_mb * 1024 * 1024 / bytes_per_stripe))

    print("  [DTW] n_author={} n_viewer={} band={} ({}fr) "
          "stripes={} stripe_size={:.0f}MB backptr_total={:.1f}GB "
          "max_mem_stripes={}".format(
              n, m, band, band_frames,
              n_stripes, bytes_per_stripe / 1e6,
              total_back_bytes / 1e9,
              max_mem_stripes), flush=True)

    store = _StripeStore(stripe_height, max_mem_stripes, tmp_dir=tmp_dir)
    try:
        final_row = _forward_pass(
            author_ints, viewer_ints, fps, viewer_fps, band, store)
        print("  [DTW] forward pass complete, tracing back...", flush=True)
        path = _traceback(
            n, m, final_row, fps, viewer_fps, band, store)
    finally:
        store.cleanup()

    if not path:
        print("  [DTW] empty path, skipping.", flush=True)
        return None

    print("  [DTW] path complete: {} points".format(len(path)), flush=True)

    # Diagnostic: fitted offset across full path (speed fixed at 1.0).
    diag_offset = _lsq_offset(path, fps)

    # RMS and max deviation (viewer_frame - author_frame) in frames.
    sum_sq = 0.0
    max_dev = 0.0
    for i, j in path:
        dev = abs(j - i - diag_offset * fps)
        sum_sq += dev * dev
        if dev > max_dev:
            max_dev = dev
    rms = (sum_sq / len(path)) ** 0.5

    # Detect black segments in author sequence and map to viewer via DTW path.
    author_breaks_raw = detect_black_segments(author_hashes, fps)
    author_breaks = _map_breaks_via_path(author_breaks_raw, path, fps)

    # Sub-frame refinement: for each non-zero-delta break, test half-frame
    # candidate TCs around the DTW prediction to pin the exact seek point
    # where viewer content resumes after the break.
    for b in author_breaks:
        if b["frame_delta"] == 0:
            continue
        old_tc = b["viewer_end_tc"]
        refined_tc = _refine_break_boundary(
            viewer_path, author_ints, b, fps)
        b["viewer_end_tc"] = refined_tc
        shift_fr = (refined_tc - old_tc) * fps
        print("  [DTW]   refine break @{:.2f}s: viewer_end_tc {:.6f}s -> {:.6f}s "
              "({:+.2f}fr)".format(
                  b["author_start_tc"], old_tc, refined_tc, shift_fr), flush=True)

    # Build per-segment transforms for the segmented conform pass.
    segments = _compute_segment_transforms(path, author_breaks, fps, n)

    print("  [DTW] offset={:.6f}s  rms={:.2f}fr  max={:.2f}fr  "
          "breaks={}  segments={}".format(
              diag_offset, rms, max_dev,
              len(author_breaks), len(segments)), flush=True)
    for b in author_breaks:
        print("  [DTW]   break {:.2f}s-{:.2f}s ({}fr) -> "
              "viewer_start={:.3f}s viewer_end={:.3f}s delta={:+d}fr".format(
                  b["author_start_tc"], b["author_end_tc"],
                  b["author_duration_frames"],
                  b["viewer_start_tc"], b["viewer_end_tc"],
                  b["frame_delta"]), flush=True)

    # Viewer TC at author frame 0: this is the true initial offset between the
    # two files, derived directly from the DTW path rather than an anchor search.
    initial_offset_seconds = path[0][1] / fps

    return {
        "initial_offset_seconds": round(initial_offset_seconds, 6),
        "diag_offset":   round(diag_offset, 6),
        "rms_frames":    round(rms, 2),
        "max_frames":    round(max_dev, 2),
        "author_breaks": author_breaks,
        "segments":      segments,
        "n_author":      n,
        "n_viewer":      m,
        "path_length":   len(path),
    }

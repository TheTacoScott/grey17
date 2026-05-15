# grey17

grey17 is a fan-edit recipe toolchain. An editor (the author) describes a
Blender VSE edit as a signed recipe. Viewers supply their own copies of the
source footage and grey17 conforms those copies to match the author's timing,
then renders the edit.

No source footage is distributed. The recipe is a compact YAML file containing
fingerprint data only.

---

## Concepts

**Recipe** - a YAML file that fully describes a fan edit: the Blender VSE
project structure, the source footage required (by filename and fingerprint),
and the per-frame pHash sequences needed to align any copy of that footage to
the author's original. After sign-recipe and fingerprint-cook, the recipe is
self-contained for distribution.

**Conform** - the process of aligning a viewer's copy of a source file to the
author's version. This corrects for differences in timing offsets, commercial
break lengths, and frame counts so that when Blender renders the edit, it draws
from the right frames.

**pHash (perceptual hash)** - each video frame is scaled to 32x32 grayscale
and DCT-hashed to a 64-bit integer. Similarity between two frames is the
Hamming distance between their hashes (0 = identical, 64 = opposite). pHash
is robust to codec differences, resolution differences, and minor color grading
between different releases of the same content.

---

## Pipeline

### Author side

```
generate-recipe -> sign-recipe -> fingerprint-cook -> [distribute recipe]
```

### Viewer side

```
match -> conform -> [verify-conform] -> cook -> [verify-cook]
```

---

## Stage 1: generate-recipe

```
python3 grey17.py generate-recipe <blend_file> <recipe.yaml>
```

Opens the `.blend` file headlessly via Blender's Python API (`bpy`) and walks
the VSE sequence editor. Records every MOVIE and SOUND strip: source filename,
timecodes, channel, and blend mode. Also captures scene settings: fps,
resolution, frame range, and output codec. Writes an unsigned recipe YAML.

No video data is read. This is purely a metadata extraction from the Blender
project.

---

## Stage 2: sign-recipe

```
python3 grey17.py sign-recipe <recipe.yaml> --search-dir /path/to/sources
```

Fingerprints each source file listed in the recipe. Three things are written
per source slot:

**SHA256** - full file hash. During matching, if the viewer's file SHA256
matches exactly, DTW is skipped entirely and conforming is trivial.

**pHash sequence** - ffmpeg pipes raw video frames at native fps through a
`scale=32:32:lanczos,format=gray` filter. Each 32x32 grayscale frame is
hashed with `imagehash.phash()`: the image is DCT-transformed, the top-left
8x8 coefficients are taken, each is compared to the mean, and the result is
encoded as a 64-bit integer. The output is a flat list of 16-char hex strings,
one per frame. For a 43-minute episode at 23.976fps this is approximately 63k
entries.

**Chromaprint audio fingerprints** - ffmpeg extracts the first and last 300
seconds of audio, and `fpcalc -raw` computes Chromaprint integers for each
window. Stored for future use.

If black bars are detected via `ffmpeg cropdetect` (sampled over 80% of the
file, mode crop value must appear in over 50% of samples), the crop rectangle
is stored so the same bars are stripped before hashing during match.

---

## Stage 2.5: fingerprint-cook

```
python3 grey17.py fingerprint-cook <recipe.yaml> <author_cooked_output.mkv>
```

The author renders their own fan edit from their original sources, then runs
this command to fingerprint the output. ffmpeg pipes the rendered file through
the same rawvideo pHash extraction used in sign-recipe, producing one hash per
frame at the render fps for the full edit duration.

The result is written to the recipe as `cooked_phash_sequence` - the ground
truth for what the fan edit should look like. Viewers can later run
verify-cook to compare their own render against this sequence.

Run this after sign-recipe and before distributing the recipe.

---

## Stage 3: match

```
python3 grey17.py match <recipe.yaml> <conform_plan.yaml> <viewer_file.mkv>
```

Aligns the viewer's copy of each source file to the author's version and writes
a conform plan. The conform plan describes exactly how to transform the viewer's
file so that frame N of the conform corresponds to frame N of the author's
original.

**SHA256 fast path** - if the viewer's file matches exactly, conforming is
trivial and DTW is skipped.

**Crop detection** - same `ffmpeg cropdetect` method as sign-recipe, applied
to the viewer file.

**Full-frame DTW** - the core alignment algorithm:

1. Extract the viewer's full pHash sequence at native fps (same rawvideo pipe,
   same 32x32 grayscale).
2. Run Sakoe-Chiba banded DTW on the full n_author x n_viewer grid (~63k x 63k
   for a 43-minute episode). The cost at each cell (i, j) is the Hamming
   distance between `author_hash[i]` and `viewer_hash[j]`. The DP recurrence
   is: `D[i,j] = cost(i,j) + min(D[i-1,j-1], D[i-1,j], D[i,j-1])`.
3. The band is 10000 frames (~417s) centered on the diagonal (speed=1.0,
   offset=0.0 assumed). This is wide enough to cover any realistic commercial
   break timing difference without a pre-search step.
4. Backpointers (one byte per in-band cell) are stored in horizontal stripes.
   Stripes beyond a RAM budget spill to a temp file and are loaded back during
   traceback. Peak RAM is bounded regardless of sequence length.
5. Traceback recovers the minimum-cost path from the best endpoint in the
   final row.
6. The initial viewer offset is read from `path[0][1] / fps` (viewer frame
   index at author frame 0).

**Black segment detection** - the author pHash sequence is scanned for runs of
frames with Hamming distance <= 10 from `0x0000000000000000` (pure black pHash).
Runs shorter than 12 frames are discarded. This gives the locations of
commercial breaks and black slates in the author's source.

**Break mapping via path deviation** - for each detected black interval, the
median path deviation (`viewer_frame - author_frame`) is computed in a 2-second
content window immediately before the break and again immediately after. The
difference is `frame_delta`: how many more or fewer frames the viewer file has
across the full break region, including fade-to-black and fade-from-black
transitions that the pure-black detector misses. `viewer_start_tc` and
`viewer_end_tc` are derived from the pre/post deviations.

**Sub-frame break refinement** - for each non-zero-delta break, 17 candidate
`viewer_end_tc` values are tested at half-frame increments (+-4 frames in 0.5-
frame steps) around the DTW prediction. At each candidate, 60 frames are
extracted from the viewer via rawvideo pipe and compared (mean Hamming distance)
against the corresponding 60 author frames starting at `author_end_frame`. The
candidate with the lowest mean distance is selected.

**Segment computation** - zero-delta breaks are transparent: the viewer and
author agree on frame count, so no split is needed. Only non-zero-delta breaks
become split points. The output is an alternating list of content and break
segment dicts with exact `author_frames` counts and `viewer_start_tc` seek
points.

---

## Stage 4: conform

```
python3 grey17.py conform <conform_plan.yaml> --work-dir ./conformed
```

Applies the timing transforms from the conform plan to the viewer's file. Two
modes are dispatched based on whether any non-zero-delta breaks were found.

**Simple conform** - a single ffmpeg pass: seek to the initial offset, trim to
duration, apply crop and fps filters if needed, encode to libx264 CRF 16 + AAC
320k. Used when the viewer's break lengths match the author's exactly.

**Segmented conform** - a single ffmpeg invocation with a `filter_complex`
graph. No intermediate files.

For each content segment, a separate `-ss viewer_start_tc -t (viewer_duration +
1s headroom) -i input` is added. Inside the filtergraph, each stream goes
through: optional crop, fps normalization, optional scale, `trim=end_frame=N`
(exact frame gate), `setpts=PTS-STARTPTS` (reset timestamps to 0), and
`format=yuv420p`.

For each break segment, a `color=c=black` lavfi source is synthesized and
trimmed to exactly `author_frames` black frames. Audio uses `anullsrc` trimmed
to the break duration.

All video pads are joined by a `concat` filter, as are all audio pads, then
mapped to the output. Because all PTS are reset within each segment before
concat, no drift accumulates across segment boundaries. The output encodes
directly to libx264 CRF 16 + AAC 320k.

---

## Stage 5: verify-conform (optional QA)

```
python3 grey17.py verify-conform <recipe.yaml> <conformed_file.mkv> \
    --slot source_0 --sample-rate 24 --output report.csv
```

Compares the conformed file against the recipe's `phash_sequence` frame by
frame. Extracts pHashes at native fps via rawvideo pipe, subsamples by
`--sample-rate` (24 gives roughly one comparison per second), and computes
Hamming distance for each sampled frame against the corresponding recipe entry.

Reports mean, median, p95, p99, max distance, a distance histogram, and the 20
worst frames. Verdict: PASS (mean < 5.0 and p95 < 10.0), WARN, or FAIL.

What to expect:
- Identical source (SHA256 match): mean near 0
- Close match (e.g. HMAX WEB-DL vs author source): mean ~2.0
- Good conform (e.g. AMZN WEB-DL, 8 non-zero-delta breaks): mean ~3.5
- Scattered high-distance frames at scene cuts are normal and not a conform error

---

## Stage 6: cook

```
python3 grey17.py cook <blend_file> <recipe.yaml> --work-dir ./conformed
```

Renders the fan edit inside Docker using Blender in headless mode.

First, `patch_blend_paths.py` opens the `.blend` via Blender's Python API,
walks all VSE strips (using introspection to handle Blender API changes across
versions), and replaces each source file path with the corresponding conformed
file path. The patched project is saved to a temporary copy.

Then Blender renders the patched project frame by frame. All VSE cuts, timing,
and effects the author built are applied exactly as designed, but drawing from
the conformed fan file instead of the author's original.

---

## Stage 7: verify-cook (optional QA)

```
python3 grey17.py verify-cook <recipe.yaml> <fan_cooked_output.mkv> \
    --sample-rate 24 --output report.csv
```

Compares the fan's rendered output against the `cooked_phash_sequence` written
by fingerprint-cook. This is the definitive quality metric: it measures how
closely the fan's final edit matches the author's own render, after all VSE
cuts have been applied.

Same extraction and comparison method as verify-conform. Verdict thresholds are
tighter: PASS requires mean < 3.0 and p95 < 6.0, because by this stage the
only remaining error should be residual conform imprecision, not codec or
resolution differences.

Isolated high-distance frames at cut points are expected (scene cuts produce
frames that exist in neither source cleanly). The signal to watch is systematic
drift (distances rising over time) or step jumps at break boundaries.

---

## Requirements

**Host machine** - Python 3 (stdlib only), Docker.

**Docker image** - built from `docker/Dockerfile`. Ubuntu 22.04 base with
ffmpeg, fpcalc (Chromaprint), Blender 5.1, and Python packages: ImageHash,
Pillow, PyYAML, jsonschema.

Build the image before running any other command:

```
python3 grey17.py build
```

---

## Quick reference

```
# Author workflow
python3 grey17.py generate-recipe  MyEdit.blend          tmp/myedit.recipe.yaml
python3 grey17.py sign-recipe      tmp/myedit.recipe.yaml --search-dir /path/to/sources
python3 grey17.py cook             MyEdit.blend tmp/myedit.recipe.yaml
python3 grey17.py fingerprint-cook tmp/myedit.recipe.yaml grey17_output/MyEdit.mkv

# Viewer workflow
python3 grey17.py match            tmp/myedit.recipe.yaml tmp/myedit.conform.yaml /path/to/viewer_file.mkv
python3 grey17.py conform          tmp/myedit.conform.yaml --work-dir ./conformed
python3 grey17.py verify-conform   tmp/myedit.recipe.yaml ./conformed/source_0.mkv --sample-rate 24
python3 grey17.py cook             MyEdit.blend tmp/myedit.recipe.yaml --work-dir ./conformed
python3 grey17.py verify-cook      tmp/myedit.recipe.yaml grey17_output/MyEdit.mkv --sample-rate 24
```

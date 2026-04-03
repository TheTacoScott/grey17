#!/usr/bin/env python3
"""
grey17 - fan edit recipe toolchain
Pure stdlib wrapper. All heavy work runs inside Docker.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import datetime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCKER_IMAGE = "grey17/blender:2.91"
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
DOCKER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker")

# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def build_image(tag=DOCKER_IMAGE, no_cache=False):
    cmd = ["docker", "build", "-t", tag]
    if no_cache:
        cmd.append("--no-cache")
    # Copy scripts into build context by building from project root
    # Dockerfile expects scripts/ to be in the build context
    cmd += ["-f", os.path.join(DOCKER_DIR, "Dockerfile"), "."]
    print("Building Docker image {}...".format(tag))
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        die("Docker build failed")


def ensure_image(tag=DOCKER_IMAGE):
    """Check if the image exists locally; build if not."""
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        print("Image {} not found locally. Building...".format(tag))
        build_image(tag)


def run_docker(image, mounts, command, capture_stdout=False):
    """
    Run a Docker container.

    mounts: list of (host_path, container_path, mode) tuples
            mode is "ro" or "rw"
    command: list of strings

    Returns subprocess.CompletedProcess.
    """
    cmd = ["docker", "run", "--rm"]
    for host_path, container_path, mode in mounts:
        cmd += ["-v", "{}:{}:{}".format(os.path.abspath(host_path), container_path, mode)]
    cmd += [image] + command
    if capture_stdout:
        return subprocess.run(cmd, capture_output=True, text=True)
    else:
        return subprocess.run(cmd)

# ---------------------------------------------------------------------------
# Simple YAML writer (stdlib only - no PyYAML dependency on host)
# ---------------------------------------------------------------------------

def _yaml_value(v, indent):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        # Quote strings that look like they need it
        if any(c in v for c in ':#{}[]|>&*!,\n') or v.strip() != v or v == "":
            escaped = v.replace("\\", "\\\\").replace('"', '\\"')
            return '"{}"'.format(escaped)
        return v
    if isinstance(v, list):
        if not v:
            return "[]"
        lines = []
        for item in v:
            if isinstance(item, dict):
                first = True
                for k, dv in item.items():
                    prefix = "{}  ".format(" " * indent)
                    if first:
                        lines.append("{}- {}:{}".format(
                            " " * indent,
                            k,
                            " " + _yaml_value(dv, indent + 4) if not isinstance(dv, (dict, list)) else ""
                        ))
                        if isinstance(dv, (dict, list)):
                            lines.append(_yaml_block(dv, indent + 4))
                        first = False
                    else:
                        lines.append("{}  {}:{}".format(
                            " " * indent,
                            k,
                            " " + _yaml_value(dv, indent + 4) if not isinstance(dv, (dict, list)) else ""
                        ))
                        if isinstance(dv, (dict, list)):
                            lines.append(_yaml_block(dv, indent + 4))
            else:
                lines.append("{}- {}".format(" " * indent, _yaml_value(item, indent + 2)))
        return "\n" + "\n".join(lines)
    if isinstance(v, dict):
        return "\n" + _yaml_block(v, indent + 2)
    return str(v)


def _yaml_block(d, indent):
    lines = []
    for k, v in d.items():
        prefix = " " * indent
        if isinstance(v, (dict, list)) and v:
            lines.append("{}{}:".format(prefix, k))
            lines.append(_yaml_block(v, indent + 2) if isinstance(v, dict) else _yaml_list(v, indent + 2))
        else:
            lines.append("{}{}: {}".format(prefix, k, _yaml_value(v, indent + 2)))
    return "\n".join(lines)


def _yaml_list(lst, indent):
    lines = []
    prefix = " " * indent
    for item in lst:
        if isinstance(item, dict):
            first_key = True
            for k, v in item.items():
                if first_key:
                    if isinstance(v, (dict, list)) and v:
                        lines.append("{}- {}:".format(prefix, k))
                        lines.append(_yaml_block(v, indent + 4) if isinstance(v, dict) else _yaml_list(v, indent + 4))
                    else:
                        lines.append("{}- {}: {}".format(prefix, k, _yaml_value(v, indent + 4)))
                    first_key = False
                else:
                    if isinstance(v, (dict, list)) and v:
                        lines.append("{}  {}:".format(prefix, k))
                        lines.append(_yaml_block(v, indent + 4) if isinstance(v, dict) else _yaml_list(v, indent + 4))
                    else:
                        lines.append("{}  {}: {}".format(prefix, k, _yaml_value(v, indent + 4)))
        else:
            lines.append("{}- {}".format(prefix, _yaml_value(item, indent + 2)))
    return "\n".join(lines)


def dict_to_yaml(d, indent=0):
    return _yaml_block(d, indent)

# ---------------------------------------------------------------------------
# Recipe helpers
# ---------------------------------------------------------------------------

def manifest_to_recipe(manifest, blend_path, title, author, blend_dir_host=None):
    scene = manifest["scene"]
    fps = scene["fps"]
    fps_base = scene["fps_base"]
    effective_fps = fps / fps_base
    total_frames = scene["frame_end"] - scene["frame_start"] + 1
    duration_seconds = total_frames / effective_fps

    # Translate container paths back to real host paths.
    # Blender ran with blend dir mounted at /work/blend/, so strip that prefix.
    def host_path(container_path):
        prefix = "/work/blend/"
        if blend_dir_host and container_path.startswith(prefix):
            rel = container_path[len(prefix):]
            return os.path.join(blend_dir_host, rel)
        return container_path

    fps = scene["fps"] / scene["fps_base"]

    def strip_timecodes_for_source(source_id):
        """Compute strip in/out timecodes for a source from the manifest."""
        seen = {}
        for strip in manifest.get("strips", []):
            if strip.get("source_id") != source_id:
                continue
            if strip["type"] not in ("MOVIE", "SOUND"):
                continue
            in_frame = strip["frame_offset_start"]
            dur_frames = strip["frame_final_duration"]
            out_frame = in_frame + dur_frames
            in_tc = round(in_frame / fps, 6)
            out_tc = round(out_frame / fps, 6)
            for tc, role in [(in_tc, "strip_in"), (out_tc, "strip_out")]:
                if tc not in seen:
                    seen[tc] = {"timecode": tc, "role": role, "strip": strip["name"]}
        return sorted(seen.values(), key=lambda x: x["timecode"])

    sources = []
    for src in manifest["sources"]:
        sources.append({
            "id": src["id"],
            # Author fills these in after generate-recipe
            "name": "",
            "description": "",
            "required": True,
            "original": {
                "filename": src["filename"],
                "filepath_on_author_machine": host_path(src["filepath"]),
                "file_size_bytes": src["file_info"].get("size_bytes"),
                "file_exists_at_sign_time": src["file_info"].get("exists", False),
                # These fields are populated by sign-recipe:
                "resolution_x": None,
                "resolution_y": None,
                "fps": None,
                "duration_seconds": None,
                "duration_frames": None,
                "video_codec": None,
                "audio_codec": None,
                "audio_channels": None,
                "audio_sample_rate": None,
            },
            # anchors populated by sign-recipe
            "anchors": [],
            # strip in/out timecodes - used by sign-recipe to place critical anchors
            "strip_timecodes": strip_timecodes_for_source(src["id"]),
        })

    recipe = {
        "grey17_version": "1",
        "recipe_version": "1.0",
        "signed": False,
        "created_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metadata": {
            "title": title or "",
            "author": author or "",
            "blend_file": os.path.basename(blend_path),
        },
        "blender": {
            "version": manifest["blender_version_string"],
            "version_tuple": manifest["blender_version"],
        },
        "scene": {
            "fps": fps,
            "fps_base": fps_base,
            "effective_fps": round(effective_fps, 6),
            "frame_start": scene["frame_start"],
            "frame_end": scene["frame_end"],
            "duration_seconds": round(duration_seconds, 3),
            "resolution_x": scene["resolution_x"],
            "resolution_y": scene["resolution_y"],
        },
        "output": {
            "format": "mkv",
            "video_codec": "libx265",
            "crf": 18,
            "audio_codec": "aac",
            "audio_bitrate": "320k",
            "resolution_x": scene["resolution_x"],
            "resolution_y": scene["resolution_y"],
            "fps": round(effective_fps, 6),
        },
        "sources": sources,
    }
    return recipe


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return "{:d}:{:02d}:{:02d}".format(h, m, s)


def print_recipe_summary(recipe, output_path):
    scene = recipe["scene"]
    fps = scene["effective_fps"]
    total_frames = scene["frame_end"] - scene["frame_start"] + 1
    duration = format_duration(scene["duration_seconds"])

    print("")
    print("Recipe written to: {}".format(output_path))
    print("")
    print("  Blend:    {}".format(recipe["metadata"]["blend_file"]))
    print("  Blender:  {}".format(recipe["blender"]["version"]))
    print("  Scene:    {}x{}  {}fps  frames {}-{}  ({})".format(
        scene["resolution_x"], scene["resolution_y"],
        scene["effective_fps"],
        scene["frame_start"], scene["frame_end"],
        duration,
    ))
    print("")
    print("Source slots detected: {}".format(len(recipe["sources"])))
    print("")

    for src in recipe["sources"]:
        orig = src["original"]
        exists = orig.get("file_exists_at_sign_time", False)
        size_str = ""
        if orig.get("file_size_bytes"):
            gb = orig["file_size_bytes"] / (1024 ** 3)
            size_str = "  {:.1f} GB".format(gb)
        print("  {}".format(src["id"]))
        print("    File:   {}".format(orig["filename"]))
        print("    Path:   {}".format(orig["filepath_on_author_machine"]))
        print("    On disk: {}{}".format("yes" if exists else "NO - file not found", size_str))
        print("")

    print("Next steps:")
    print("  1. Open {} in a text editor.".format(output_path))
    print('     Fill in "name" and "description" for each source slot.')
    print("     Viewers will read these to know what files to provide.")
    print("")
    print("  2. Run: python3 grey17.py sign-recipe {}".format(output_path))
    print("     (slow - fingerprints all source videos)")
    print("")

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_build(args):
    build_image(no_cache=args.no_cache)
    print("Done.")


def cmd_generate_recipe(args):
    blend_path = os.path.abspath(args.blend_file)
    output_path = os.path.abspath(args.output_recipe)

    if not os.path.exists(blend_path):
        die("Blend file not found: {}".format(blend_path))

    ensure_image()

    blend_dir = os.path.dirname(blend_path)
    blend_filename = os.path.basename(blend_path)

    # We need a writable temp dir inside the container for the manifest output
    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_filename = "manifest.json"
        container_blend = "/work/blend/{}".format(blend_filename)
        container_manifest = "/work/out/{}".format(manifest_filename)
        host_manifest = os.path.join(tmpdir, manifest_filename)

        mounts = [
            (blend_dir, "/work/blend", "ro"),
            (tmpdir, "/work/out", "rw"),
            (SCRIPTS_DIR, "/scripts", "ro"),
        ]

        blender_cmd = [
            "blender",
            "--background", container_blend,
            "--python", "/scripts/extract_vse.py",
            "--",
            "--output", container_manifest,
        ]

        print("Extracting VSE metadata from {}...".format(blend_filename))
        result = run_docker(DOCKER_IMAGE, mounts, blender_cmd)

        if result.returncode != 0:
            die("Blender extraction failed (exit {})".format(result.returncode))

        if not os.path.exists(host_manifest):
            die("Manifest file not written by extraction script")

        with open(host_manifest) as f:
            manifest = json.load(f)

    if not manifest.get("sources"):
        print("WARNING: no source files found in VSE timeline.")

    recipe = manifest_to_recipe(manifest, blend_path, args.title, args.author, blend_dir_host=blend_dir)

    # Write recipe as YAML
    yaml_str = "# grey17 recipe - generated by generate-recipe\n"
    yaml_str += "# Fill in 'name' and 'description' for each source, then run sign-recipe.\n\n"
    yaml_str += dict_to_yaml(recipe)
    yaml_str += "\n"

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(yaml_str)

    print_recipe_summary(recipe, output_path)


def cmd_inspect(args):
    blend_path = os.path.abspath(args.blend_file)
    if not os.path.exists(blend_path):
        die("Blend file not found: {}".format(blend_path))

    ensure_image()

    blend_dir = os.path.dirname(blend_path)
    blend_filename = os.path.basename(blend_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest_filename = "manifest.json"
        container_blend = "/work/blend/{}".format(blend_filename)
        container_manifest = "/work/out/{}".format(manifest_filename)
        host_manifest = os.path.join(tmpdir, manifest_filename)

        mounts = [
            (blend_dir, "/work/blend", "ro"),
            (tmpdir, "/work/out", "rw"),
            (SCRIPTS_DIR, "/scripts", "ro"),
        ]

        blender_cmd = [
            "blender",
            "--background", container_blend,
            "--python", "/scripts/extract_vse.py",
            "--",
            "--output", container_manifest,
        ]

        result = run_docker(DOCKER_IMAGE, mounts, blender_cmd)
        if result.returncode != 0:
            die("Blender extraction failed (exit {})".format(result.returncode))

        if not os.path.exists(host_manifest):
            die("Manifest file not written")

        with open(host_manifest) as f:
            data = json.load(f)

    output = args.output or None
    if output:
        with open(output, "w") as f:
            json.dump(data, f, indent=2)
        print("Manifest written to: {}".format(output))
    else:
        print(json.dumps(data, indent=2))

# ---------------------------------------------------------------------------
# Recipe reader (stdlib only - no PyYAML on host)
# ---------------------------------------------------------------------------

def parse_recipe_minimal(recipe_path):
    """
    Extract just what the wrapper needs from a recipe.yaml without a full
    YAML parser. Only reliable for YAML written by grey17 generate-recipe.

    Returns dict with keys:
      signed: bool
      sources: list of {id, filename, filepath_on_author_machine}
    """
    result = {"signed": False, "sources": []}
    current_source = None
    in_sources = False
    in_original = False
    indent_sources = None

    with open(recipe_path) as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if not stripped:
                continue

            # Top-level signed flag
            if line.startswith("signed:"):
                val = line.split(":", 1)[1].strip().lower()
                result["signed"] = val == "true"
                continue

            # Detect sources: block start
            if line == "sources:":
                in_sources = True
                indent_sources = 0
                continue

            # Detect end of sources block (top-level key at same or less indent)
            if in_sources and indent <= (indent_sources or 0) and stripped and not stripped.startswith("-") and ":" in stripped:
                if indent == 0 and not line.startswith(" "):
                    in_sources = False
                    in_original = False
                    current_source = None
                    continue

            if not in_sources:
                continue

            # New source entry: "  - id: source_N"
            if stripped.startswith("- id:"):
                if current_source:
                    result["sources"].append(current_source)
                slot_id = stripped.split(":", 1)[1].strip()
                current_source = {"id": slot_id, "filename": None, "filepath_on_author_machine": None}
                in_original = False
                continue

            if current_source is None:
                continue

            # Enter original: block
            if stripped == "original:":
                in_original = True
                continue

            if in_original:
                if stripped.startswith("filename:"):
                    val = stripped.split(":", 1)[1].strip().strip('"')
                    current_source["filename"] = val
                elif stripped.startswith("filepath_on_author_machine:"):
                    val = stripped.split(":", 1)[1].strip().strip('"')
                    current_source["filepath_on_author_machine"] = val
                # Exit original block when we hit a key at lower indent level
                elif indent <= 4 and stripped and ":" in stripped and not stripped.startswith("#"):
                    in_original = False

    if current_source:
        result["sources"].append(current_source)

    return result


def resolve_source_paths(sources, explicit_sources, search_dirs):
    """
    For each source slot, resolve the actual file path to use.

    explicit_sources: dict of {slot_id: path} from --source args
    search_dirs: list of directory paths from --search-dir args

    Returns dict of {slot_id: resolved_path} for slots that were found.
    Missing slots are omitted.
    """
    resolved = {}
    for src in sources:
        slot_id = src["id"]
        filename = src.get("filename")

        # 1. Explicit --source mapping
        if slot_id in explicit_sources:
            path = explicit_sources[slot_id]
            if os.path.exists(path):
                resolved[slot_id] = path
            else:
                print("WARNING: explicit path not found for {}: {}".format(slot_id, path))
            continue

        # 2. Search directories by filename
        if filename and search_dirs:
            for d in search_dirs:
                candidate = os.path.join(d, filename)
                if os.path.exists(candidate):
                    resolved[slot_id] = candidate
                    break

        if slot_id in resolved:
            continue

        # 3. Original path from recipe
        orig_path = src.get("filepath_on_author_machine")
        if orig_path and os.path.exists(orig_path):
            resolved[slot_id] = orig_path

    return resolved


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_sign_recipe(args):
    recipe_path = os.path.abspath(args.recipe)
    if not os.path.exists(recipe_path):
        die("Recipe file not found: {}".format(recipe_path))

    recipe_meta = parse_recipe_minimal(recipe_path)

    if recipe_meta["signed"] and not args.force:
        die("Recipe is already signed. Use --force to re-sign.")

    if not recipe_meta["sources"]:
        die("No sources found in recipe.")

    # Resolve source file paths
    explicit = {}
    for s in args.source:
        if "=" not in s:
            die("Invalid --source format (expected slot_id=/path): {}".format(s))
        slot_id, path = s.split("=", 1)
        explicit[slot_id.strip()] = os.path.abspath(path.strip())

    search_dirs = [os.path.abspath(d) for d in (args.search_dir or [])]

    resolved = resolve_source_paths(recipe_meta["sources"], explicit, search_dirs)

    # Report resolution
    print("\nSource file resolution:")
    all_found = True
    for src in recipe_meta["sources"]:
        slot_id = src["id"]
        path = resolved.get(slot_id)
        if path:
            print("  {} -> {}".format(slot_id, path))
        else:
            print("  {} -> NOT FOUND (filename: {})".format(slot_id, src.get("filename")))
            all_found = False

    if not all_found:
        print("")
        print("Some sources could not be resolved. Provide paths via:")
        print("  --source slot_id=/path/to/file")
        print("  --search-dir /directory/containing/files")
        die("Cannot sign recipe with missing sources.")

    ensure_image()

    recipe_dir = os.path.dirname(recipe_path)
    recipe_filename = os.path.basename(recipe_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build mounts: recipe dir (rw) + each source file (ro) + tmp work dir (rw)
        mounts = [
            (recipe_dir, "/work/recipe", "rw"),
            (tmpdir, "/work/tmp", "rw"),
            (SCRIPTS_DIR, "/scripts", "ro"),
        ]

        sign_cmd = [
            "python3", "/scripts/sign_recipe.py",
            "--recipe", "/work/recipe/{}".format(recipe_filename),
            "--work-dir", "/work/tmp",
            "--anchor-interval", str(args.anchor_interval),
        ]

        # Mount each source and add --source arg
        for src in recipe_meta["sources"]:
            slot_id = src["id"]
            path = resolved[slot_id]
            filename = src.get("filename") or os.path.basename(path)
            container_path = "/work/sources/{}/{}".format(slot_id, filename)
            mounts.append((path, container_path, "ro"))
            sign_cmd += ["--source", "{}={}".format(slot_id, container_path)]

        print("\nSigning recipe: {}".format(recipe_filename))
        result = run_docker(DOCKER_IMAGE, mounts, sign_cmd)

        if result.returncode != 0:
            die("sign-recipe failed (exit {})".format(result.returncode))

    print("\nDone. Recipe is signed and ready to share.")


def cmd_match(args):
    recipe_path = os.path.abspath(args.recipe)
    output_path = os.path.abspath(args.output_conform_plan)

    if not os.path.exists(recipe_path):
        die("Recipe file not found: {}".format(recipe_path))

    if not args.files and not args.slot:
        die("Provide at least one candidate file or --slot mapping.")

    recipe_meta = parse_recipe_minimal(recipe_path)

    if not recipe_meta.get("signed"):
        die("Recipe has not been signed yet. Run sign-recipe first.")

    # Build slot -> file mapping
    # 1. Explicit --slot args take priority
    explicit = {}
    for s in (args.slot or []):
        if "=" not in s:
            die("Invalid --slot format (expected slot_id=/path): {}".format(s))
        slot_id, path = s.split("=", 1)
        explicit[slot_id.strip()] = os.path.abspath(path.strip())

    # 2. Auto-assign positional files to slots by order
    candidate_files = [os.path.abspath(f) for f in (args.files or [])]
    slots = recipe_meta.get("sources", [])

    slot_file_map = {}
    auto_idx = 0
    for src in slots:
        sid = src["id"]
        if sid in explicit:
            slot_file_map[sid] = explicit[sid]
        elif auto_idx < len(candidate_files):
            slot_file_map[sid] = candidate_files[auto_idx]
            auto_idx += 1

    if not slot_file_map:
        die("Could not assign any files to source slots.")

    # Validate files exist
    for sid, path in slot_file_map.items():
        if not os.path.exists(path):
            die("File not found for {}: {}".format(sid, path))

    ensure_image()

    recipe_dir = os.path.dirname(recipe_path)
    recipe_filename = os.path.basename(recipe_path)
    output_dir = os.path.dirname(output_path)
    output_filename = os.path.basename(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Use a work dir under tmp/ - avoids permission issues with Docker-created root files
    project_root = os.path.dirname(os.path.abspath(__file__))
    work_dir = os.path.join(project_root, "tmp", "_match_work")
    os.makedirs(work_dir, exist_ok=True)

    mounts = [
        (recipe_dir, "/work/recipe", "ro"),
        (work_dir, "/work/tmp", "rw"),
        (output_dir, "/work/out", "rw"),
        (SCRIPTS_DIR, "/scripts", "ro"),
    ]

    match_cmd = [
        "python3", "/scripts/match_recipe.py",
        "--recipe", "/work/recipe/{}".format(recipe_filename),
        "--output", "/work/out/{}".format(output_filename),
        "--work-dir", "/work/tmp",
        "--threshold", str(args.threshold),
    ]

    for src in slots:
        sid = src["id"]
        if sid not in slot_file_map:
            continue
        host_file = slot_file_map[sid]
        filename = os.path.basename(host_file)
        container_path = "/work/candidates/{}/{}".format(sid, filename)
        mounts.append((host_file, container_path, "ro"))
        match_cmd += ["--slot", "{}={}".format(sid, container_path)]

    print("Running match...")
    print("  Recipe: {}".format(recipe_filename))
    for sid, path in slot_file_map.items():
        print("  {}: {}".format(sid, os.path.basename(path)))
    print()

    result = run_docker(DOCKER_IMAGE, mounts, match_cmd)

    if result.returncode != 0:
        die("match failed (exit {})".format(result.returncode))

    # The container wrote container-internal paths as input_file values.
    # Replace them with the real host paths so the conform plan is portable.
    # The file may be root-owned (Docker writes as root), so write to a sibling
    # temp file and rename over it - rename only needs directory write permission.
    if os.path.exists(output_path):
        with open(output_path) as f:
            content = f.read()
        for src in slots:
            sid = src["id"]
            if sid not in slot_file_map:
                continue
            host_file = slot_file_map[sid]
            filename = os.path.basename(host_file)
            container_path = "/work/candidates/{}/{}".format(sid, filename)
            content = content.replace(container_path, host_file)
        tmp_path = output_path + ".tmp"
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, output_path)

    print("Conform plan written to: {}".format(output_path))


def parse_conform_plan_minimal(plan_path):
    """
    Extract what cmd_conform needs from a conform plan without a YAML parser.
    Returns dict with:
      all_suitable: bool
      sources: list of {slot_id, status, input_file, output_filename}
    """
    result = {"all_suitable": False, "sources": []}
    current_source = None
    in_sources = False

    with open(plan_path) as f:
        for raw_line in f:
            line = raw_line.rstrip()
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if not stripped:
                continue

            if line.startswith("all_suitable:"):
                val = line.split(":", 1)[1].strip().lower()
                result["all_suitable"] = val == "true"
                continue

            if line == "sources:":
                in_sources = True
                continue

            if in_sources and indent == 0 and ":" in stripped and not stripped.startswith("-"):
                in_sources = False
                current_source = None
                continue

            if not in_sources:
                continue

            if stripped.startswith("- slot_id:"):
                if current_source:
                    result["sources"].append(current_source)
                sid = stripped.split(":", 1)[1].strip().strip('"')
                current_source = {
                    "slot_id": sid,
                    "status": None,
                    "input_file": None,
                    "output_filename": None,
                }
                continue

            if current_source is None:
                continue

            if stripped.startswith("status:"):
                current_source["status"] = stripped.split(":", 1)[1].strip().strip('"')
            elif stripped.startswith("input_file:"):
                current_source["input_file"] = stripped.split(":", 1)[1].strip().strip('"')
            elif stripped.startswith("output_filename:"):
                current_source["output_filename"] = stripped.split(":", 1)[1].strip().strip('"')

    if current_source:
        result["sources"].append(current_source)

    return result


def cmd_conform(args):
    plan_path = os.path.abspath(args.conform_plan)
    if not os.path.exists(plan_path):
        die("Conform plan not found: {}".format(plan_path))

    plan_meta = parse_conform_plan_minimal(plan_path)

    if not plan_meta["sources"]:
        die("No sources found in conform plan.")

    suitable = [s for s in plan_meta["sources"] if s.get("status") == "suitable"]
    if not suitable:
        die("No suitable sources in conform plan. Re-run match.")

    if not plan_meta["all_suitable"]:
        print("WARNING: not all sources are suitable. Only suitable sources will be conformed.")

    # Validate input files exist on the host
    for source in suitable:
        input_file = source.get("input_file")
        if not input_file:
            die("No input_file recorded for slot {}. Re-run match.".format(source["slot_id"]))
        if not os.path.exists(input_file):
            die("Input file not found for {}: {}".format(source["slot_id"], input_file))

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    ensure_image()

    plan_dir = os.path.dirname(plan_path)
    plan_filename = os.path.basename(plan_path)

    mounts = [
        (plan_dir, "/work/plan", "ro"),
        (work_dir, "/work/output", "rw"),
        (SCRIPTS_DIR, "/scripts", "ro"),
    ]

    conform_cmd = [
        "python3", "/scripts/conform_sources.py",
        "--plan", "/work/plan/{}".format(plan_filename),
        "--work-dir", "/work/output",
    ]

    # Mount each input file and pass container path via --slot
    for source in suitable:
        slot_id = source["slot_id"]
        host_file = source["input_file"]
        filename = os.path.basename(host_file)
        container_path = "/work/sources/{}/{}".format(slot_id, filename)
        mounts.append((host_file, container_path, "ro"))
        conform_cmd += ["--slot", "{}={}".format(slot_id, container_path)]

    print("Conforming sources from: {}".format(plan_filename))
    for source in suitable:
        print("  {} -> {}".format(source["slot_id"], source.get("output_filename", "?")))
    print()

    result = run_docker(DOCKER_IMAGE, mounts, conform_cmd)

    if result.returncode != 0:
        die("conform failed (exit {})".format(result.returncode))

    print("\nConformed files written to: {}".format(work_dir))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def die(msg):
    print("ERROR: {}".format(msg), file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="grey17",
        description="Fan edit recipe toolchain",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # build
    p_build = sub.add_parser("build", help="Build the Docker image")
    p_build.add_argument("--no-cache", action="store_true")
    p_build.set_defaults(func=cmd_build)

    # generate-recipe
    p_gen = sub.add_parser(
        "generate-recipe",
        help="Extract VSE metadata from a .blend file and write an initial recipe",
    )
    p_gen.add_argument("blend_file", metavar="<blend_file>")
    p_gen.add_argument("output_recipe", metavar="<output_recipe.yaml>")
    p_gen.add_argument("--title", default="", metavar="<title>")
    p_gen.add_argument("--author", default="", metavar="<handle>")
    p_gen.set_defaults(func=cmd_generate_recipe)

    # sign-recipe
    p_sign = sub.add_parser(
        "sign-recipe",
        help="Fingerprint source videos and write anchor points into a recipe",
    )
    p_sign.add_argument("recipe", metavar="<recipe.yaml>")
    p_sign.add_argument("--source", action="append", default=[],
                        metavar="slot_id=/path/to/file",
                        help="Explicitly map a source slot to a file (repeatable)")
    p_sign.add_argument("--search-dir", action="append", default=[],
                        metavar="<dir>",
                        help="Directory to search for source files by filename (repeatable)")
    p_sign.add_argument("--anchor-interval", type=float, default=1.0,
                        metavar="<seconds>")
    p_sign.add_argument("--force", action="store_true",
                        help="Re-sign even if recipe is already signed")
    p_sign.set_defaults(func=cmd_sign_recipe)

    # match
    p_match = sub.add_parser(
        "match",
        help="Match viewer files against a signed recipe and produce a conform plan",
    )
    p_match.add_argument("recipe", metavar="<recipe.yaml>")
    p_match.add_argument("output_conform_plan", metavar="<output.conform.yaml>")
    p_match.add_argument("files", nargs="*", metavar="<file>",
                         help="Candidate video files (auto-assigned to slots in order)")
    p_match.add_argument("--slot", action="append", default=[],
                         metavar="slot_id=/path/to/file",
                         help="Explicitly assign a file to a slot (repeatable)")
    p_match.add_argument("--threshold", type=float, default=0.85,
                         metavar="<0.0-1.0>",
                         help="Minimum anchor match rate for a file to be suitable (default: 0.85)")
    p_match.set_defaults(func=cmd_match)

    # conform
    p_conform = sub.add_parser(
        "conform",
        help="Apply timing transforms from a conform plan to produce conformed source files",
    )
    p_conform.add_argument("conform_plan", metavar="<conform_plan.yaml>")
    p_conform.add_argument("--work-dir", default="./grey17_work",
                           metavar="<dir>",
                           help="Directory to write conformed files into (default: ./grey17_work/)")
    p_conform.set_defaults(func=cmd_conform)

    # inspect (debug)
    p_inspect = sub.add_parser(
        "inspect",
        help="Dump raw VSE manifest JSON from a .blend file (debug)",
    )
    p_inspect.add_argument("blend_file", metavar="<blend_file>")
    p_inspect.add_argument("--output", metavar="<manifest.json>", default=None)
    p_inspect.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

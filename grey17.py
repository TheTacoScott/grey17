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

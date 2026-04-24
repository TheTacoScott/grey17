"""
Runs inside Blender's Python environment (blender --background --python).
Extracts VSE metadata from the active scene and writes JSON to a file.

Usage (from blender invocation):
    blender --background file.blend --python extract_vse.py -- --output /tmp/manifest.json
"""
import bpy
import json
import sys
import os


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    output = None
    for i, arg in enumerate(argv):
        if arg == "--output" and i + 1 < len(argv):
            output = argv[i + 1]
    return output


def abs_path(path):
    if not path:
        return path
    return os.path.normpath(bpy.path.abspath(path))


def strip_source_path(strip):
    """Return the resolved absolute filepath for a strip, or None."""
    if strip.type == "MOVIE":
        return abs_path(strip.filepath)
    if strip.type == "SOUND":
        if strip.sound:
            return abs_path(strip.sound.filepath)
    if strip.type == "IMAGE":
        if strip.directory:
            return abs_path(strip.directory)
    return None


def probe_file(path):
    """Return basic file info without needing ffprobe."""
    if not path or not os.path.exists(path):
        return {"exists": False}
    size = os.path.getsize(path)
    return {"exists": True, "size_bytes": size}


def collect_strips(sequences, sources_by_path, strips_out, meta_parent=None):
    for strip in sequences:
        entry = {
            "name": strip.name,
            "type": strip.type,
            "channel": strip.channel,
            "frame_start": strip.frame_start,
            "frame_final_start": strip.frame_final_start,
            "frame_final_end": strip.frame_final_end,
            "frame_final_duration": strip.frame_final_duration,
            "frame_offset_start": strip.frame_offset_start,
            "frame_offset_end": strip.frame_offset_end,
            "mute": strip.mute,
        }
        if meta_parent:
            entry["meta_parent"] = meta_parent

        # blend mode / alpha (only on effect/movie strips)
        if hasattr(strip, "blend_type"):
            entry["blend_type"] = strip.blend_type
        if hasattr(strip, "blend_alpha"):
            entry["blend_alpha"] = strip.blend_alpha

        # volume / pan for sound strips (pitch was removed in Blender 5.x)
        if strip.type == "SOUND":
            entry["volume"] = strip.volume
            entry["pan"] = strip.pan

        # source file
        filepath = strip_source_path(strip)
        if filepath:
            entry["filepath"] = filepath
            if filepath not in sources_by_path:
                idx = len(sources_by_path)
                sources_by_path[filepath] = {
                    "id": "source_{}".format(idx),
                    "filepath": filepath,
                    "filename": os.path.basename(filepath),
                    "file_info": probe_file(filepath),
                    "strip_names": [],
                    # placeholder fields the author will fill in
                    "name": "",
                    "description": "",
                }
            sources_by_path[filepath]["strip_names"].append(strip.name)
            entry["source_id"] = sources_by_path[filepath]["id"]

        strips_out.append(entry)

        # recurse into META strips
        if strip.type == "META":
            collect_strips(strip.strips, sources_by_path, strips_out, meta_parent=strip.name)


def main():
    output_path = parse_args()

    scene = bpy.context.scene
    render = scene.render

    manifest = {
        "blender_version": list(bpy.app.version),
        "blender_version_string": bpy.app.version_string.strip(),
        "scene": {
            "fps": render.fps,
            "fps_base": render.fps_base,
            "frame_start": scene.frame_start,
            "frame_end": scene.frame_end,
            "resolution_x": render.resolution_x,
            "resolution_y": render.resolution_y,
        },
        "sources": [],
        "strips": [],
    }

    seq_editor = scene.sequence_editor
    if seq_editor is None:
        print("WARNING: no sequence editor found in this scene", file=sys.stderr)
    else:
        sources_by_path = {}
        strips_out = []
        collect_strips(seq_editor.strips_all, sources_by_path, strips_out)
        manifest["sources"] = list(sources_by_path.values())
        manifest["strips"] = strips_out

    output = json.dumps(manifest, indent=2)

    if output_path:
        with open(output_path, "w") as f:
            f.write(output)
        print("Manifest written to: {}".format(output_path), file=sys.stderr)
    else:
        print(output)


main()

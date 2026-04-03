"""
Runs inside Blender's Python environment (blender --background --python).
Patches VSE strip source paths in a copy of the .blend file, then saves it.

Usage:
    blender --background src.blend --python patch_blend_paths.py -- \
        --output /work/scratch/patched.blend \
        --map /original/path/film.mkv=/work/conformed/film.mkv \
        --map /original/path/extras.mkv=/work/conformed/extras.mkv
"""
import bpy
import os
import sys


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    path_map = {}
    output_path = None

    i = 0
    while i < len(argv):
        if argv[i] == "--map" and i + 1 < len(argv):
            parts = argv[i + 1].split("=", 1)
            if len(parts) == 2:
                path_map[parts[0]] = parts[1]
            i += 2
        elif argv[i] == "--output" and i + 1 < len(argv):
            output_path = argv[i + 1]
            i += 2
        else:
            i += 1

    return path_map, output_path


def abs_path(path):
    if not path:
        return path
    return os.path.normpath(bpy.path.abspath(path))


def patch_sequences(sequences, path_map, patched, skipped):
    for strip in sequences:
        if strip.type == "MOVIE":
            orig = abs_path(strip.filepath)
            if orig in path_map:
                strip.filepath = path_map[orig]
                patched.append((orig, path_map[orig]))
            else:
                skipped.append(("MOVIE", orig))
        elif strip.type == "SOUND" and strip.sound:
            orig = abs_path(strip.sound.filepath)
            if orig in path_map:
                strip.sound.filepath = path_map[orig]
                patched.append((orig, path_map[orig]))
            else:
                skipped.append(("SOUND", orig))
        elif strip.type == "META":
            patch_sequences(strip.sequences, path_map, patched, skipped)


def main():
    path_map, output_path = parse_args()

    if not path_map:
        print("WARNING: no --map arguments provided, nothing to patch")

    seq_editor = bpy.context.scene.sequence_editor
    if not seq_editor:
        print("ERROR: no sequence editor found in scene")
        sys.exit(1)

    patched = []
    skipped = []
    patch_sequences(seq_editor.sequences_all, path_map, patched, skipped)

    print("Patched {} strips:".format(len(patched)))
    for orig, new in patched:
        print("  {} -> {}".format(orig, new))

    if skipped:
        print("Unmatched strips ({} total, showing up to 10):".format(len(skipped)))
        for kind, orig in skipped[:10]:
            print("  [{}] {}".format(kind, orig))

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=output_path)
        print("Saved patched .blend to: {}".format(output_path))
    else:
        bpy.ops.wm.save_mainfile()
        print("Saved patched .blend in-place")


main()

"""
Runs inside Blender's Python environment (blender --background --python).
Applies output settings from the recipe and renders the VSE timeline.

Usage:
    blender --background patched.blend --python render_vse.py -- \
        --output /work/output/result.mkv \
        --frame-start 0 \
        --frame-end 60355 \
        --format mkv \
        --video-codec libx265 \
        --crf 18 \
        --audio-codec aac \
        --audio-bitrate 320k \
        --resolution 1440x1080 \
        --fps 23.976023
"""
import bpy
import sys


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    opts = {
        "output": None,
        "frame_start": None,
        "frame_end": None,
        "format": None,
        "video_codec": None,
        "crf": None,
        "audio_codec": None,
        "audio_bitrate": None,
        "resolution": None,
        "fps": None,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--output", "--format", "--video-codec", "--audio-codec",
                   "--audio-bitrate", "--resolution", "--fps",
                   "--frame-start", "--frame-end", "--crf") and i + 1 < len(argv):
            key = arg.lstrip("-").replace("-", "_")
            opts[key] = argv[i + 1]
            i += 2
        else:
            i += 1

    return opts


# Map recipe/CLI codec names to Blender enum values
CONTAINER_MAP = {
    "mkv":  "MKV",
    "mp4":  "MPEG4",
    "mov":  "QUICKTIME",
    "avi":  "AVI",
}

VIDEO_CODEC_MAP = {
    "libx264": "H264",  "h264":  "H264",
    "libx265": "H265",  "h265":  "H265",  "hevc": "H265",
    "mpeg4":   "MPEG4",
    "theora":  "THEORA",
    "vp9":     "WEBM",
}

AUDIO_CODEC_MAP = {
    "aac": "AAC",
    "mp3": "MP3",
    "ac3": "AC3",
    "flac": "FLAC",
    "pcm": "PCM",
    "none": "NONE",
}

# Map numeric CRF to Blender's constant_rate_factor enum
def crf_to_blender(crf_int):
    if crf_int <= 15:
        return "HIGH"
    if crf_int <= 22:
        return "MEDIUM"
    if crf_int <= 28:
        return "LOW"
    return "VERYLOW"


def main():
    opts = parse_args()
    scene = bpy.context.scene
    render = scene.render

    # Use sequencer output
    render.use_sequencer = True

    # Output format: ffmpeg
    render.image_settings.file_format = "FFMPEG"

    fmt = (opts["format"] or "mkv").lower()
    render.ffmpeg.format = CONTAINER_MAP.get(fmt, "MKV")

    codec = (opts["video_codec"] or "libx264").lower()
    blender_codec = VIDEO_CODEC_MAP.get(codec, "H264")
    try:
        render.ffmpeg.codec = blender_codec
    except TypeError:
        # Codec not available in this Blender version - fall back to H264
        print("WARNING: codec {} not available, falling back to H264".format(blender_codec))
        render.ffmpeg.codec = "H264"

    if opts["crf"] is not None:
        try:
            render.ffmpeg.constant_rate_factor = crf_to_blender(int(opts["crf"]))
        except (ValueError, TypeError):
            render.ffmpeg.constant_rate_factor = "MEDIUM"
    else:
        render.ffmpeg.constant_rate_factor = "MEDIUM"

    acodec = (opts["audio_codec"] or "aac").lower()
    render.ffmpeg.audio_codec = AUDIO_CODEC_MAP.get(acodec, "AAC")

    if opts["audio_bitrate"]:
        bitrate_str = str(opts["audio_bitrate"]).lower().rstrip("k")
        try:
            render.ffmpeg.audio_bitrate = int(bitrate_str)
        except ValueError:
            render.ffmpeg.audio_bitrate = 320
    else:
        render.ffmpeg.audio_bitrate = 320

    # Resolution
    if opts["resolution"]:
        try:
            w, h = opts["resolution"].split("x", 1)
            render.resolution_x = int(w)
            render.resolution_y = int(h)
        except (ValueError, AttributeError):
            pass
    render.resolution_percentage = 100

    # FPS
    if opts["fps"]:
        try:
            fps_val = float(opts["fps"])
            # Express as integer fps + fps_base for fractional rates
            # e.g. 23.976023 -> fps=24000, fps_base=1001
            # Common cases:
            common = {
                23.976: (24000, 1001),
                24.0:   (24, 1),
                25.0:   (25, 1),
                29.97:  (30000, 1001),
                30.0:   (30, 1),
                50.0:   (50, 1),
                59.94:  (60000, 1001),
                60.0:   (60, 1),
            }
            matched = False
            for known_fps, (num, den) in common.items():
                if abs(fps_val - known_fps) < 0.01:
                    render.fps = num
                    render.fps_base = den
                    matched = True
                    break
            if not matched:
                # Generic: use rational approximation via fps + fps_base
                render.fps = int(round(fps_val * 1000))
                render.fps_base = 1000.0
        except (ValueError, TypeError):
            pass

    # Frame range
    if opts["frame_start"] is not None:
        try:
            scene.frame_start = int(opts["frame_start"])
        except (ValueError, TypeError):
            pass
    if opts["frame_end"] is not None:
        try:
            scene.frame_end = int(opts["frame_end"])
        except (ValueError, TypeError):
            pass

    # Output path
    if opts["output"]:
        render.filepath = opts["output"]

    print("Rendering VSE timeline:")
    print("  Frames:     {}-{}".format(scene.frame_start, scene.frame_end))
    print("  Resolution: {}x{}".format(render.resolution_x, render.resolution_y))
    print("  Container:  {}".format(render.ffmpeg.format))
    print("  Video:      {}  quality={}".format(
        render.ffmpeg.codec, render.ffmpeg.constant_rate_factor))
    print("  Audio:      {}  {}kbps".format(
        render.ffmpeg.audio_codec, render.ffmpeg.audio_bitrate))
    print("  Output:     {}".format(render.filepath))

    bpy.ops.render.render(animation=True)
    print("Render complete.")


main()

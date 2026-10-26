"""Generate disposable video fixtures for the real Civitai metadata parser."""
import base64
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from saver.video import build_video_metadata, embed_video_metadata, find_ffmpeg, native_video_metadata


BASE = dict(
    modelname="fallback.safetensors", positive="ABCD positive", negative="negative",
    width=832, height=480, seed=42, steps=20, cfg=5.5,
    sampler_name="dpmpp_2m", scheduler_name="karras", denoise=0.75,
)
ORIGINAL = {
    "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "original.safetensors"}},
    "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "style.safetensors", "strength_model": 0.6}},
    "3": {"class_type": "CustomVideoSampler", "inputs": {"seed": 7}},
}
WORKFLOW = {"nodes": [{"id": 3, "type": "CustomVideoSampler"}], "extra": {"note": "preserve me"}}
CASES = {
    "air": [{"modelVersionId": 101, "air": "urn:air:sdxl:checkpoint:civitai:11@101"},
            {"modelVersionId": 102, "air": "urn:air:sdxl:lora:civitai:12@102", "weight": 0.8}],
    "id_only": [{"modelVersionId": 101, "type": "checkpoint"}, {"modelVersionId": 102, "type": "lora", "weight": 0.8}],
    "missing_type": [{"modelVersionId": 103}],
    "embedding": [{"modelVersionId": 104, "type": "textualinversion", "weight": 0.5}],
    "auxiliary": [{"modelVersionId": 105, "type": "other"}],
    "invalid_air": [{"modelVersionId": 106, "type": "checkpoint", "air": "broken"}],
    "offline": None,
    "no_workflow": [{"modelVersionId": 101, "type": "checkpoint"}],
}


def main():
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("FFprobe is required for compatibility tests")
    fixtures = []
    with tempfile.TemporaryDirectory(prefix="video_saver_civitai_") as temporary:
        directory = Path(temporary)
        for extension, video_codec, audio_codec in [("mp4", "libx264", "aac"), ("webm", "libvpx-vp9", "libopus")]:
            source = directory / f"source.{extension}"
            subprocess.run([
                find_ffmpeg(), "-loglevel", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.1",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=0.1",
                "-c:v", video_codec, "-c:a", audio_codec, "-shortest", str(source),
            ], check=True, capture_output=True)
            for name, resources in CASES.items():
                parameters = 'ABCD positive\nNegative prompt: negative\nSteps: 20, Hashes: {"model":"ABCDEF1234","LORA:style":"123456ABCD"}'
                if resources is not None:
                    parameters += ", Civitai resources: " + json.dumps(resources)
                metadata = SimpleNamespace(**BASE, a111_params=parameters)
                tags = build_video_metadata(metadata, None if name == "no_workflow" else ORIGINAL,
                                            None if name == "no_workflow" else {"workflow": WORKFLOW})
                # Native VIDEO JSON-serializes values before our final remux.
                native = native_video_metadata(tags)
                assert native.get("prompt") == json.loads(tags["prompt"])
                output = embed_video_metadata(source, tags, suffix=f"_{name}")
                probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags:stream=codec_type,codec_name", "-of", "json", output],
                                       check=True, capture_output=True, text=True, encoding="utf-8")
                fixtures.append({"case": name, "container": extension, "probe": json.loads(probe.stdout),
                                 "bytes": base64.b64encode(Path(output).read_bytes()).decode("ascii")})
    print(json.dumps(fixtures))


if __name__ == "__main__":
    main()

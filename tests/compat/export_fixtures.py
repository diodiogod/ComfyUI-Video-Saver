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
from saver.metadata_reader import read_video_tags, metadata_fields, read_resource_records
from saver.resource_refs import export_resource_references, parse_reference, resource_from_identifier


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
    "unresolved_weighted": None,
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
                if name == "unresolved_weighted":
                    parameters += ', Resource references: ["LORA:style:123456ABCD:0.65"]'
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
                # Rebuild the metadata through the reader, as in an independent
                # upscaling workflow whose prompt/resources are unrelated.
                recovered, width, height = read_video_tags(output)
                restored = SimpleNamespace(**metadata_fields(recovered, width, height))
                processed_tags = build_video_metadata(restored,
                    {"999": {"class_type": "UpscaleVideo", "inputs": {"seed": 999}}},
                    {"workflow": {"nodes": [{"id": 999, "type": "UpscaleVideo"}], "extra": {"note": "processing workflow"}}})
                processed_output = embed_video_metadata(source, processed_tags, suffix=f"_{name}_restored")
                processed_probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags:stream=codec_type,codec_name", "-of", "json", processed_output],
                                                 check=True, capture_output=True, text=True, encoding="utf-8")
                fixtures.append({"case": name, "stage": "restored", "container": extension, "probe": json.loads(processed_probe.stdout),
                                 "bytes": base64.b64encode(Path(processed_output).read_bytes()).decode("ascii")})
                if name == "air":
                    references = export_resource_references(read_resource_records(recovered), restored.final_hashes)
                    references += ",urn:air:sdxl:lora:civitai:17@107:0.6"
                    carried = []
                    for entry in references.split(","):
                        _, weight, identifier = parse_reference(entry)
                        resource = resource_from_identifier(identifier)
                        if resource is not None:
                            if weight is not None:
                                resource["weight"] = weight
                            carried.append(resource)
                    upscale = SimpleNamespace(**(BASE | dict(positive="New upscale prompt", negative="new negative", seed=99,
                        steps=12, cfg=4.0, sampler_name="euler", scheduler_name="normal", width=1024, height=768)),
                        a111_params='New upscale prompt\nNegative prompt: new negative\nSteps: 12, Hashes: {"model":"ABCDEF1234","LORA:style":"123456ABCD"}, Civitai resources: ' + json.dumps(carried))
                    upscale_tags = build_video_metadata(upscale, {"999": {"class_type": "UpscaleVideo"}}, {"workflow": {"nodes": [], "extra": {"note": "upscale"}}})
                    upscale_output = embed_video_metadata(source, upscale_tags, suffix="_upscale")
                    upscale_probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "format_tags:stream=codec_type,codec_name", "-of", "json", upscale_output],
                                                   check=True, capture_output=True, text=True, encoding="utf-8")
                    fixtures.append({"case": name, "stage": "upscale", "container": extension, "probe": json.loads(upscale_probe.stdout),
                                     "bytes": base64.b64encode(Path(upscale_output).read_bytes()).decode("ascii")})
    print(json.dumps(fixtures))


if __name__ == "__main__":
    main()

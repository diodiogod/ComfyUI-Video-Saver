import json
import shutil
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import pytest

from .video import (
    build_video_metadata,
    embed_video_metadata,
    escape_ffmetadata,
    native_video_metadata,
    resources_to_prompt_json,
    select_vhs_video_file,
    unpack_vhs_filenames,
)


def video_metadata(resources=None, **overrides):
    values = dict(
        modelname="fallback.safetensors", positive="ABCD positive", negative="negative",
        width=832, height=480, seed=42, steps=20, cfg=5.5,
        sampler_name="dpmpp_2m", scheduler_name="karras", denoise=0.75,
        a111_params='ABCD positive\nNegative prompt: negative\nSteps: 20, Hashes: {"model":"ABCDEF1234"}',
    )
    if resources is not None:
        values["a111_params"] += ", Civitai resources: " + json.dumps(resources)
    values.update(overrides)
    return SimpleNamespace(**values)


def sampler(graph):
    return next(node["inputs"] for node in graph.values() if node["class_type"] == "KSampler")


@pytest.mark.parametrize("resource", [
    {"modelVersionId": 123, "type": "checkpoint"},
    {"modelVersionId": "123", "type": "textualinversion"},
    {"modelVersionId": 123},
    {"air": "urn:air:sdxl:checkpoint:civitai:12@123"},
    {"modelVersionId": 123, "type": "lora", "air": "invalid"},
])
def test_compatibility_graph_accepts_optional_air_and_type(resource):
    graph = json.loads(resources_to_prompt_json(video_metadata(), [resource]))
    assert any("@123" in str(node["inputs"]) for node in graph.values())
    assert sampler(graph)["seed"] == 42


def test_compatibility_graph_preserves_settings_and_source_graph():
    original = {"99": {"class_type": "CustomVideoSampler", "inputs": {"prompt": "old"}}}
    workflow = {"nodes": [{"id": 99, "type": "CustomVideoSampler"}]}
    metadata = video_metadata([{"modelVersionId": 123, "type": "checkpoint"}])
    tags = build_video_metadata(metadata, original, {"workflow": workflow})
    graph = json.loads(tags["prompt"])
    settings = sampler(graph)
    assert graph[settings["positive"][0]]["inputs"]["text"] == "ABCD positive"
    assert graph[settings["negative"][0]]["inputs"]["text"] == "negative"
    assert graph[settings["latent_image"][0]]["inputs"] == {"width": 832, "height": 480}
    assert settings["scheduler"] == "karras"
    assert settings["denoise"] == 0.75
    assert json.loads(tags["prompt_original"]) == original
    assert json.loads(tags["workflow"]) == workflow
    assert tags["parameters"] == metadata.a111_params


def test_offline_graph_retains_original_resource_names_without_old_sampler():
    original = {
        "7": {"class_type": "UNETLoader", "inputs": {"unet_name": "original.safetensors"}},
        "8": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["7", 0], "lora_name": "style.safetensors", "strength_model": 0.6}},
        "9": {"class_type": "KSampler", "inputs": {"seed": 1}},
    }
    tags = build_video_metadata(video_metadata(), original, None)
    graph = json.loads(tags["prompt"])
    assert any(node["inputs"].get("unet_name") == "original.safetensors" for node in graph.values())
    assert any(node["inputs"].get("lora_name") == "style.safetensors" for node in graph.values())
    assert len([node for node in graph.values() if node["class_type"] == "KSampler"]) == 1
    assert sampler(graph)["seed"] == 42
    assert json.loads(tags["prompt_original"]) == original


def test_metadata_without_workflow_keeps_model_filename_and_valid_empty_workflow():
    tags = build_video_metadata(video_metadata(), None, None)
    assert any(node["inputs"].get("unet_name") == "fallback.safetensors" for node in json.loads(tags["prompt"]).values())
    assert tags["workflow"] == "{}"
    assert "prompt_original" not in tags


@pytest.mark.parametrize("resources", [None, {}, [None, "invalid", {"modelVersionId": False}, {"modelVersionId": 1.5}]])
def test_missing_or_malformed_resources_do_not_break_video_metadata(resources):
    graph = json.loads(resources_to_prompt_json(video_metadata(), resources))
    assert sampler(graph)["seed"] == 42


def test_graph_builder_does_not_mutate_inputs_or_resource_weights():
    resources = [{"modelVersionId": 123, "type": "lora", "weight": 0.0}]
    before = deepcopy(resources)
    graph = json.loads(resources_to_prompt_json(video_metadata(), resources))
    assert resources == before
    assert any(node["inputs"].get("strength_model") == 0.0 for node in graph.values())


def test_auxiliary_resource_is_not_used_in_place_of_the_model():
    graph = json.loads(resources_to_prompt_json(video_metadata(), [{"modelVersionId": 55, "type": "other"}]))
    model = graph[sampler(graph)["model"][0]]
    assert model["class_type"] == "UNETLoader"
    assert model["inputs"]["unet_name"] == "fallback.safetensors"


def test_unresolved_lora_hash_name_survives_without_original_workflow():
    metadata = video_metadata(a111_params='positive\nSteps: 20, Hashes: {"LORA:unresolved":"ABCDEF1234"}')
    graph = json.loads(resources_to_prompt_json(metadata, [{"modelVersionId": 123, "type": "lora"}]))
    assert any(node["inputs"].get("lora_name") == "unresolved" for node in graph.values())


def test_hash_names_do_not_duplicate_original_lora_filenames():
    metadata = video_metadata(a111_params='positive\nSteps: 20, Hashes: {"LORA:style":"ABCDEF1234"}')
    original = {"1": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "style.safetensors"}}}
    graph = json.loads(resources_to_prompt_json(metadata, None, original))
    assert len([node for node in graph.values() if "lora_name" in node["inputs"]]) == 1


def test_no_explicit_metadata_preserves_original_prompt():
    original = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "original"}}}
    tags = build_video_metadata(None, original, {"workflow": {"nodes": []}})
    assert json.loads(tags["prompt"]) == original
    assert "prompt_original" not in tags


def test_build_video_metadata_preserves_civitai_contract():
    metadata = SimpleNamespace(
        a111_params=(
            "positive\nNegative prompt: negative\n"
            "Steps: 12, Hashes: {\"model\":\"ABCDEF1234\"}, "
            "Civitai resources: [{\"modelVersionId\":123,\"type\":\"checkpoint\"}]"
        )
    )
    tags = build_video_metadata(
        metadata,
        {"1": {"class_type": "KSampler"}},
        {"workflow": {"nodes": []}},
    )

    assert tags["parameters"] == metadata.a111_params
    assert json.loads(tags["prompt"])["1"]["class_type"] == "KSampler"
    assert json.loads(tags["workflow"]) == {"nodes": []}
    assert json.loads(tags["civitaiResources"])[0]["modelVersionId"] == 123
    assert json.loads(tags["extraMetadata"])["civitaiResources"][0]["type"] == "checkpoint"


def test_escape_ffmetadata():
    assert escape_ffmetadata(r"a=b;c#d\e\nf") == r"a\=b\;c\#d\\e\\nf"


def test_unpack_vhs_filenames():
    assert unpack_vhs_filenames((False, ["one.mp4", "two.webm"])) == (False, ["one.mp4", "two.webm"])
    assert unpack_vhs_filenames(["one.mp4"]) == (True, ["one.mp4"])


def test_select_vhs_video_file_ignores_metadata_png():
    files = ["AnimateDiff_00001.png", "AnimateDiff_00001.mp4"]
    assert select_vhs_video_file(files) == "AnimateDiff_00001.mp4"


def test_select_vhs_video_file_prefers_audio_muxed_output():
    files = [
        "AnimateDiff_00001.png",
        "AnimateDiff_00001.mp4",
        "AnimateDiff_00001-audio.mp4",
    ]
    assert select_vhs_video_file(files) == "AnimateDiff_00001-audio.mp4"


def test_select_vhs_video_file_rejects_image_formats_clearly():
    with pytest.raises(ValueError, match="no MP4 or WebM"):
        select_vhs_video_file(["AnimateDiff_00001.png", "AnimateDiff_00001.gif"])


def test_native_video_metadata_avoids_double_serializing_parameters():
    tags = {
        "parameters": "positive\nNegative prompt: negative",
        "prompt": '{"1":{"class_type":"KSampler"}}',
        "civitaiResources": '[{"modelVersionId":123}]',
    }

    native = native_video_metadata(tags)

    assert "parameters" not in native
    assert native["prompt"] == {"1": {"class_type": "KSampler"}}
    assert native["civitaiResources"] == [{"modelVersionId": 123}]


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="FFmpeg and FFprobe are required")
def test_embed_video_metadata_mp4(tmp_path):
    source = tmp_path / "source.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=16x16:d=0.1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source),
        ],
        check=True,
    )

    output = embed_video_metadata(
        source,
        {
            "parameters": "positive\nNegative prompt: negative",
            "civitaiResources": '[{"modelVersionId":123}]',
        },
    )
    probe = subprocess.run(
        [
            shutil.which("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format_tags=parameters,civitaiResources",
            "-of",
            "json",
            output,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tags = json.loads(probe.stdout)["format"]["tags"]
    assert tags["parameters"] == "positive\nNegative prompt: negative"
    assert json.loads(tags["civitaiResources"])[0]["modelVersionId"] == 123

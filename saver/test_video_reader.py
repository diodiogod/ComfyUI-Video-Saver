import io
import json
import shutil
import subprocess
from copy import deepcopy
from types import SimpleNamespace

import av
import pytest

from .video import build_video_metadata, embed_video_metadata, MAX_METADATA_VALUE_BYTES
from .metadata_reader import read_video_tags, metadata_fields
from .resource_refs import parse_reference, resource_from_identifier


PARAMETERS = ('ABCD landscape, "morning"\nsecond line\nNegative prompt: blurry\nnoisy\n'
              'Steps: 24, Sampler: DPM++ 2M Karras, Schedule type: karras, CFG scale: 5.5, '
              'Seed: 18446744073709551614, Size: 832x480, Clip skip: 2, Model: original, '
              'Hashes: {"model":"ABCDEF1234","LORA:style":"123456ABCD"}, '
              'Civitai resources: [{"modelVersionId":123,"type":"checkpoint"},{"modelVersionId":456,"type":"lora","weight":0.6}]')
WORKFLOW = {"nodes": [{"id": 99, "type": "OriginalVideoGeneration"}], "extra": {"note": "source workflow"}}
PROCESSING = {"nodes": [{"id": 1, "type": "UpscaleVideo"}]}


def source_tags():
    metadata = SimpleNamespace(
        modelname="original", positive='ABCD landscape, "morning"\nsecond line', negative="blurry\nnoisy",
        width=832, height=480, seed=18446744073709551614, steps=24, cfg=5.5,
        sampler_name="dpmpp_2m", scheduler_name="karras", denoise=0.75, a111_params=PARAMETERS,
    )
    return build_video_metadata(metadata, {"99": {"class_type": "OriginalVideoGeneration", "inputs": {}}}, {"workflow": WORKFLOW})


def test_reader_recovers_exact_parameters_without_rehashing_models():
    tags = source_tags()
    fields = metadata_fields(tags, 32, 32)
    assert fields["positive"] == 'ABCD landscape, "morning"\nsecond line'
    assert fields["negative"] == "blurry\nnoisy"
    assert fields["seed"] == 18446744073709551614
    assert fields["steps"] == 24
    assert fields["cfg"] == 5.5
    assert (fields["width"], fields["height"]) == (832, 480)
    assert fields["sampler_name"] == "dpmpp_2m"
    assert fields["scheduler_name"] == "karras"
    assert fields["denoise"] == 0.75
    assert fields["clip_skip"] == 2
    assert fields["a111_params"] == PARAMETERS
    references = [parse_reference(entry) for entry in fields["final_hashes"].split(",")]
    assert {identifier for _, _, identifier in references if not resource_from_identifier(identifier)} == {"ABCDEF1234", "123456ABCD"}
    assert {resource_from_identifier(identifier)["modelVersionId"]: weight
            for _, weight, identifier in references if resource_from_identifier(identifier)} == {123: None, 456: 0.6}
    assert fields["ckpt_path"] == ""
    assert fields["source_metadata_tags"] == tags


def test_graph_only_metadata_remains_graph_only_and_preserves_resource_ids():
    tags = source_tags()
    del tags["parameters"]
    fields = metadata_fields(tags, 16, 16)
    assert fields["positive"].startswith("ABCD")
    assert fields["negative"] == "blurry\nnoisy"
    assert fields["steps"] == 24
    assert fields["seed"] == 18446744073709551614
    restored = build_video_metadata(SimpleNamespace(**fields), {"1": {"class_type": "UpscaleVideo"}}, {"workflow": PROCESSING})
    assert restored["prompt"] == tags["prompt"]
    assert restored["civitaiResources"] == tags["civitaiResources"]
    assert "parameters" not in restored


def test_resaving_preserves_generation_and_both_workflows_without_mutation():
    tags = source_tags()
    original = deepcopy(tags)
    metadata = SimpleNamespace(**metadata_fields(tags, 16, 16))
    processing_prompt = {"1": {"class_type": "UpscaleVideo", "inputs": {"seed": 999}}}
    restored = build_video_metadata(metadata, processing_prompt, {"workflow": PROCESSING})
    for key in ("parameters", "prompt", "prompt_original", "civitaiResources", "extraMetadata"):
        assert restored[key] == original[key]
    assert json.loads(restored["workflow"]) == PROCESSING
    assert json.loads(restored["workflow_original"]) == WORKFLOW
    assert json.loads(restored["prompt_postprocess"]) == processing_prompt
    assert tags == original
    assert metadata.source_metadata_tags == original


def test_repeated_processing_keeps_first_generation_workflow_without_nesting_history():
    metadata = SimpleNamespace(**metadata_fields(source_tags(), 16, 16))
    first = build_video_metadata(metadata, {"1": {"class_type": "UpscaleVideo"}}, {"workflow": PROCESSING})
    second = build_video_metadata(SimpleNamespace(**metadata_fields(first, 32, 32)), {}, {"workflow": {"nodes": ["interpolate"]}})
    assert json.loads(second["workflow_original"]) == WORKFLOW
    assert len(second) == len(first)
    assert second["prompt"] == first["prompt"]


def test_workflow_saving_off_retains_generation_but_removes_workflow_history():
    metadata = SimpleNamespace(**metadata_fields(source_tags(), 16, 16))
    restored = build_video_metadata(metadata, None, None)
    assert restored["parameters"] == PARAMETERS
    assert restored["prompt"] == metadata.source_metadata_tags["prompt"]
    assert restored["workflow"] == "{}"
    assert "prompt_original" not in restored
    assert "workflow_original" not in restored
    assert "prompt_postprocess" not in restored


def test_parameter_only_video_can_get_a_native_compatibility_graph():
    fields = metadata_fields({"parameters": PARAMETERS}, 16, 16)
    # The node converts display sampler names with the existing sampler map.
    fields["sampler_name"] = "dpmpp_2m"
    restored = build_video_metadata(SimpleNamespace(**fields), {}, {"workflow": PROCESSING})
    graph = json.loads(restored["prompt"])
    assert restored["parameters"] == PARAMETERS
    assert any("@123" in str(node["inputs"]) for node in graph.values())
    assert any("@456" in str(node["inputs"]) for node in graph.values())


@pytest.mark.parametrize("extension,video_codec,audio_codec", [("mp4", "libx264", "aac"), ("webm", "libvpx-vp9", "libopus")])
def test_real_video_upscale_metadata_roundtrip(tmp_path, extension, video_codec, audio_codec):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("FFmpeg is required for video round trips")
    source = tmp_path / f"source.{extension}"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=16x16:d=0.2:r=10",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", "-c:v", video_codec, "-c:a", audio_codec,
                    "-shortest", str(source)], check=True, capture_output=True)
    tagged = embed_video_metadata(source, source_tags())
    tags, width, height = read_video_tags(tagged)
    assert (width, height) == (16, 16)
    assert tags["parameters"] == PARAMETERS
    metadata = SimpleNamespace(**metadata_fields(tags, width, height))

    upscaled = tmp_path / f"upscaled.{extension}"
    subprocess.run([ffmpeg, "-v", "error", "-y", "-i", tagged, "-vf", "scale=32:32", "-c:v", video_codec,
                    "-c:a", "copy", "-map_metadata", "-1", str(upscaled)], check=True, capture_output=True)
    restored_tags = build_video_metadata(metadata, {"1": {"class_type": "UpscaleVideo"}}, {"workflow": PROCESSING})
    restored = embed_video_metadata(upscaled, restored_tags)
    readback, new_width, new_height = read_video_tags(restored)
    assert (new_width, new_height) == (32, 32)
    assert readback == restored_tags
    assert readback["parameters"] == PARAMETERS
    assert readback["prompt"] == tags["prompt"]
    with av.open(restored) as container:
        assert len(container.streams.audio) == 1
        assert container.streams.audio[0].codec_context.name == ("aac" if extension == "mp4" else "opus")

    with open(restored, "rb") as file:
        buffer = io.BytesIO(file.read())
    buffer.seek(7)
    assert read_video_tags(buffer)[0] == restored_tags
    assert buffer.tell() == 7
    assert not buffer.closed


def test_native_json_quoted_parameters_and_uppercase_webm_tags(tmp_path):
    file = tmp_path / "native.webm"
    with av.open(str(file), "w") as container:
        container.metadata["PARAMETERS"] = json.dumps(PARAMETERS)
        stream = container.add_stream("libvpx-vp9", rate=1)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame(16, 16, "yuv420p")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    assert read_video_tags(str(file))[0]["parameters"] == PARAMETERS


def test_no_generation_metadata_is_reported_clearly(tmp_path):
    file = tmp_path / "plain.webm"
    with av.open(str(file), "w") as container:
        stream = container.add_stream("libvpx-vp9", rate=1)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for packet in stream.encode(av.VideoFrame(16, 16, "yuv420p")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with pytest.raises(ValueError, match="No generation metadata"):
        read_video_tags(str(file))


def test_restored_metadata_keeps_existing_size_limit():
    with pytest.raises(ValueError, match="2 MiB"):
        build_video_metadata(SimpleNamespace(source_metadata_tags={"parameters": "X" * (MAX_METADATA_VALUE_BYTES + 1)}), {}, {})


def test_zero_values_and_legacy_model_hash_are_recovered():
    fields = metadata_fields({"parameters": 'ABCD\nSteps: 1, CFG scale: 0, Seed: 0, Model hash: ABCDEF1234'}, 32, 16)
    assert fields["seed"] == 0
    assert fields["cfg"] == 0
    assert fields["positive"] == "ABCD"
    assert fields["final_hashes"] == "model:ABCDEF1234"

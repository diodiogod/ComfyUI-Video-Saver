import importlib
import json
import os
import sys
from types import SimpleNamespace, ModuleType

import av
import numpy as np
import pytest
from PIL import Image, features
from PIL.PngImagePlugin import PngInfo

import comfy
import folder_paths
from image_saver_under_test.nodes import ImageSaver, Metadata
from image_saver_under_test.nodes import ImageSaverMetadata
from image_saver_under_test import utils_civitai
from image_saver_under_test.saver.metadata_reader import MissingGenerationMetadata, read_image_tags
from image_saver_under_test.saver.saver import save_image


PARAMETERS = ('ABCD original\nNegative prompt: blurry\nSteps: 24, Sampler: DPM++ 2M Karras, '
              'CFG scale: 5.5, Seed: 123456, Size: 832x480, Model: original, Hashes: {"model":"ABCDEF1234"}')
IMAGE_FORMATS = ["png", "jpeg", "webp", pytest.param("avif", marks=pytest.mark.skipif(
    not features.check("avif"), reason="Pillow built without AVIF support"))]


@pytest.fixture
def reader_module(monkeypatch):
    # The real V3 schema and registry are checked separately in the ComfyUI
    # runtime smoke test. These tests isolate execution from server/GPU imports.
    api = ModuleType("comfy_api.latest")
    api.io = SimpleNamespace(ComfyNode=type("ComfyNode", (), {}),
                             NodeOutput=lambda *values, **kwargs: SimpleNamespace(result=values, ui=kwargs.get("ui")))
    monkeypatch.setitem(sys.modules, "comfy_api", ModuleType("comfy_api"))
    monkeypatch.setitem(sys.modules, "comfy_api.latest", api)
    graph_utils = ModuleType("comfy_execution.graph_utils")
    graph_utils.ExecutionBlocker = type("ExecutionBlocker", (), {"__init__": lambda self, message: setattr(self, "message", message)})
    monkeypatch.setitem(sys.modules, "comfy_execution", ModuleType("comfy_execution"))
    monkeypatch.setitem(sys.modules, "comfy_execution.graph_utils", graph_utils)
    samplers = ModuleType("comfy.samplers")
    samplers.KSampler = SimpleNamespace(SAMPLERS=["euler", "dpmpp_2m"], SCHEDULERS=["normal", "karras"])
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers)
    monkeypatch.setattr(comfy, "samplers", samplers, raising=False)
    module_name = "image_saver_under_test.nodes_metadata_reader"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    sys.modules.pop(module_name, None)


@pytest.fixture
def source_video(tmp_path):
    file = tmp_path / "source.mp4"
    with av.open(str(file), "w", options={"movflags": "use_metadata_tags"}) as container:
        container.metadata["parameters"] = PARAMETERS
        stream = container.add_stream("libx264", rate=1)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for packet in stream.encode(av.VideoFrame(16, 16, "yuv420p")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return file


def test_native_video_input_yields_real_saver_metadata_and_individual_fields(reader_module, source_video):
    class LoadedVideo:
        def get_stream_source(self):
            return str(source_video)

        def get_components(self):
            pytest.fail("Reading metadata must not decode image/audio components")

    result = reader_module.ImageSaverReadVideoMetadata.execute(video=LoadedVideo())
    assert len(result.result) == 1, "Media readers must expose only METADATA; use the expander for individual fields"
    metadata = result.result[0]
    positive, negative, hashes, params = metadata.positive, metadata.negative, metadata.final_hashes, metadata.a111_params
    assert isinstance(metadata, Metadata)
    assert positive == "ABCD original"
    assert negative == "blurry"
    assert hashes == "model:ABCDEF1234"
    assert params == PARAMETERS
    assert metadata.sampler_name == "dpmpp_2m"
    assert metadata.scheduler_name == "karras"
    expanded = reader_module.ReadImageSaverMetadata.execute(metadata).result
    assert expanded == ("original", positive, negative, 832, 480, 123456, 24, 5.5,
                        "dpmpp_2m", "karras", 1.0, 0, hashes, params)


def test_file_picker_uses_comfyui_path_resolver(reader_module, source_video, monkeypatch):
    requested = []

    def resolve(file):
        requested.append(file)
        return str(source_video)

    monkeypatch.setattr(folder_paths, "get_annotated_filepath", resolve, raising=False)
    result = reader_module.ImageSaverReadVideoMetadata.execute(file="source.mp4").result
    assert requested == ["source.mp4"]
    assert result[0].source_metadata_tags["parameters"] == PARAMETERS


def test_reader_cache_invalidates_when_source_is_replaced(reader_module, source_video, monkeypatch):
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(source_video), raising=False)
    before = reader_module.ImageSaverReadVideoMetadata.fingerprint_inputs(file="source.mp4")
    stat = source_video.stat()
    os.utime(source_video, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    after = reader_module.ImageSaverReadVideoMetadata.fingerprint_inputs(file="source.mp4")
    assert before != after


def test_reader_requires_a_source_and_validates_file_format(reader_module, tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Select a source"):
        reader_module.ImageSaverReadVideoMetadata.execute()
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(tmp_path / "source.png"), raising=False)
    with pytest.raises(ValueError, match="MP4 or WebM"):
        reader_module.ImageSaverReadVideoMetadata.execute(file="source.png")


@pytest.mark.parametrize("reader_name,read_tags", [
    ("ImageSaverReadVideoMetadata", "read_video_tags"),
    ("ImageSaverReadImageMetadata", "read_image_tags"),
])
def test_missing_metadata_blocks_downstream_and_reports_warning_without_raising(reader_module, tmp_path, monkeypatch, reader_name, read_tags):
    message = "No generation metadata was found. Use a source saved with metadata enabled."

    def missing(_):
        raise MissingGenerationMetadata(message)

    monkeypatch.setattr(reader_module, read_tags, missing)
    file = "plain.mp4" if "Video" in reader_name else "plain.png"
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(tmp_path / file), raising=False)
    output = getattr(reader_module, reader_name).execute(file=file)
    assert len(output.result) == 1
    assert type(output.result[0]).__name__ == "ExecutionBlocker"
    assert output.result[0].message is None, "Missing metadata must not trigger ComfyUI's execution_error path"
    assert output.ui == {"metadata_warning": [message]}


@pytest.mark.parametrize("extension", IMAGE_FORMATS)
def test_image_reader_returns_only_metadata_and_uses_shared_expander(reader_module, tmp_path, monkeypatch, extension):
    file = tmp_path / f"source.{extension}"
    save_image(Image.new("RGB", (16, 16)), str(file), extension, 95, True, False,
               PARAMETERS, {"1": {"class_type": "CustomGeneration"}}, {"workflow": {"nodes": []}}, True)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(file), raising=False)
    output = reader_module.ImageSaverReadImageMetadata.execute(file=file.name)
    assert len(output.result) == 1
    metadata = output.result[0]
    assert isinstance(metadata, Metadata)
    assert metadata.positive == "ABCD original"
    assert metadata.final_hashes == "model:ABCDEF1234"
    assert reader_module.ReadImageSaverMetadata.execute(metadata).result[5:10] == (123456, 24, 5.5, "dpmpp_2m", "karras")


@pytest.mark.parametrize("extension", IMAGE_FORMATS)
def test_plain_image_is_an_expected_warning(reader_module, tmp_path, monkeypatch, extension):
    file = tmp_path / f"plain.{extension}"
    Image.new("RGB", (8, 8)).save(file)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(file), raising=False)
    output = reader_module.ImageSaverReadImageMetadata.execute(file=file.name)
    assert output.result[0].message is None
    assert "No generation metadata" in output.ui["metadata_warning"][0]


def test_corrupt_files_are_not_misreported_as_missing_metadata(reader_module, tmp_path, monkeypatch):
    file = tmp_path / "corrupt.png"
    file.write_text("not an image")
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(file), raising=False)
    with pytest.raises(Image.UnidentifiedImageError):
        reader_module.ImageSaverReadImageMetadata.execute(file=file.name)


def test_native_png_graph_without_parameters_can_be_expanded(reader_module, tmp_path, monkeypatch):
    file = tmp_path / "native.png"
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "original graph prompt"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry"}},
        "3": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 480}},
        "4": {"class_type": "KSampler", "inputs": {"positive": ["1", 0], "negative": ["2", 0],
              "latent_image": ["3", 0], "seed": 42, "steps": 12, "cfg": 4, "sampler_name": "euler", "scheduler": "normal"}},
    }
    info = PngInfo()
    info.add_text("prompt", json.dumps(graph))
    Image.new("RGB", (16, 16)).save(file, pnginfo=info)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(file), raising=False)
    metadata = reader_module.ImageSaverReadImageMetadata.execute(file=file.name).result[0]
    assert metadata.positive == "original graph prompt"
    assert metadata.negative == "blurry"
    assert metadata.seed == 42
    assert (metadata.width, metadata.height) == (832, 480)
    assert metadata.a111_params == ""
    assert metadata.source_metadata_tags["prompt"] == json.dumps(graph)


@pytest.mark.parametrize("extension", IMAGE_FORMATS)
def test_image_read_save_roundtrip_preserves_source_prompt_and_parameters(reader_module, tmp_path, monkeypatch, extension):
    source = tmp_path / f"source.{extension}"
    original_prompt = {"1": {"class_type": "OriginalGeneration", "inputs": {"seed": 123456}}}
    original_workflow = {"nodes": [{"id": 1, "type": "OriginalGeneration"}]}
    processing_workflow = {"nodes": [{"id": 99, "type": "UpscaleImage"}]}
    save_image(Image.new("RGB", (16, 16)), str(source), extension, 95, True, False,
               PARAMETERS, original_prompt, {"workflow": original_workflow}, True)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(source), raising=False)
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    metadata = reader_module.ImageSaverReadImageMetadata.execute(file=source.name).result[0]
    image = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((32, 32, 3))))
    names = ImageSaver.save_images([image], "processed", extension, "", 95, True, False,
                                  {"99": {"class_type": "UpscaleImage"}}, {"workflow": processing_workflow},
                                  False, True, 0, "%Y", metadata)
    tags, width, height = read_image_tags(str(tmp_path / names[0]))
    assert (width, height) == (32, 32)
    assert tags["parameters"] == PARAMETERS
    assert json.loads(tags["prompt"]) == original_prompt
    assert json.loads(tags["workflow"]) == processing_workflow
    assert json.loads(tags["workflow_original"]) == original_workflow


@pytest.mark.parametrize("reader_name", ["ImageSaverReadVideoMetadata", "ImageSaverReadImageMetadata"])
@pytest.mark.parametrize("intermediate_supported", [False, True], ids=["older-v3", "current-v3"])
def test_reader_schema_supports_older_and_current_v3(reader_module, tmp_path, monkeypatch, reader_name, intermediate_supported):
    def older_schema(*, node_id, display_name, category, description, search_aliases, inputs, outputs):
        return SimpleNamespace(node_id=node_id, outputs=outputs, has_intermediate_output=False)

    def current_schema(*, node_id, display_name, category, description, search_aliases, inputs, outputs, has_intermediate_output=False):
        return SimpleNamespace(node_id=node_id, outputs=outputs, has_intermediate_output=has_intermediate_output)

    monkeypatch.setattr(reader_module.io, "Schema", current_schema if intermediate_supported else older_schema, raising=False)
    monkeypatch.setattr(reader_module.io, "Combo", SimpleNamespace(Input=lambda *args, **kwargs: None), raising=False)
    monkeypatch.setattr(reader_module.io, "Video", SimpleNamespace(Input=lambda *args, **kwargs: None), raising=False)
    monkeypatch.setattr(reader_module.io, "UploadType", SimpleNamespace(video="video", image="image"), raising=False)
    monkeypatch.setattr(reader_module.io, "Custom", lambda _: SimpleNamespace(Output=lambda *args, **kwargs: None), raising=False)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    schema = getattr(reader_module, reader_name).define_schema()
    assert len(schema.outputs) == 1
    assert schema.has_intermediate_output is intermediate_supported


@pytest.mark.parametrize("storage", ["parameters", "container_tag", "extraMetadata"])
def test_expander_hashes_carries_id_only_resources_and_weights(reader_module, storage):
    resources = [
        {"modelVersionId": 123, "type": "checkpoint", "modelName": "Original model"},
        {"modelVersionId": 456, "type": "lora", "modelName": "Style LoRA", "weight": 0.65,
         "air": "urn:air:sdxl:lora:civitai:45@456"},
    ]
    tags = {"parameters": "original prompt\nSteps: 24, Seed: 42"}
    if storage == "parameters":
        tags["parameters"] += ", Civitai resources: " + json.dumps(resources)
    elif storage == "container_tag":
        tags["civitaiResources"] = json.dumps(resources)
    else:
        tags["extraMetadata"] = json.dumps({"civitaiResources": resources})
    metadata = reader_module._metadata_from_tags(tags, 832, 480)
    output = reader_module.ReadImageSaverMetadata.execute(metadata).result
    assert len(output) == 14
    entries = ImageSaver.parse_manual_hashes(output[12], set(), False)
    assert set(entries) == {"civitai:123", "civitai:456"}
    assert entries["civitai:456"][1:] == (0.65, "urn:air:sdxl:lora:civitai:45@456")
    assert output[12] == metadata.final_hashes


def test_expander_exposes_resources_from_existing_metadata_node_without_source_tags(reader_module):
    resources = [{"modelVersionId": 456, "type": "lora", "weight": 0.8}]
    metadata = reader_module._metadata_from_tags({"parameters": PARAMETERS + ", Civitai resources: " + json.dumps(resources)}, 16, 16)
    metadata.source_metadata_tags = None
    output = reader_module.ReadImageSaverMetadata.execute(metadata).result
    entries = ImageSaver.parse_manual_hashes(output[12], set(), False)
    assert entries["model"] == (None, None, "ABCDEF1234")
    assert entries["civitai:456"][1] == 0.8


@pytest.mark.parametrize("download", [False, True])
def test_creator_and_expander_hashes_have_the_same_complete_format(reader_module, monkeypatch, download):
    monkeypatch.setattr(utils_civitai, "get_civitai_info", lambda path, hash: (
        {"id": 456, "name": "v1", "model": {"name": "Style", "type": "LORA"}} if hash == "DEADBEEF12" else None))
    metadata, hashes, _ = ImageSaverMetadata().get_metadata(
        additional_hashes="civitai:123,LORA:style:DEADBEEF12:0.65", download_civitai_data=download)
    assert metadata.source_metadata_tags is None
    assert reader_module.ReadImageSaverMetadata.execute(metadata).result[12] == hashes
    entries = ImageSaver.parse_manual_hashes(hashes, set(), download)
    assert len(entries) == 2
    assert "civitai:123" in entries
    if download:
        assert entries["civitai:456"][1] == 0.65
    else:
        assert next(value for value in entries.values() if value[2] == "DEADBEEF12")[1] == 0.65


@pytest.mark.parametrize("extension", IMAGE_FORMATS)
def test_reader_export_plus_extra_hash_writes_new_prompt_and_all_resources(reader_module, tmp_path, monkeypatch, extension):
    source_resources = [{"modelVersionId": 101, "type": "checkpoint"},
                        {"modelVersionId": 102, "type": "lora", "weight": 0.65}]
    original = PARAMETERS + ", Civitai resources: " + json.dumps(source_resources)
    source = tmp_path / f"source.{extension}"
    save_image(Image.new("RGB", (16, 16)), str(source), extension, 95, True, False,
               original, {}, {"workflow": {"nodes": []}}, True)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(source), raising=False)
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    recovered = reader_module.ImageSaverReadImageMetadata.execute(file=source.name).result[0]
    export = reader_module.ReadImageSaverMetadata.execute(recovered).result[12]
    calls = []

    def lookup(path, hash):
        calls.append(hash)
        if hash == "FEEDBEEF12":
            return {"id": 103, "name": "v1", "model": {"name": "Upscale LoRA", "type": "LORA"}}
        return None

    monkeypatch.setattr(utils_civitai, "get_civitai_info", lookup)
    metadata = ImageSaverMetadata().get_metadata(
        modelname="", positive="New upscale prompt", negative="New negative", width=1024, height=768,
        seed_value=99, steps=12, cfg=4, sampler_name="euler", scheduler_name="normal",
        additional_hashes=export + ",ExtraLoRA:FEEDBEEF12:0.8", download_civitai_data=True,
    )[0]
    image = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((32, 32, 3))))
    names = ImageSaver.save_images([image], "upscaled", extension, "", 95, True, False,
                                  {}, {"workflow": {"nodes": []}}, False, True, 0, "%Y", metadata)
    tags, _, _ = read_image_tags(str(tmp_path / names[0]))
    result = reader_module._metadata_from_tags(tags, 32, 32)
    output = reader_module.ReadImageSaverMetadata.execute(result).result
    resources = {r["modelVersionId"]: r for r in reader_module.read_resource_records(tags) if "modelVersionId" in r}
    exported = ImageSaver.parse_manual_hashes(output[12], set(), False)
    assert result.positive == "New upscale prompt"
    assert result.negative == "New negative"
    assert result.seed == 99
    assert result.steps == 12
    assert set(resources) == {101, 102, 103}
    assert resources[102]["weight"] == 0.65
    assert resources[103]["weight"] == 0.8
    assert all(f"civitai:{version}" in exported for version in (101, 102, 103))
    assert all(not hash.startswith(("civitai:", "urn:air:")) for hash in calls)
    assert recovered.a111_params == original


def test_graph_only_airs_and_saved_lora_hash_weights_are_exported(reader_module):
    tags = {
        "parameters": 'original\nSteps: 20, Hashes: {"LORA:local-style":"DEADBEEF12"}',
        "prompt": json.dumps({
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "urn:air:sdxl:checkpoint:civitai:11@101"}},
            "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "urn:air:sdxl:lora:civitai:12@102", "strength_model": 0.65}},
            "3": {"class_type": "LoraLoader", "inputs": {"lora_name": "local-style.safetensors", "strength_model": 0.8}},
        }),
    }
    metadata = reader_module._metadata_from_tags(tags, 16, 16)
    export = reader_module.ReadImageSaverMetadata.execute(metadata).result[12]
    entries = ImageSaver.parse_manual_hashes(export, set(), True)
    assert entries["civitai:101"][2] == "urn:air:sdxl:checkpoint:civitai:11@101"
    assert entries["civitai:102"][1] == 0.65
    assert entries["LORA:local-style"] == (None, 0.8, "DEADBEEF12")


@pytest.mark.parametrize("download", [False, True])
@pytest.mark.parametrize("extension", IMAGE_FORMATS)
def test_unresolved_weighted_hash_survives_save_and_reexport(reader_module, tmp_path, monkeypatch, download, extension):
    monkeypatch.setattr(utils_civitai, "get_civitai_info", lambda path, hash: None)
    metadata = ImageSaverMetadata().get_metadata(positive="Upscale prompt", negative="",
        additional_hashes="LORA:local-style:DEADBEEF12:0.65", download_civitai_data=download)[0]
    file = tmp_path / f"unresolved.{extension}"
    save_image(Image.new("RGB", (16, 16)), str(file), extension, 95, True, False,
               metadata.a111_params, None, None, False)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda _: str(file), raising=False)
    recovered = reader_module.ImageSaverReadImageMetadata.execute(file=file.name).result[0]
    refs = reader_module.ReadImageSaverMetadata.execute(recovered).result[12]
    entries = ImageSaver.parse_manual_hashes(refs, set(), download)
    assert entries["LORA:local-style"] == (None, 0.65, "DEADBEEF12")

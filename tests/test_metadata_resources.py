import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, features
from PIL.PngImagePlugin import PngInfo

import folder_paths
from image_saver_under_test.nodes import ImageSaver
from image_saver_under_test import nodes as saver_nodes
from image_saver_under_test.nodes_image_loader import to_saver_metadata
from image_saver_under_test.nodes_metadata_edit import EditMetadata, edit_preview_response
from image_saver_under_test.saver.image_metadata import parse_parameters, read_image_metadata
from image_saver_under_test.saver.metadata_resources import metadata_choices, resource_references, selection_values


PARAMETERS = ('Original prompt\nNegative prompt: blurry\nSteps: 24, Sampler: Euler, Seed: 42, Size: 384x256, '
              'Hashes: {"LORA:Local Style":"ABCDEF1234"}, '
              'Civitai resources: [{"modelName":"Original model","modelVersionId":101,"type":"checkpoint"},'
              '{"modelName":"Style, soft","modelVersionName":"v2","modelVersionId":102,"type":"lora","weight":0.65}], '
              'Custom field: "keep, this", Version: ComfyUI')


@pytest.fixture
def source(tmp_path, monkeypatch):
    path = tmp_path / "source.png"
    info = PngInfo()
    info.add_text("parameters", PARAMETERS)
    Image.new("RGB", (384, 256), "teal").save(path, pnginfo=info)
    monkeypatch.setattr(folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_temp_directory", lambda: str(tmp_path), raising=False)
    monkeypatch.setattr(folder_paths, "get_annotated_filepath", lambda name: str(tmp_path / name), raising=False)
    monkeypatch.setattr(folder_paths, "output_directory", str(tmp_path))
    return to_saver_metadata(read_image_metadata(str(path)))


def test_loader_and_expander_include_ids_and_hashes(source):
    exported = resource_references(source.a111_params, source.final_hashes)
    entries = ImageSaver.parse_manual_hashes(exported, set(), False)
    assert len(entries) == 3
    assert entries["civitai:102"][1] == 0.65
    assert "ABCDEF1234" in exported


def test_unedited_metadata_is_preserved_verbatim(source):
    source.a111_params = "\n" + PARAMETERS.replace(", Sampler:", ",  Sampler:") + "\n"
    result = EditMetadata().edit(source)[0]
    assert result is source
    assert result.a111_params == source.a111_params


def test_add_resources_preserves_original_settings_and_updates_only_explicit_weights(source):
    result = EditMetadata().edit(source, positive="Upscale prompt", additional_hashes=(
        "civitai:102:0.9,urn:air:sdxl:lora:civitai:12@103:0.8,New:FEEDBEEF12:0.5,Same:abcdef1234"))[0]
    positive, _, settings = parse_parameters(result.a111_params)
    assert positive == "Upscale prompt"
    assert settings["Custom field"] == "keep, this"
    resources = {item["modelVersionId"]: item for item in settings["Civitai resources"]}
    assert set(resources) == {101, 102, 103}
    assert resources[102]["modelName"] == "Style, soft"
    assert resources[102]["weight"] == 0.9
    assert resources[103]["weight"] == 0.8
    assert len(settings["Hashes"]) == 2
    assert source.a111_params == PARAMETERS


def test_adding_existing_resource_without_weight_preserves_source_weight(source):
    result = EditMetadata().edit(source, additional_hashes="civitai:102")[0]
    assert parse_parameters(result.a111_params)[2]["Civitai resources"][1]["weight"] == 0.65


def test_air_only_resources_keep_names_and_merge_by_version(source):
    source.a111_params = source.a111_params.replace(
        '"modelVersionId":102,"type":"lora",', '"air":"urn:air:sdxl:lora:civitai:12@102","type":"lora",')
    assert any(item["label"] == "Style, soft — v2" for item in metadata_choices(source)["resources"])
    edited = EditMetadata().edit(source, additional_hashes="civitai:102:0.9")[0]
    records = parse_parameters(edited.a111_params)[2]["Civitai resources"]
    assert len(records) == 2
    assert records[1]["weight"] == 0.9
    removed = EditMetadata().edit(source, remove_resources='["civitai:102"]')[0]
    assert len(parse_parameters(removed.a111_params)[2]["Civitai resources"]) == 1
    assert "@102" not in resource_references(removed.a111_params, removed.final_hashes)


def test_selected_resource_uses_identity_instead_of_typing_its_comma_name(source):
    choices = metadata_choices(source)
    selection = next(item for item in choices["resources"] if item["label"] == "Style, soft — v2")
    result = EditMetadata().edit(source, remove_resources=json.dumps([selection["value"]]))[0]
    records = parse_parameters(result.a111_params)[2]["Civitai resources"]
    assert [item["modelVersionId"] for item in records] == [101]
    assert "@102" not in result.final_hashes
    assert len(choices["resources"]) == 3


def test_remove_settings_also_removes_exported_version_ids(source):
    result = EditMetadata().edit(source, remove_settings='["Civitai resources","Custom field"]')[0]
    settings = parse_parameters(result.a111_params)[2]
    assert "Custom field" not in settings
    assert "Civitai resources" not in settings
    assert "@101" not in result.final_hashes and "@102" not in result.final_hashes
    assert "ABCDEF1234" in result.final_hashes


def test_removal_before_addition_can_replace_a_resource(source):
    result = EditMetadata().edit(source, remove_resources='["civitai:102"]', additional_hashes="civitai:102:0.2")[0]
    records = parse_parameters(result.a111_params)[2]["Civitai resources"]
    assert len([item for item in records if item["modelVersionId"] == 102]) == 1
    assert next(item for item in records if item["modelVersionId"] == 102)["weight"] == 0.2


def test_preview_and_execution_use_the_same_edits(source):
    changes = {"positive": "[original], sharp", "remove_resources": '["civitai:102"]',
               "remove_settings": '["Version"]', "additional_hashes": "civitai:103:0.8"}
    status, preview = edit_preview_response({"filename": "source.png", "edits": [changes]})
    executed = EditMetadata().execute(source, **changes)
    assert status == 200
    assert preview["result"] == executed["ui"]["metadata_editor"][0]["result"]
    assert executed["result"][0].positive == "Original prompt, sharp"
    assert len(preview["source"]["resources"]) == 3


@pytest.mark.parametrize("extension", ["png", "jpeg", "webp", pytest.param("avif", marks=pytest.mark.skipif(
    not features.check("avif"), reason="Pillow built without AVIF support"))])
def test_save_and_reload_retains_edits_and_complete_resources(source, tmp_path, extension):
    edited = EditMetadata().edit(source, positive="New prompt", additional_hashes="civitai:103:0.8")[0]
    image = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros((32, 32, 3))))
    names = ImageSaver.save_images([image], "edited", extension, "", 90, True, False, None, None,
                                   False, False, 0, "%Y", edited)
    loaded = to_saver_metadata(read_image_metadata(str(tmp_path / names[0])))
    assert loaded.positive == "New prompt"
    assert len(metadata_choices(loaded)["resources"]) == 4


def test_invalid_reference_is_reported_in_preview(source):
    status, response = edit_preview_response({"filename": "source.png", "edits": [{"additional_hashes": "civitai:0"}]})
    assert status == 400 and "Invalid additional" in response["error"]


@pytest.mark.parametrize("field,value", [("positive", []), ("remove_settings", {}), ("seed", "bad"), ("cfg", True)])
def test_preview_rejects_invalid_edit_value_types(source, field, value):
    status, response = edit_preview_response({"filename": "source.png", "edits": [{field: value}]})
    assert status == 400 and f"Invalid value for {field}" in response["error"]


def test_preview_rejects_paths_outside_comfy_directories(source):
    assert edit_preview_response({"filename": "../../secret.png", "edits": []})[0] == 403


def test_json_choices_preserve_commas():
    assert selection_values('["A, B", "Custom field"]') == ["A, B", "Custom field"]


def test_metadata_builder_accepts_reader_references_without_lookups(source, monkeypatch):
    monkeypatch.setattr(saver_nodes, "PromptMetadataExtractor", lambda _: SimpleNamespace(get_loras=lambda: {}, get_embeddings=lambda: {}))
    refs = resource_references(source.a111_params, source.final_hashes)
    metadata, hashes, _ = saver_nodes.ImageSaverMetadata().get_metadata(
        positive="New generation", additional_hashes=refs, download_civitai_data=False, easy_remix=False)
    settings = parse_parameters(metadata.a111_params)[2]
    assert {item["modelVersionId"] for item in settings["Civitai resources"]} == {101, 102}
    assert "@102:0.65" in hashes


def test_removing_steps_does_not_hide_the_remaining_settings(source):
    result = EditMetadata().edit(source, remove_settings='["Steps"]')[0]
    positive, _, settings = parse_parameters(result.a111_params)
    assert positive == "Original prompt"
    assert "Steps" not in settings
    assert settings["Seed"] == "42"
    assert len(metadata_choices(result)["resources"]) == 3

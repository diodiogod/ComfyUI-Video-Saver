"""
Tests for the "Civitai resources" entries built by utils_civitai.get_civitai_metadata.

Civitai reads `air` (or `modelVersionId`), `type` and `weight`; other tools such as
Stability Matrix need `modelVersionId` and `type` directly (issue #140).
"""
import pytest

from image_saver_under_test import utils_civitai
from image_saver_under_test import nodes as saver_nodes
from image_saver_under_test.utils_civitai import get_civitai_metadata
from image_saver_under_test.nodes import ImageSaver, ImageSaverMetadata
from image_saver_under_test.saver.video import extract_json_after


def civitai_info(version_id, model_name, version_name, model_type, air=None):
    info = {"id": version_id, "name": version_name, "model": {"name": model_name}}
    if model_type is not None:
        info["model"]["type"] = model_type
    if air is not None:
        info["air"] = air
    return info


INFOS = {
    "ckpt_hash": civitai_info(290640, "Pony Diffusion V6 XL", "V6", "Checkpoint", "urn:air:sdxl:checkpoint:civitai:257749@290640"),
    "lora_hash": civitai_info(333590, "Some Style", "Anime 2", "LORA", "urn:air:sdxl:lora:civitai:296485@333590"),
    "locon_hash": civitai_info(111, "Some LoCon", "v1", "LoCon", "urn:air:sdxl:lycoris:civitai:100@111"),
    "embed_hash": civitai_info(5637, "Deep Negative V1.x", "V1 75T", "TextualInversion"),
    "untyped_hash": civitai_info(42, "Old Cache", "v1", None),
}


@pytest.fixture(autouse=True)
def fake_civitai(monkeypatch):
    monkeypatch.setattr(utils_civitai, "get_civitai_info", lambda path, hash: INFOS.get(hash))


def resources_by_version(**kwargs):
    defaults = dict(modelname="pony", ckpt_path="pony.safetensors", modelhash="ckpt_hash",
                    loras={}, embeddings={}, manual_entries={}, download_civitai_data=True)
    resources, _, _ = get_civitai_metadata(**(defaults | kwargs))
    return {r["modelVersionId"]: r for r in resources}


def test_checkpoint_entry_has_version_id_type_and_air():
    resources = resources_by_version()
    assert resources[290640] == {
        "modelName": "Pony Diffusion V6 XL",
        "modelVersionName": "V6",
        "modelVersionId": 290640,
        "type": "checkpoint",
        "air": "urn:air:sdxl:checkpoint:civitai:257749@290640",
    }


def test_lora_types_are_lowercased_civitai_types():
    resources = resources_by_version(loras={
        "style": ("style.safetensors", 0.8, "lora_hash"),
        "locon": ("locon.safetensors", 0.5, "locon_hash"),
    })
    assert resources[333590]["type"] == "lora"
    assert resources[333590]["weight"] == 0.8
    assert resources[111]["type"] == "locon"


def test_embedding_without_air_still_has_version_id():
    resources = resources_by_version(embeddings={"neg": ("neg.pt", 1.0, "embed_hash")})
    assert resources[5637]["type"] == "textualinversion"
    assert "air" not in resources[5637]


def test_missing_model_type_omits_type():
    resources = resources_by_version(manual_entries={"old": (None, None, "untyped_hash")})
    assert "type" not in resources[42]


def test_no_resources_without_civitai_download():
    resources, hashes, add_model_hash = get_civitai_metadata(
        "pony", "pony.safetensors", "ckpt_hash", {}, {}, {}, download_civitai_data=False)
    assert resources == []
    assert add_model_hash == "ckpt_hash"


@pytest.mark.parametrize("download", [False, True])
def test_ids_and_airs_are_carried_without_hash_lookups(monkeypatch, download):
    calls = []

    def lookup(path, hash):
        calls.append(hash)
        return INFOS.get(hash)

    monkeypatch.setattr(utils_civitai, "get_civitai_info", lookup)
    manual = ImageSaver.parse_manual_hashes(
        "civitai:123456:0,urn:air:sdxl:lora:civitai:12@234567:0.65,Style:lora_hash:0.8", set(), download)
    resources, hashes, _ = get_civitai_metadata("pony", "pony.safetensors", "ckpt_hash", {}, {}, manual, download)
    by_id = {resource["modelVersionId"]: resource for resource in resources}
    assert by_id[123456] == {"modelVersionId": 123456, "weight": 0.0}
    assert by_id[234567]["weight"] == 0.65
    assert all(not hash.startswith(("civitai:", "urn:air:")) for hash in calls)
    if not download:
        assert calls == []
        assert hashes["Style"] == "lora_hash"


def test_direct_and_hash_references_to_same_version_are_not_added_twice():
    manual = ImageSaver.parse_manual_hashes("civitai:333590:0.5", set(), True)
    resources = resources_by_version(loras={"style": ("style.safetensors", 0.8, "lora_hash")}, manual_entries=manual)
    assert resources[333590]["weight"] == 0.8
    all_resources, _, _ = get_civitai_metadata("", "", "", {"style": ("style.safetensors", 0.8, "lora_hash")}, {}, manual, True)
    assert len(all_resources) == 1


def test_manual_resource_references_deduplicate_by_version_and_keep_hashes():
    manual = ImageSaver.parse_manual_hashes(
        "civitai:123:0.65,urn:air:sdxl:lora:civitai:12@123:0.2,ABCDEF1234,Name:abcdef1234,DEADBEEF12:0.8", set(), True)
    assert len(manual) == 3
    assert manual["civitai:123"] == (None, 0.65, "civitai:123")
    assert manual["manual2"] == (None, 0.8, "DEADBEEF12")


def test_invalid_explicit_references_are_not_sent_to_civitai(monkeypatch):
    manual = ImageSaver.parse_manual_hashes("civitai:0,urn:air:bad,civitai:123:NaN", set(), True)
    assert manual == {}


def test_metadata_creation_keeps_explicit_ids_when_download_is_disabled():
    metadata = ImageSaverMetadata().get_metadata(
        modelname="", positive="Upscale prompt", negative="Upscale negative", width=1024, height=768,
        seed_value=99, steps=12, cfg=4, sampler_name="euler", scheduler_name="normal",
        additional_hashes="civitai:123456:0.65,urn:air:sdxl:lora:civitai:12@234567:0.8", download_civitai_data=False,
    )[0]
    resources = extract_json_after(metadata.a111_params, "Civitai resources:")
    assert {r["modelVersionId"] for r in resources} == {123456, 234567}
    assert resources[0]["weight"] == 0.65
    entries = ImageSaver.parse_manual_hashes(metadata.final_hashes, set(), False)
    assert set(entries) == {"civitai:123456", "civitai:234567"}
    assert entries["civitai:123456"][1] == 0.65
    assert entries["civitai:234567"][1] == 0.8
    assert metadata.positive == "Upscale prompt"


@pytest.mark.parametrize("download", [False, True])
def test_current_model_and_carried_hash_are_not_repeated(monkeypatch, download):
    monkeypatch.setattr(saver_nodes, "full_checkpoint_path_for", lambda _: "pony.safetensors")
    monkeypatch.setattr(saver_nodes, "get_sha256", lambda _: "ckpt_hash")
    metadata, hashes, _ = ImageSaverMetadata().get_metadata(
        modelname="pony", additional_hashes="OldModel:CKPT_HASH,ckpt_hash", download_civitai_data=download)
    entries = ImageSaver.parse_manual_hashes(hashes, set(), False)
    assert len(entries) == 1
    if download:
        resources = extract_json_after(metadata.a111_params, "Civitai resources:")
        assert [r["modelVersionId"] for r in resources] == [290640]
        assert set(entries) == {"civitai:290640"}
    else:
        assert next(iter(entries.values()))[2] == "ckpt_hash"


def test_current_model_and_carried_version_are_not_repeated(monkeypatch):
    monkeypatch.setattr(saver_nodes, "full_checkpoint_path_for", lambda _: "pony.safetensors")
    monkeypatch.setattr(saver_nodes, "get_sha256", lambda _: "ckpt_hash")
    metadata, hashes, _ = ImageSaverMetadata().get_metadata(
        modelname="pony", additional_hashes="civitai:290640,urn:air:sdxl:checkpoint:civitai:257749@290640,OldModel:ckpt_hash")
    resources = extract_json_after(metadata.a111_params, "Civitai resources:")
    assert len(resources) == 1
    assert resources[0]["modelVersionId"] == 290640
    assert resources[0]["modelName"] == "Pony Diffusion V6 XL"
    entries = ImageSaver.parse_manual_hashes(hashes, set(), False)
    assert len(entries) == 1
    assert set(entries) == {"civitai:290640"}

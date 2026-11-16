import pytest

from .resource_refs import export_resource_references, parse_reference, resource_from_identifier


@pytest.mark.parametrize("text,expected", [
    ("ABCDEF1234", (None, None, "ABCDEF1234")),
    (" Style : ABCDEF1234 ", ("Style", None, "ABCDEF1234")),
    ("ABCDEF1234:0.65", (None, 0.65, "ABCDEF1234")),
    ("Style:ABCDEF1234:-0.5", ("Style", -0.5, "ABCDEF1234")),
    ("LORA:style:123456ABCD", ("LORA:style", None, "123456ABCD")),
    ("LORA:style:1234567890", ("LORA:style", None, "1234567890")),
    ("LORA:style:123456ABCD:0", ("LORA:style", 0.0, "123456ABCD")),
    ("Model:1234567890", ("Model", None, "1234567890")),
    ("civitai:123456", (None, None, "civitai:123456")),
    ("CIVITAI:123456:0.65", (None, 0.65, "civitai:123456")),
    ("urn:air:sdxl:lora:civitai:12@123456:0.65", (None, 0.65, "urn:air:sdxl:lora:civitai:12@123456")),
    ("urn:air:sd1:embedding:civitai:0@123456:-1", (None, -1.0, "urn:air:sd1:embedding:civitai:0@123456")),
])
def test_legacy_hashes_and_explicit_resources(text, expected):
    assert parse_reference(text) == expected


@pytest.mark.parametrize("text", ["", "civitai:0", "civitai:-1", "civitai:1.5", "civitai:123:bad", "civitai:123:1e999",
                                      "urn:air:broken", "urn:air:sdxl:lora:civitai:12@0", "X" * 17, "hash with spaces"])
def test_invalid_references_are_not_treated_as_hashes(text):
    assert parse_reference(text) is None


def test_export_keeps_identity_type_and_weights_without_resolving_hashes():
    resources = [{"modelVersionId": 101, "type": "checkpoint"},
                 {"modelVersionId": 102, "air": "urn:air:sdxl:lora:civitai:12@102", "weight": 0.65},
                 {"modelVersionId": 103, "weight": 0},
                 {"modelVersionId": 102, "weight": 0.2}]
    text = export_resource_references(resources, "model:ABCDEF1234,LORA:style:123456ABCD,model:abcdef1234")
    entries = [parse_reference(item) for item in text.split(",")]
    assert len(entries) == 5
    assert resource_from_identifier(entries[0][2]) == {
        "modelVersionId": 101, "type": "checkpoint", "air": "urn:air:other:checkpoint:civitai:0@101"}
    assert entries[1] == (None, 0.65, "urn:air:sdxl:lora:civitai:12@102")
    assert entries[2] == (None, 0.0, "civitai:103")
    assert entries[3:] == [("model", None, "ABCDEF1234"), ("LORA:style", None, "123456ABCD")]


def test_export_unknown_or_missing_resources_is_empty():
    assert export_resource_references([None, {}, {"modelVersionId": False}, {"modelVersionId": 0}], "") == ""

"""Read image/video generation metadata without model hashing or lookups."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import av
import piexif
import piexif.helper
from PIL import Image

from .video import MAX_METADATA_VALUE_BYTES, extract_json_after
from .resource_refs import export_resource_references, parse_reference, resource_from_identifier


GENERATION_TAG_NAMES = {
    name.lower(): name for name in (
        "parameters", "prompt", "workflow", "prompt_original", "workflow_original",
        "prompt_postprocess", "civitaiResources", "extraMetadata",
    )
}


class MissingGenerationMetadata(ValueError):
    """A valid media file has no generation information to recover."""


def _check_generation_tags(tags: dict[str, str], media: str) -> None:
    for name, value in tags.items():
        if len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES:
            raise ValueError(f"Metadata field '{name}' exceeds the 2 MiB safety limit.")
    if not any(tags.get(name, "").strip() not in {"", "{}", "[]", "null"}
               for name in ("parameters", "prompt", "workflow", "civitaiResources", "extraMetadata")):
        raise MissingGenerationMetadata(f"No generation metadata was found in this {media}. Use a source saved with metadata enabled.")


def read_video_tags(source) -> tuple[dict[str, str], int, int]:
    """Read container/stream tags; PyAV does not decode the video or audio."""
    position = source.tell() if not isinstance(source, (str, os.PathLike)) else None
    try:
        if position is not None:
            source.seek(0)
        with av.open(source, mode="r") as container:
            if not container.streams.video:
                raise ValueError("The selected file contains no video stream.")
            stream = container.streams.video[0]
            raw_tags = dict(stream.metadata)
            raw_tags.update(container.metadata)
            tags = {}
            for key, value in raw_tags.items():
                name = GENERATION_TAG_NAMES.get(key.lower())
                if name is None:
                    continue
                if len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES:
                    raise ValueError(f"Video metadata field '{name}' exceeds the 2 MiB safety limit.")
                # Native ComfyUI encoders may JSON-quote string metadata.
                if value.startswith('"'):
                    try:
                        decoded = json.loads(value)
                        if isinstance(decoded, str):
                            value = decoded
                    except json.JSONDecodeError:
                        pass
                tags[name] = value
            _check_generation_tags(tags, "video")
            return tags, stream.codec_context.width, stream.codec_context.height
    finally:
        if position is not None:
            source.seek(position)


def read_image_tags(source) -> tuple[dict[str, str], int, int]:
    """Read PNG text and JPEG/WebP/AVIF EXIF written by Image Saver or ComfyUI."""
    with Image.open(source) as image:
        tags = {
            GENERATION_TAG_NAMES[key.lower()]: value for key, value in image.info.items()
            if key.lower() in GENERATION_TAG_NAMES and isinstance(value, str)
        }
        exif_bytes = image.info.get("exif")
        if exif_bytes:
            exif = piexif.load(exif_bytes)
            comment = exif.get("Exif", {}).get(piexif.ExifIFD.UserComment)
            if comment:
                tags.setdefault("parameters", piexif.helper.UserComment.load(comment))
            for value in exif.get("0th", {}).values():
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                if not isinstance(value, str):
                    continue
                key, separator, payload = value.partition(":")
                name = GENERATION_TAG_NAMES.get(key.lower())
                if separator and name:
                    tags.setdefault(name, payload)
        _check_generation_tags(tags, "image")
        return tags, image.width, image.height


def _json_object(value: str, name: str) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"The '{name}' metadata is not valid JSON.") from error


def read_resource_records(tags: dict[str, str]) -> list[dict[str, Any]]:
    parameters = tags.get("parameters", "")
    resources = extract_json_after(parameters, "Civitai resources:")
    if resources is None:
        resources = _json_object(tags.get("civitaiResources", ""), "civitaiResources")
    if not resources:
        extra = _json_object(tags.get("extraMetadata", ""), "extraMetadata")
        resources = extra.get("civitaiResources", []) if isinstance(extra, dict) else []
    resources = [record.copy() for record in resources if isinstance(record, dict)] if isinstance(resources, list) else []
    references = extract_json_after(parameters, "Resource references:")
    for entry in references if isinstance(references, list) else []:
        parsed = parse_reference(entry) if isinstance(entry, str) else None
        if parsed is None:
            continue
        name, weight, identifier = parsed
        resource = resource_from_identifier(identifier) or {"hash": identifier, "name": name}
        if weight is not None:
            resource["weight"] = weight
        key = "modelVersionId" if "modelVersionId" in resource else "hash"
        if not any(existing.get(key) == resource[key] for existing in resources):
            resources.append(resource)
    graph = _json_object(tags.get("prompt", ""), "prompt")
    hashes = extract_json_after(parameters, "Hashes:") or {}
    for node in graph.values() if isinstance(graph, dict) else []:
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        inputs = node["inputs"]
        for key in ("lora_name", "unet_name", "ckpt_name"):
            filename = inputs.get(key)
            if not isinstance(filename, str):
                continue
            resource = resource_from_identifier(filename)
            if resource is None and key == "lora_name" and isinstance(hashes, dict):
                stem = os.path.splitext(os.path.basename(filename.replace("\\", "/")))[0].casefold()
                hash_name = next((name for name in hashes if isinstance(name, str) and name.lower().startswith("lora:")
                                  and name.split(":", 1)[1].casefold() == stem), None)
                if hash_name and isinstance(hashes[hash_name], str):
                    resource = {"hash": hashes[hash_name], "name": hash_name, "type": "lora"}
            if resource is None:
                continue
            weight = inputs.get("strength_model")
            if key == "lora_name" and isinstance(weight, (int, float)) and not isinstance(weight, bool) and math.isfinite(weight):
                resource["weight"] = weight
            identity_key = "modelVersionId" if "modelVersionId" in resource else "hash"
            if not any(existing.get(identity_key) == resource[identity_key] for existing in resources):
                resources.append(resource)
    return resources


def metadata_fields(tags: dict[str, str], width: int, height: int) -> dict[str, Any]:
    """Recover saver fields while retaining the source tags as the authority."""
    parameters = tags.get("parameters", "")
    positive, negative, settings = "", "", {}
    if parameters:
        lines = list(re.finditer(r"^(?:Steps|Sampler|Seed|CFG scale):\s*", parameters, flags=re.MULTILINE))
        boundary = lines[-1].start() if lines else len(parameters)
        prompts = parameters[:boundary].rstrip()
        parts = re.split(r"\nNegative prompt:\s?", prompts, maxsplit=1)
        positive = parts[0].strip()
        negative = parts[1].strip() if len(parts) == 2 else ""
        settings_text = parameters[boundary:]
        matches = list(re.finditer(r"(?:^|,\s*)([\w /-]+):\s*", settings_text))
        settings = {
            match[1].strip(): settings_text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(settings_text)].strip()
            for index, match in enumerate(matches)
        }

    graph = _json_object(tags.get("prompt", ""), "prompt")
    graph = graph if isinstance(graph, dict) else {}
    samplers = [node.get("inputs", {}) for node in graph.values()
                if isinstance(node, dict) and node.get("class_type") in {"KSampler", "KSamplerAdvanced"}]
    inputs = samplers[-1] if samplers else {}

    def linked_inputs(link):
        node = graph.get(str(link[0]), {}) if isinstance(link, list) and len(link) == 2 else {}
        return node.get("inputs", {}) if isinstance(node, dict) else {}

    def text_from(link):
        value = linked_inputs(link)
        return next((value[key] for key in ("text", "text_g", "positive", "negative")
                     if isinstance(value.get(key), str)), "")

    positive = positive or text_from(inputs.get("positive"))
    negative = negative or text_from(inputs.get("negative"))
    latent = linked_inputs(inputs.get("latent_image"))
    size = re.fullmatch(r"(\d+)x(\d+)", settings.get("Size", ""))

    def number(setting, key, default, integer=False):
        value = settings.get(setting, inputs.get(key, default))
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            return default
        try:
            result = int(value) if integer else float(value)
            return result if math.isfinite(result) else default
        except (ValueError, OverflowError):
            return default

    hashes = extract_json_after(parameters, "Hashes:") or {}
    hash_parts = [f"{name}:{value}" for name, value in hashes.items() if isinstance(value, str)] if isinstance(hashes, dict) else []
    if settings.get("Model hash") and not any(part.lower().startswith("model:") for part in hash_parts):
        hash_parts.insert(0, f"model:{settings['Model hash']}")
    resources = read_resource_records(tags)
    if resources and not tags.get("civitaiResources"):
        tags = {**tags, "civitaiResources": json.dumps(resources, ensure_ascii=False)}

    modelname = settings.get("Model", "")
    if not modelname:
        for node in graph.values():
            if isinstance(node, dict) and node.get("class_type") in {"UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader"}:
                controls = node.get("inputs", {})
                modelname = controls.get("unet_name", controls.get("ckpt_name", ""))
                if isinstance(modelname, str):
                    break
                modelname = ""

    return dict(
        modelname=modelname, positive=positive, negative=negative,
        width=int(size[1]) if size else latent["width"] if isinstance(latent.get("width"), int) else width,
        height=int(size[2]) if size else latent["height"] if isinstance(latent.get("height"), int) else height,
        seed=number("Seed", "seed", inputs.get("noise_seed", 0), integer=True),
        steps=number("Steps", "steps", 20, integer=True),
        cfg=number("CFG scale", "cfg", 7.0),
        sampler_name=inputs.get("sampler_name", settings.get("Sampler", "")),
        scheduler_name=inputs.get("scheduler", settings.get("Schedule type", settings.get("Scheduler", "normal"))),
        denoise=number("Denoising strength", "denoise", 1.0),
        clip_skip=number("Clip skip", "clip_skip", 0, integer=True),
        custom="", additional_hashes="", ckpt_path="",
        a111_params=parameters, final_hashes=export_resource_references(resources, ",".join(hash_parts)), source_metadata_tags=dict(tags),
    )

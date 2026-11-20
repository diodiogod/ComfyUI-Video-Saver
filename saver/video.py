"""Video metadata helpers for Civitai-compatible Image Saver output."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

import av

from .resource_refs import parse_reference, resource_identity
from .image_metadata import parse_parameters


MAX_METADATA_VALUE_BYTES = 2 * 1024 * 1024


def extract_json_after(text: str, marker: str) -> Any | None:
    """Decode the first JSON value following ``marker``."""
    if not text:
        return None

    marker_position = text.lower().find(marker.lower())
    if marker_position < 0:
        return None

    payload = text[marker_position + len(marker):].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
        return value
    except json.JSONDecodeError:
        return None


def _json_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def native_video_metadata(tags: dict[str, str]) -> dict[str, Any]:
    """Convert our container tags to values accepted by native VIDEO.

    The native component encoder JSON-serializes every metadata value itself.
    ``parameters`` is deliberately omitted here because it is an A1111 text
    field and must be written as a raw container tag by the lossless remux
    performed after the native encoder finishes.
    """
    native: dict[str, Any] = {}
    for key, value in tags.items():
        if key == "parameters":
            continue
        try:
            native[key] = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            native[key] = value
    return native


def build_video_metadata(
    metadata: Any | None,
    prompt: dict[str, Any] | None,
    extra_pnginfo: dict[str, Any] | None,
) -> dict[str, str]:
    """Build container tags for Civitai's native ComfyUI video metadata reader.

    ``parameters`` remains the A1111-compatible source of positive/negative
    prompts, settings, hashes, and Civitai resources. A compact compatibility
    graph supplies Civitai's native reader with the explicit saver settings.
    The original execution graph and UI workflow are preserved separately.
    """
    source_tags = getattr(metadata, "source_metadata_tags", None)
    if source_tags is not None:
        tags = dict(source_tags)
        if "prompt" not in tags and tags.get("parameters"):
            resources = extract_json_after(tags["parameters"], "Civitai resources:")
            if resources is None and tags.get("civitaiResources"):
                resources = json.loads(tags["civitaiResources"])
            compatibility_prompt = resources_to_prompt_json(metadata, resources)
            if compatibility_prompt:
                tags["prompt"] = compatibility_prompt
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                if key == "workflow":
                    if tags.get("workflow") and tags["workflow"] != "{}":
                        tags.setdefault("workflow_original", tags["workflow"])
                    tags["workflow"] = _json_value(value)
                elif key not in tags:
                    tags[str(key)] = _json_value(value)
        if prompt is not None:
            tags["prompt_postprocess"] = _json_value(prompt)
        if prompt is None and extra_pnginfo is None:
            for key in ("workflow_original", "prompt_original", "prompt_postprocess"):
                tags.pop(key, None)
            tags["workflow"] = "{}"
        tags.setdefault("workflow", "{}")
        return _check_metadata_size(tags)

    tags: dict[str, str] = {}

    parameters = getattr(metadata, "a111_params", "") if metadata is not None else ""
    if parameters:
        tags["parameters"] = parameters

    extra = dict(extra_pnginfo or {})
    for key, value in extra.items():
        tags[str(key)] = _json_value(value)

    resources = extract_json_after(parameters, "Civitai resources:")
    prompt_json = resources_to_prompt_json(metadata, resources, prompt)
    if prompt_json:
        tags["prompt"] = prompt_json
        # The compatibility graph describes the saved metadata, not the full
        # execution. Retain the real graph for other readers and diagnostics.
        if prompt is not None:
            tags["prompt_original"] = _json_value(prompt)
        # Civitai also stores these two fields as a JSON object. An explicit
        # empty workflow keeps that object valid when workflow saving is off.
        tags.setdefault("workflow", "{}")
    elif prompt is not None:
        tags["prompt"] = _json_value(prompt)

    if resources is not None:
        serialized_resources = _json_value(resources)
        tags["civitaiResources"] = serialized_resources

        # Keep a structured fallback for readers that accept extraMetadata.
        extra_metadata = None
        if "extraMetadata" in tags:
            try:
                extra_metadata = json.loads(tags["extraMetadata"])
            except json.JSONDecodeError:
                pass
        if isinstance(extra_metadata, dict):
            extra_metadata.setdefault("civitaiResources", resources)
            tags["extraMetadata"] = _json_value(extra_metadata)
        elif "extraMetadata" not in tags:
            tags["extraMetadata"] = _json_value({"civitaiResources": resources})

    return _check_metadata_size(tags)


def _check_metadata_size(tags: dict[str, str]) -> dict[str, str]:
    for key, value in tags.items():
        if len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES:
            raise ValueError(f"Video metadata field '{key}' exceeds the 2 MiB safety limit.")
    return tags


def update_edited_generation_tags(original, edited, parameters_before, removed_settings, removed_resources):
    """Synchronize the text and Civitai-facing tags, keeping the source workflow and original prompt."""
    tags = dict(original.source_metadata_tags)
    previous = parse_parameters(parameters_before)[2]
    current = parse_parameters(edited.a111_params)[2]
    records = current.get("Civitai resources", [])
    records = records if isinstance(records, list) else []
    if edited.a111_params:
        tags["parameters"] = edited.a111_params
    else:
        tags.pop("parameters", None)
    if records:
        tags["civitaiResources"] = _json_value(records)
    else:
        tags.pop("civitaiResources", None)
    extra = json.loads(tags.get("extraMetadata") or "{}")
    if isinstance(extra, dict):
        if records:
            extra["civitaiResources"] = records
        else:
            extra.pop("civitaiResources", None)
        if extra:
            tags["extraMetadata"] = _json_value(extra)
        else:
            tags.pop("extraMetadata", None)

    old_prompt = json.loads(tags.get("prompt") or "{}")
    if old_prompt:
        tags.setdefault("prompt_original", tags["prompt"])
    old_hashes, new_hashes = previous.get("Hashes"), current.get("Hashes")
    removed_names = {_filename_key(name) for name in removed_resources}
    if isinstance(old_hashes, dict):
        remaining = {str(value).casefold() for value in new_hashes.values()} if isinstance(new_hashes, dict) else set()
        removed_names.update(_filename_key(name.split(":", 1)[-1]) for name, value in old_hashes.items()
                             if str(value).casefold() not in remaining)

    fallback_nodes = {}
    for index, node in enumerate(_source_resource_nodes(old_prompt if isinstance(old_prompt, dict) else {})):
        inputs = node["inputs"]
        filename = next(iter(inputs.values()))
        if (filename.lower().startswith(("urn:air:", "civitai:")) or _filename_key(filename) in removed_names
                or (_resource_node_kind(inputs) == "model" and edited.modelname != original.modelname)):
            continue
        fallback_nodes[str(index)] = node

    graph = json.loads(resources_to_prompt_json(edited, records, fallback_nodes))
    removed = {key.casefold() for key in removed_settings}
    original_sampler = next((node["inputs"] for node in old_prompt.values()
                             if isinstance(node, dict) and node.get("class_type") == "KSampler"
                             and isinstance(node.get("inputs"), dict)), None) if isinstance(old_prompt, dict) else None
    sampler = next(node["inputs"] for node in graph.values() if node["class_type"] == "KSampler")
    for key, field in (("Steps", "steps"), ("Seed", "seed"), ("CFG scale", "cfg"),
                       ("Sampler", "sampler_name"), ("Size", "latent_image"),
                       ("Denoising strength", "denoise"), ("Schedule type", "scheduler")):
        if key not in current and (key.casefold() in removed or
                                   (original_sampler is not None and field not in original_sampler and key not in previous)):
            sampler.pop(field, None)
    tags["prompt"] = _json_value(graph)
    return tags

def _source_resource_nodes(prompt: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Keep recognizable resource filenames when Civitai lookup data is absent."""
    fields = {
        "CheckpointLoader": "ckpt_name", "CheckpointLoaderSimple": "ckpt_name",
        "CheckpointLoader|pysssss": "ckpt_name", "CheckpointLoaderSimple|pysssss": "ckpt_name",
        "UNETLoader": "unet_name", "UnetLoaderGGUF": "unet_name",
        "LoraLoader": "lora_name", "LoraLoaderModelOnly": "lora_name",
        "LoraLoader|pysssss": "lora_name", "LoraLoaderModelOnly|pysssss": "lora_name",
        "VAELoader": "vae_name", "UpscaleModelLoader": "model_name",
        "ControlNetLoader": "control_net_name",
    }
    result = []
    for node in (prompt or {}).values():
        if not isinstance(node, dict) or node.get("class_type") not in fields:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        key = fields[node["class_type"]]
        # A filename supplied by another node is left in prompt_original;
        # copying that link alone would point into the compatibility graph.
        if not isinstance(inputs.get(key), str) or not inputs[key]:
            continue
        kept = {key: inputs[key]}
        for strength in ("strength_model", "strength_clip"):
            value = inputs.get(strength)
            if isinstance(value, (int, float)) and math.isfinite(value):
                kept[strength] = value
        result.append({"class_type": node["class_type"], "inputs": kept})
    return result


def _resource_node_kind(inputs: dict[str, Any]) -> str:
    """Group source filename controls without guessing a model from list order."""
    for key, kind in (("unet_name", "model"), ("ckpt_name", "model"),
                      ("lora_name", "lora"), ("vae_name", "vae"), ("model_name", "upscaler")):
        if key in inputs:
            return kind
    return "controlnet"


def _filename_key(name: str) -> str:
    """Compare local resource filenames with the stem names in A1111 hashes."""
    path = Path(name.replace("\\", "/"))
    return (path.stem if path.suffix.lower() in {".safetensors", ".ckpt", ".pt", ".bin", ".gguf"} else path.name).casefold()


def resources_to_prompt_json(
    metadata: Any | None,
    resources: list[dict[str, Any]] | None,
    prompt: dict[str, Any] | None = None,
) -> str | None:
    """Describe saver settings as nodes understood by Civitai's graph parser.

    This is a metadata compatibility graph, not a runnable replacement for the
    saved UI workflow. Explicit resource IDs take priority; recognizable source
    filenames are retained for kinds without a resolved resource.
    """
    if (not isinstance(getattr(metadata, "positive", None), str)
            or not isinstance(getattr(metadata, "negative", None), str)):
        return None

    graph: dict[str, Any] = {}

    def add_node(class_type: str, inputs: dict[str, Any]) -> list[Any]:
        node_id = str(len(graph) + 1)
        graph[node_id] = {"class_type": class_type, "inputs": inputs}
        return [node_id, 0]

    modelname = getattr(metadata, "modelname", "")
    local_model = modelname.strip() if isinstance(modelname, str) else ""
    if local_model.lower().startswith(("urn:air:", "civitai:")):
        local_model = ""
    local_model = Path(local_model.replace("\\", "/")).name
    model_ref = None
    resolved_kinds = set()
    seen = set()
    for resource in resources if isinstance(resources, list) else []:
        if not isinstance(resource, dict):
            continue
        identity = resource_identity(resource)
        if not identity or identity[0] in seen:
            continue
        air, kind = identity
        seen.add(air)
        is_model = kind in {"checkpoint", "unet", "diffusionmodel", "model"}
        if is_model:
            if "model" not in resolved_kinds and local_model.lower() not in {"", "..", "unknown", "none"}:
                # Civitai displays the first model and uses its local name to recognize an external upload.
                # Keep the separate AIR below so the same checkpoint version is still linked.
                add_node("UNETLoader", {"unet_name": local_model})
            model_ref = add_node("UNETLoader", {"unet_name": air})
        else:
            weight = resource.get("weight", 1.0)
            if not isinstance(weight, (int, float)) or not math.isfinite(weight):
                weight = 1.0
            inputs = {"lora_name": air, "strength_model": weight}
            if model_ref:
                inputs["model"] = model_ref
            # This recognized loader is a carrier for auxiliary AIRs too.
            # Civitai uses the AIR's real type, not the loader's LoRA label.
            reference = add_node("LoraLoaderModelOnly", inputs)
            if kind in {"lora", "lycoris", "locon"}:
                model_ref = reference
        resolved_kinds.add("model" if is_model else "lora" if kind in {"lora", "lycoris", "locon"} else kind)

    for node in _source_resource_nodes(prompt):
        inputs = node["inputs"]
        kind = _resource_node_kind(inputs)
        if kind in resolved_kinds:
            continue
        reference = add_node(node["class_type"], inputs)
        if kind in {"model", "lora"}:
            model_ref = reference

    # A saver can have metadata without an embedded workflow. Retain its model
    # filename even when Civitai downloads were disabled or the lookup failed.
    has_model = any("unet_name" in node["inputs"] or "ckpt_name" in node["inputs"] for node in graph.values())
    if not has_model and isinstance(modelname, str) and modelname and modelname.lower() not in {"unknown", "none"}:
        model_ref = add_node("UNETLoader", {"unet_name": modelname})

    # Hash entries are unresolved resources when lookups were attempted, or
    # the complete local list when downloads were disabled. Their LoRA names
    # remain useful even if no original graph was embedded.
    hashes = extract_json_after(getattr(metadata, "a111_params", ""), "Hashes:")
    references = extract_json_after(getattr(metadata, "a111_params", ""), "Resource references:")
    hash_weights = {}
    for entry in references if isinstance(references, list) else []:
        parsed = parse_reference(entry) if isinstance(entry, str) else None
        if parsed and parsed[1] is not None:
            hash_weights[parsed[2].lower()] = parsed[1]
    weighted_names = {_filename_key(name.split(":", 1)[1]): hash_weights[str(value).lower()]
                      for name, value in hashes.items() if isinstance(name, str) and name.lower().startswith("lora:")
                      and str(value).lower() in hash_weights} if isinstance(hashes, dict) else {}
    for node in graph.values():
        filename = node["inputs"].get("lora_name")
        if isinstance(filename, str) and _filename_key(filename) in weighted_names:
            node["inputs"]["strength_model"] = weighted_names[_filename_key(filename)]
    lora_names = {
        _filename_key(node["inputs"]["lora_name"])
        for node in graph.values() if "lora_name" in node["inputs"]
    }
    for name in hashes if isinstance(hashes, dict) else []:
        if not isinstance(name, str) or not name.lower().startswith("lora:"):
            continue
        filename = name.split(":", 1)[1]
        if not filename or _filename_key(filename) in lora_names:
            continue
        lora_names.add(_filename_key(filename))
        inputs = {"lora_name": filename, "strength_model": hash_weights.get(str(hashes[name]).lower(), 1.0)}
        if model_ref:
            inputs["model"] = model_ref
        model_ref = add_node("LoraLoaderModelOnly", inputs)

    positive = add_node("CLIPTextEncode", {"text": metadata.positive.strip()})
    negative = add_node("CLIPTextEncode", {"text": metadata.negative.strip()})
    latent = add_node("EmptyLatentImage", {"width": getattr(metadata, "width", 512), "height": getattr(metadata, "height", 512)})
    inputs = {
        "positive": positive, "negative": negative, "latent_image": latent,
        "seed": getattr(metadata, "seed", 0), "steps": getattr(metadata, "steps", 20),
        "cfg": getattr(metadata, "cfg", 7.0), "sampler_name": getattr(metadata, "sampler_name", "euler"),
        "scheduler": getattr(metadata, "scheduler_name", "normal"), "denoise": getattr(metadata, "denoise", 1.0),
    }
    if model_ref:
        inputs["model"] = model_ref
    add_node("KSampler", inputs)
    return _json_value(graph)


def _output_path(source: Path, suffix: str, overwrite: bool) -> Path:
    if overwrite:
        return source

    suffix = suffix or "_civitai"
    candidate = source.with_name(f"{source.stem}{suffix}{source.suffix}")
    counter = 1
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}{suffix}_{counter}{source.suffix}")
        counter += 1
    return candidate


def embed_video_metadata(
    source: str | os.PathLike[str],
    tags: dict[str, str],
    suffix: str = "_civitai",
    overwrite: bool = False,
) -> str:
    """Copy a video while adding container metadata, preserving all streams."""
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {source_path}")

    extension = source_path.suffix.lower()
    if extension not in {".mp4", ".webm"}:
        raise ValueError(f"Only MP4 and WebM videos are supported, got '{source_path.suffix}'.")

    output_path = _output_path(source_path, suffix, overwrite)
    temporary_output: Path | None = None
    if overwrite:
        temporary_file = tempfile.NamedTemporaryFile(
            prefix=f"{source_path.stem}_",
            suffix=source_path.suffix,
            dir=source_path.parent,
            delete=False,
        )
        temporary_file.close()
        temporary_output = Path(temporary_file.name)
        remux_output = temporary_output
    else:
        remux_output = output_path

    try:
        options = {"movflags": "use_metadata_tags+faststart"} if extension == ".mp4" else {}
        with av.open(str(source_path), mode="r") as source_container:
            with av.open(str(remux_output), mode="w", options=options) as output_container:
                output_container.metadata.update(tags)
                streams = {}
                for stream in source_container.streams:
                    output_stream = output_container.add_stream_from_template(stream, opaque=True)
                    output_stream.metadata.update(stream.metadata)
                    output_stream.disposition = stream.disposition
                    if stream.codec_context is not None:
                        # The container is unchanged; retain tags such as HEVC's hvc1.
                        output_stream.codec_context.codec_tag = stream.codec_context.codec_tag
                    if stream.type == "video" and stream.average_rate is not None:
                        output_stream.codec_context.framerate = stream.average_rate
                    streams[stream.index] = output_stream
                for packet in source_container.demux():
                    if packet.size == 0:
                        continue
                    packet.stream = streams[packet.stream.index]
                    output_container.mux(packet)

        if overwrite and temporary_output is not None:
            os.replace(temporary_output, source_path)
        return str(output_path)
    except av.error.FFmpegError as error:
        raise RuntimeError(f"Could not write video metadata: {error}") from error
    finally:
        if temporary_output is not None and temporary_output.exists():
            temporary_output.unlink()


def unpack_vhs_filenames(value: Any) -> tuple[bool, list[str]]:
    """Accept the tuple returned by VideoHelperSuite's VHS_FILENAMES type."""
    if isinstance(value, tuple) and len(value) == 2:
        save_output, files = value
        if isinstance(files, (list, tuple)):
            return bool(save_output), [os.fspath(item) for item in files]
    if isinstance(value, (list, tuple)):
        return True, [os.fspath(item) for item in value]
    raise TypeError("Expected the VHS_FILENAMES value returned by VideoHelperSuite.")


def select_vhs_video_file(files: Iterable[str]) -> str:
    """Return VHS's most complete supported video output.

    VHS places its metadata PNG first, then the encoded video, and finally an
    audio-muxed video when audio is connected. Its contract defines the last
    entry as the most complete output, so search backward past image artifacts.
    """
    paths = [os.fspath(item) for item in files]
    for path in reversed(paths):
        if Path(path).suffix.lower() in {".mp4", ".webm"}:
            return path

    produced = ", ".join(
        sorted({Path(path).suffix.lower() or "(no extension)" for path in paths})
    )
    detail = f" VHS produced: {produced}." if produced else ""
    raise ValueError(
        "Image Saver Video Metadata found no MP4 or WebM in the VHS output. "
        "Choose a video/* MP4 or WebM format in Video Combine; image/gif and "
        f"image/webp cannot carry this metadata.{detail}"
    )


def pack_vhs_filenames(save_output: bool, files: Iterable[str]) -> tuple[bool, list[str]]:
    return bool(save_output), list(files)

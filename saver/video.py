"""Video metadata helpers for Civitai-compatible Image Saver output."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


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
    """Convert our FFmpeg-ready tags to values accepted by native VIDEO.

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

    for key, value in list(tags.items()):
        if len(value.encode("utf-8")) > MAX_METADATA_VALUE_BYTES:
            raise ValueError(f"Video metadata field '{key}' exceeds the 2 MiB safety limit.")

    return tags

def _positive_id(value: Any) -> int | None:
    """Accept a Civitai ID without coercing booleans or fractional numbers."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal() and int(value) > 0:
        return int(value)
    return None


def _resource_identity(resource: dict[str, Any]) -> tuple[str, str] | None:
    """Return an AIR and resource kind, including version-ID-only records.

    Civitai's graph parser resolves the version portion of an AIR. Zero is a
    placeholder for an unknown model ID, never a substitute version ID. No
    lookup or guessed real model ID is needed when the version is known.
    """
    air = resource.get("air")
    match = re.fullmatch(
        r"urn:air:[\w/-]+:([\w/-]+):civitai:\d+@([1-9]\d*)(?:\+\d+)?(?:\.[\w-]+)?",
        air,
        flags=re.IGNORECASE,
    ) if isinstance(air, str) else None
    version_id = _positive_id(resource.get("modelVersionId"))
    if match and (version_id is None or version_id == int(match[2])):
        return air, match[1].lower()
    if version_id is None:
        return None
    kind = resource.get("type")
    kind = kind.lower() if isinstance(kind, str) else "other"
    kind = {"textualinversion": "embedding", "locon": "lycoris"}.get(kind, kind)
    if not re.fullmatch(r"[a-z0-9_/-]+", kind):
        kind = "other"
    model_id = _positive_id(resource.get("modelId")) or 0
    return f"urn:air:other:{kind}:civitai:{model_id}@{version_id}", kind


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

    model_ref = None
    resolved_kinds = set()
    seen = set()
    for resource in resources if isinstance(resources, list) else []:
        if not isinstance(resource, dict):
            continue
        identity = _resource_identity(resource)
        if not identity or identity[0] in seen:
            continue
        air, kind = identity
        seen.add(air)
        is_model = kind in {"checkpoint", "unet", "diffusionmodel", "model"}
        resolved_kinds.add("model" if is_model else "lora" if kind in {"lora", "lycoris", "locon"} else kind)
        if is_model:
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
    modelname = getattr(metadata, "modelname", "")
    has_model = any("unet_name" in node["inputs"] or "ckpt_name" in node["inputs"] for node in graph.values())
    if not has_model and isinstance(modelname, str) and modelname and modelname.lower() not in {"unknown", "none"}:
        model_ref = add_node("UNETLoader", {"unet_name": modelname})

    # Hash entries are unresolved resources when lookups were attempted, or
    # the complete local list when downloads were disabled. Their LoRA names
    # remain useful even if no original graph was embedded.
    hashes = extract_json_after(getattr(metadata, "a111_params", ""), "Hashes:")
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
        inputs = {"lora_name": filename, "strength_model": 1.0}
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


def escape_ffmetadata(value: str) -> str:
    """Escape a value for FFmpeg's FFMETADATA1 format."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace("#", "\\#")
        .replace("=", "\\=")
        .replace("\n", "\\\n")
    )


def write_ffmetadata(path: str | os.PathLike[str], tags: dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as metadata_file:
        metadata_file.write(";FFMETADATA1\n")
        for key, value in tags.items():
            metadata_file.write(f"{escape_ffmetadata(key)}={escape_ffmetadata(value)}\n")


def find_ffmpeg() -> str:
    forced_path = os.environ.get("VHS_FORCE_FFMPEG_PATH") or os.environ.get("IMAGE_SAVER_FFMPEG_PATH")
    if forced_path and os.path.isfile(forced_path):
        return forced_path

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    try:
        from imageio_ffmpeg import get_ffmpeg_exe

        return get_ffmpeg_exe()
    except Exception as error:
        raise RuntimeError(
            "Image Saver Video Metadata requires FFmpeg. Install FFmpeg or set "
            "IMAGE_SAVER_FFMPEG_PATH to its executable."
        ) from error


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
        ffmpeg_output = temporary_output
    else:
        ffmpeg_output = output_path

    metadata_file = tempfile.NamedTemporaryFile(
        prefix="image_saver_video_metadata_",
        suffix=".txt",
        mode="w",
        encoding="utf-8",
        newline="\n",
        delete=False,
    )
    metadata_file.close()

    try:
        write_ffmetadata(metadata_file.name, tags)
        command = [
            find_ffmpeg(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-f",
            "ffmetadata",
            "-i",
            metadata_file.name,
            "-map",
            "0",
            "-map_metadata",
            "1",
            "-c",
            "copy",
        ]
        if extension == ".mp4":
            command.extend(["-movflags", "use_metadata_tags+faststart"])
        command.append(str(ffmpeg_output))
        subprocess.run(command, check=True, capture_output=True, text=True)

        if overwrite and temporary_output is not None:
            os.replace(temporary_output, source_path)
        return str(output_path)
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(f"FFmpeg could not write video metadata: {details}") from error
    finally:
        try:
            os.remove(metadata_file.name)
        except FileNotFoundError:
            pass
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

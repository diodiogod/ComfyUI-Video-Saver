"""
"Edit Metadata": change or remove parts of finished METADATA (e.g. from Load Image with Metadata or
Image Saver Metadata) without rebuilding it. Only the changed parts of the A1111 parameters are rewritten;
everything else, including hashes and Civitai resources, is kept as written. No hashing, no Civitai lookups.

Inputs follow Edit Image Saver Pipe: text fields use the [original] placeholder, numbers and
sampler/scheduler only apply when connected.
"""
import dataclasses
import asyncio
import re
import json
from pathlib import Path
from typing import Any

from .nodes import Metadata, parse_checkpoint_name_without_extension
from .nodes_pipe import EditImageSaverPipe, apply_string_edit
from .saver.image_metadata import edit_parameters, parse_parameters, read_image_metadata
from .saver.metadata_reader import MissingGenerationMetadata, read_image_tags, read_resource_records, read_video_tags
from .saver.metadata_resources import merge_resources, metadata_choices, remove_references, resource_references, resource_version, selection_values, settings_resources
from .saver.resource_refs import parse_reference, resource_from_identifier
from .saver.video import update_edited_generation_tags
from .utils_civitai import get_civitai_sampler_name
from .nodes_image_loader import metadata_path, to_saver_metadata, PromptServer

EDIT_FIELDS = ("modelname", "positive", "negative", "width", "height", "seed", "steps", "cfg",
               "sampler_name", "scheduler_name", "denoise", "clip_skip")


class EditMetadata:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        pipe_fields = EditImageSaverPipe.INPUT_TYPES()["optional"]
        optional: dict[str, Any] = {key: pipe_fields[key] for key in EDIT_FIELDS if key in pipe_fields}
        optional["modelname"] = (optional["modelname"][0], {**optional["modelname"][1],
            "tooltip": "model name shown in the parameters and used by %model in filenames. "
                       "Doesn't change hashes or Civitai resources (remove them with remove_settings if needed). "
                       "Supports the [original] placeholder; empty removes it."})
        for key in ("positive", "negative"):
            optional[key] = (optional[key][0], {**optional[key][1],
                "tooltip": f"{key} prompt. '[original]' keeps it, '[original], more' appends, other text replaces it, empty removes it."})
        optional["additional_hashes"] = ("STRING", {"default": "", "multiline": True, "forceInput": True,
            "tooltip": "Connect resource references from Expand Image Saver Metadata or a STRING node. "
                       "Add resources while preserving the original list. Comma-separated hashes, name:HASH:weight, "
                       "civitai:VERSION_ID:weight or AIR:weight. Matching hashes/versions merge; an explicit weight updates it. No network requests."})
        optional["remove_settings"] = ("STRING", {"default": "", "multiline": False,
            "tooltip": "Select existing settings in the advanced checklist below. API workflows can supply a JSON array or comma-separated setting names."})
        optional["remove_resources"] = ("STRING", {"default": "", "multiline": False,
            "tooltip": "Select resources in the checklist below. Removes their saved attribution, not LoRA loading or prompt text. "
                       "API workflows can supply a JSON array or comma-separated names, hashes, version IDs or AIRs."})
        return {"required": {"metadata": ("METADATA", {"tooltip": "metadata to edit"})}, "optional": optional}

    RETURN_TYPES = ("METADATA",)
    RETURN_NAMES = ("metadata",)
    OUTPUT_TOOLTIPS = ("the edited metadata, for Image Saver Simple or another Edit Metadata",)
    FUNCTION = "execute"
    CATEGORY = "ImageSaver"
    DESCRIPTION = ("Change or remove parts of existing metadata, e.g. from Load Image with Metadata. Only what you change "
                   "is rewritten; hashes, Civitai resources and other settings stay as they are. Numbers and "
                   "sampler/scheduler only change when connected.")

    def execute(self, metadata, **kwargs):
        result = self.edit(metadata, **kwargs)
        return {"result": result, "ui": {"metadata_editor": [{"source": metadata_choices(metadata),
                                                              "result": metadata_choices(result[0])}]}}

    def edit(self, metadata: Metadata | None, remove_settings: str = "", remove_resources: str = "", additional_hashes: str = "",
             **kwargs: Any) -> tuple[Metadata | None]:
        if metadata is None:
            print("ComfyUI-Image-Saver: Edit Metadata received no metadata, passing it through.")
            return (None,)

        fields: dict[str, Any] = {}
        changes: dict[str, str | None] = {}

        positive = apply_string_edit(kwargs.get("positive"), metadata.positive)
        negative = apply_string_edit(kwargs.get("negative"), metadata.negative)
        if positive != metadata.positive:
            fields["positive"] = positive
        if negative != metadata.negative:
            fields["negative"] = negative

        modelname = apply_string_edit(kwargs.get("modelname"), metadata.modelname)
        if modelname != metadata.modelname:
            fields["modelname"] = modelname
            changes["Model"] = parse_checkpoint_name_without_extension(modelname) if modelname else None

        def connected(key: str) -> Any:
            return kwargs.get(key)

        if (seed := connected("seed")) is not None:
            fields["seed"] = int(seed)
            changes["Seed"] = str(int(seed))
        if (steps := connected("steps")) is not None:
            fields["steps"] = int(steps)
            changes["Steps"] = str(int(steps))
        if (cfg := connected("cfg")) is not None:
            fields["cfg"] = float(cfg)
            changes["CFG scale"] = str(float(cfg))
        if (denoise := connected("denoise")) is not None:
            fields["denoise"] = float(denoise)
            # Like A1111: only written when it isn't a full denoise
            changes["Denoising strength"] = str(float(denoise)) if float(denoise) != 1.0 else None
        if (clip_skip := connected("clip_skip")) is not None:
            fields["clip_skip"] = int(clip_skip)
            changes["Clip skip"] = str(abs(int(clip_skip))) if int(clip_skip) != 0 else None
        if connected("width") is not None or connected("height") is not None:
            width = int(connected("width") if connected("width") is not None else metadata.width)
            height = int(connected("height") if connected("height") is not None else metadata.height)
            fields["width"], fields["height"] = width, height
            changes["Size"] = f"{width}x{height}"
        if connected("sampler_name") is not None or connected("scheduler_name") is not None:
            sampler = str(connected("sampler_name") if connected("sampler_name") is not None else metadata.sampler_name)
            scheduler = str(connected("scheduler_name") if connected("scheduler_name") is not None else metadata.scheduler_name)
            fields["sampler_name"], fields["scheduler_name"] = sampler, scheduler
            changes["Sampler"] = get_civitai_sampler_name(sampler.replace("_gpu", ""), scheduler)
            changes["Schedule type"] = None  # newer A1111 field; the scheduler is part of Sampler now

        remove = selection_values(remove_settings)
        resources = selection_values(remove_resources)
        if not fields and not remove and not resources and not additional_hashes.strip():
            return (metadata,)
        parameters = metadata.a111_params
        if metadata.source_metadata_tags:
            if not parameters:
                raise ValueError("This source has no saved parameter text. Connect its metadata directly to a saver to preserve it.")
            # Older videos may carry resource IDs only in the graph or in structured tags.
            # Apply the same removal and weight rules to those records before rebuilding the graph.
            saved_records = [record for record in read_resource_records(metadata.source_metadata_tags)
                             if resource_version(record)]
            if saved_records and saved_records != settings_resources(parse_parameters(parameters)[2]):
                parameters = edit_parameters(parameters, settings={"Civitai resources": json.dumps(saved_records)})
        final_hashes = drop_named_hashes(metadata.final_hashes, resources)
        if parameters:
            fields["a111_params"] = edit_parameters(parameters, fields.get("positive"), fields.get("negative"),
                                                    changes, remove, resources)
            final_hashes = drop_removed_hashes(final_hashes, parameters, fields["a111_params"])
            if remove or resources:
                # Keep version-only references synchronized with the edited text, too.
                _, _, old_settings = parse_parameters(parameters)
                old_records = old_settings.get("Civitai resources", [])
                old_versions = {resource_version(item) for item in old_records if isinstance(item, dict)} if isinstance(old_records, list) else set()
                current_refs = resource_references(fields["a111_params"])
                current_versions = {str(record["modelVersionId"]) for entry in current_refs.split(",")
                                    if (parsed := parse_reference(entry)) and (record := resource_from_identifier(parsed[2]))}
                final_hashes = ",".join(entry for entry in final_hashes.split(",") if entry and not (
                    (parsed := parse_reference(entry)) and (record := resource_from_identifier(parsed[2]))
                    and str(record["modelVersionId"]) in old_versions - current_versions))
        if additional_hashes.strip():
            parameters, final_hashes = merge_resources(fields.get("a111_params", metadata.a111_params), final_hashes, additional_hashes)
            fields["a111_params"] = parameters
        if final_hashes != metadata.final_hashes:
            fields["final_hashes"] = final_hashes

        result = dataclasses.replace(metadata, **fields)
        if metadata.source_metadata_tags and result != metadata:
            result.source_metadata_tags = update_edited_generation_tags(metadata, result, parameters, remove, resources)
        return (result,)


HASH_TOKEN = re.compile(r"\b[0-9a-fA-F]{8,16}\b")


def drop_named_hashes(final_hashes: str, names: list[str]) -> str:
    """Drop hashes output entries ("name:HASH[:weight]") whose name or hash is in `names`."""
    if not names:
        return final_hashes
    return remove_references(final_hashes, names)


def drop_removed_hashes(final_hashes: str, old_parameters: str, new_parameters: str) -> str:
    """Drop the hashes output entries ("name:HASH[:weight]") whose hash was in the parameters before the edit but
    isn't anymore, e.g. after removing "Hashes" (the model stays if "Model hash" is kept). Entries that never
    appeared in the parameters (e.g. resources resolved via Civitai) are kept."""
    old = {token.lower() for token in HASH_TOKEN.findall(old_parameters)}
    new = {token.lower() for token in HASH_TOKEN.findall(new_parameters)}
    gone = old - new
    if not gone:
        return final_hashes
    kept = [entry for entry in final_hashes.split(",")
            if entry.strip() and not any(part.strip().lower() in gone for part in entry.split(":"))]
    return ",".join(kept)


def edit_preview_response(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("filename"), str) or not payload["filename"]:
        return 400, {"error": "Select an image or video in the connected metadata reader first."}
    edits = payload.get("edits", [])
    allowed = set(EDIT_FIELDS) | {"remove_resources", "remove_settings", "additional_hashes"}
    if not isinstance(edits, list) or len(edits) > 32 or any(not isinstance(edit, dict) or set(edit) - allowed for edit in edits):
        return 400, {"error": "Invalid metadata edits."}
    text_fields = {"modelname", "positive", "negative", "sampler_name", "scheduler_name",
                   "remove_resources", "remove_settings", "additional_hashes"}
    for edit in edits:
        for key, value in edit.items():
            if (key in text_fields and not isinstance(value, str)) or (
                    key not in text_fields and (isinstance(value, bool) or not isinstance(value, (int, float)))):
                return 400, {"error": f"Invalid value for {key}."}
    try:
        path = metadata_path(payload["filename"])
    except ValueError:
        return 403, {"error": "invalid path"}
    except FileNotFoundError:
        return 404, {"error": "file not found"}
    try:
        is_video = Path(path).suffix.lower() in {".mp4", ".webm"}
        if is_video or payload.get("reader") == "image":
            from .nodes_metadata_reader import _metadata_from_tags
            read_tags = read_video_tags if is_video else read_image_tags
            metadata = _metadata_from_tags(*read_tags(path))
        else:
            metadata = to_saver_metadata(read_image_metadata(path))
        source = metadata
        for changes in edits:
            source = metadata
            metadata = EditMetadata().edit(metadata, **changes)[0]
        warning = "" if metadata else "No generation metadata was found in this image."
        return 200, {"source": metadata_choices(source), "result": metadata_choices(metadata), "warning": warning}
    except MissingGenerationMetadata as error:
        return 200, {"source": metadata_choices(None), "result": metadata_choices(None), "warning": str(error)}
    except (ValueError, TypeError, OSError) as error:
        return 400, {"error": str(error)}


if PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    from aiohttp import web

    @PromptServer.instance.routes.post("/image_saver/edit_metadata_preview")
    async def edit_metadata_preview_route(request):
        payload = await request.json()
        status, body = await asyncio.to_thread(edit_preview_response, payload)
        return web.json_response(body, status=status)

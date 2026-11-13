"""Media metadata readers and the shared Image Saver metadata expander."""

from pathlib import Path
from inspect import signature

import folder_paths
import comfy.samplers
from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker

from .nodes import Metadata
from .saver.metadata_reader import MissingGenerationMetadata, metadata_fields, read_image_tags, read_resource_records, read_video_tags
from .saver.resource_refs import export_resource_references
from .utils_civitai import CIVITAI_SAMPLER_MAP


USE_VIDEO_INPUT = "(use connected VIDEO input)"


def _reader_ui_options():
    # Older V3 schemas do not expose the cached intermediate-UI flag.
    return {"has_intermediate_output": True} if "has_intermediate_output" in signature(io.Schema).parameters else {}


def _missing_metadata_output(error: MissingGenerationMetadata):
    return io.NodeOutput(ExecutionBlocker(None), ui={"metadata_warning": [str(error)]})


def _metadata_from_tags(tags, width, height):
    fields = metadata_fields(tags, width, height)
    for sampler, display in CIVITAI_SAMPLER_MAP.items():
        name = fields["sampler_name"]
        if name == display or name in (f"{display} Karras", f"{display} Exponential"):
            fields["sampler_name"] = sampler
            if name.endswith(" Karras"):
                fields["scheduler_name"] = "karras"
            elif name.endswith(" Exponential"):
                fields["scheduler_name"] = "exponential"
            break
    return Metadata(**fields)


class ImageSaverReadVideoMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = Path(folder_paths.get_input_directory())
        files = sorted(path.name for path in input_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in {".mp4", ".webm"})
        return io.Schema(
            node_id="Image Saver Read Video Metadata",
            display_name="Image Saver — Read Video Metadata",
            category="ImageSaver",
            description="Read original prompts, settings, hashes and resource IDs from an MP4/WebM. Connect metadata to either Image Saver video saver to preserve generation information through upscaling. Does not load video frames.",
            search_aliases=["video prompt reader", "preserve video metadata", "video metadata loader"],
            inputs=[
                io.Combo.Input("file", options=[USE_VIDEO_INPUT, *files], upload=io.UploadType.video,
                               tooltip="Upload/select the source MP4/WebM, or connect a native Load Video output below. A connected VIDEO takes priority over this file."),
                io.Video.Input("video", optional=True,
                               tooltip="Optional native VIDEO from Load Video, before upscaling or rebuilding frames. Reads the original file's tags without decoding frames."),
            ],
            outputs=[
                io.Custom("METADATA").Output("metadata", tooltip="Connect directly to metadata on Save Native Video or Add Metadata to VHS Video."),
            ],
            **_reader_ui_options(),
        )

    @classmethod
    def execute(cls, file=USE_VIDEO_INPUT, video=None):
        if video is not None:
            source = video.get_stream_source()
        else:
            if file == USE_VIDEO_INPUT:
                raise ValueError("Select a source MP4/WebM or connect the native Load Video output to video.")
            source = folder_paths.get_annotated_filepath(file)
            if Path(source).suffix.lower() not in {".mp4", ".webm"}:
                raise ValueError("Select an MP4 or WebM video.")
        try:
            tags, width, height = read_video_tags(source)
        except MissingGenerationMetadata as error:
            return _missing_metadata_output(error)
        return io.NodeOutput(_metadata_from_tags(tags, width, height))

    @classmethod
    def fingerprint_inputs(cls, file=USE_VIDEO_INPUT, video=None):
        source = video.get_stream_source() if video is not None else (
            folder_paths.get_annotated_filepath(file) if file != USE_VIDEO_INPUT else None
        )
        if isinstance(source, str):
            stat = Path(source).stat()
            return stat.st_mtime_ns, stat.st_size
        return float("nan")

    @classmethod
    def validate_inputs(cls, file):
        if file == USE_VIDEO_INPUT:
            return True
        if not folder_paths.exists_annotated_filepath(file):
            return "The selected source video does not exist."
        if Path(folder_paths.get_annotated_filepath(file)).suffix.lower() not in {".mp4", ".webm"}:
            return "Select an MP4 or WebM video."
        return True


class ImageSaverReadImageMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        input_dir = Path(folder_paths.get_input_directory())
        files = sorted(path.name for path in input_dir.iterdir()
                       if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        return io.Schema(
            node_id="Image Saver Read Image Metadata",
            display_name="Image Saver — Read Image Metadata",
            category="ImageSaver",
            description="Read prompts, settings and hashes from a PNG/JPEG/WebP file. Connect metadata to an Image Saver or Read Image Saver Metadata for individual fields. Reads the source file, not decoded IMAGE tensors.",
            search_aliases=["image prompt reader", "load image metadata"],
            inputs=[io.Combo.Input("file", options=files, upload=io.UploadType.image,
                                   tooltip="Select/upload the original image. Image tensors from Load Image do not contain file metadata.")],
            outputs=[io.Custom("METADATA").Output("metadata", tooltip="Reusable saver metadata; connect Read Image Saver Metadata to inspect individual fields.")],
            **_reader_ui_options(),
        )

    @classmethod
    def execute(cls, file):
        source = folder_paths.get_annotated_filepath(file)
        if Path(source).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            raise ValueError("Select a PNG, JPEG or WebP image.")
        try:
            tags, width, height = read_image_tags(source)
        except MissingGenerationMetadata as error:
            return _missing_metadata_output(error)
        return io.NodeOutput(_metadata_from_tags(tags, width, height))

    @classmethod
    def fingerprint_inputs(cls, file):
        stat = Path(folder_paths.get_annotated_filepath(file)).stat()
        return stat.st_mtime_ns, stat.st_size

    @classmethod
    def validate_inputs(cls, file):
        if not folder_paths.exists_annotated_filepath(file):
            return "The selected source image does not exist."
        if Path(folder_paths.get_annotated_filepath(file)).suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            return "Select a PNG, JPEG or WebP image."
        return True


class ReadImageSaverMetadata(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Read Image Saver Metadata",
            category="ImageSaver",
            description="Expand Image Saver METADATA into prompts and generation settings. hashes carries all saved resource references and weights in the format accepted by additional_hashes. No hashing or Civitai requests.",
            inputs=[io.Custom("METADATA").Input("metadata", tooltip="The complete metadata bundle from a media reader or Image Saver Metadata, including saved resource identities.")],
            outputs=[
                io.String.Output("modelname"),
                io.String.Output("positive"),
                io.String.Output("negative"),
                io.Int.Output("width", tooltip="Original generation width, which can differ from the loaded/upscaled video width."),
                io.Int.Output("height", tooltip="Original generation height."),
                io.Int.Output("seed"),
                io.Int.Output("steps"),
                io.Float.Output("cfg"),
                io.Combo.Output("sampler_name", options=comfy.samplers.KSampler.SAMPLERS),
                io.Combo.Output("scheduler_name", options=comfy.samplers.KSampler.SCHEDULERS),
                io.Float.Output("denoise"),
                io.Int.Output("clip_skip"),
                io.String.Output("hashes", tooltip="All available resource references and weights: literal hashes, Civitai model-version IDs or AIRs. Connect directly to additional_hashes, or join more entries with commas. Existing identities are carried without looking up missing hashes."),
                io.String.Output("a1111_params", tooltip="Full A1111 parameter text, including any embedded Hashes and Civitai resources sections."),
            ],
        )

    @classmethod
    def execute(cls, metadata):
        tags = dict(metadata.source_metadata_tags or {})
        if metadata.a111_params:
            tags["parameters"] = metadata.a111_params
        resources = read_resource_records(tags)
        return io.NodeOutput(
            metadata.modelname, metadata.positive, metadata.negative,
            metadata.width, metadata.height, metadata.seed, metadata.steps, metadata.cfg,
            metadata.sampler_name, metadata.scheduler_name, metadata.denoise, metadata.clip_skip,
            export_resource_references(resources, metadata.final_hashes), metadata.a111_params,
        )

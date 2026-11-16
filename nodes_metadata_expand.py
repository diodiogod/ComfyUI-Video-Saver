from .saver.metadata_resources import resource_references


class ExpandImageSaverMetadata:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"metadata": ("METADATA",)}}

    RETURN_NAMES = ("modelname", "positive", "negative", "width", "height", "seed", "steps", "cfg",
                    "sampler_name", "scheduler_name", "denoise", "clip_skip", "hashes", "a1111_params")
    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "INT", "INT", "INT", "FLOAT",
                    "STRING", "STRING", "FLOAT", "INT", "STRING", "STRING")
    OUTPUT_TOOLTIPS = tuple(f"Saved {name.replace('_', ' ')}." for name in RETURN_NAMES[:12]) + (
        "All saved resource references and weights: hashes, Civitai version IDs and AIRs. Connect to additional_hashes.",
        "Complete A1111 parameter text.",
    )
    FUNCTION = "expand"
    CATEGORY = "ImageSaver"
    DESCRIPTION = "Extract individual fields from METADATA. hashes includes saved hashes, Civitai version IDs/AIRs and weights, for additional_hashes. No file loading or Civitai requests."

    def expand(self, metadata):
        if metadata is None:
            raise ValueError("No generation metadata to expand. Select a source image with metadata.")
        values = [getattr(metadata, name) for name in self.RETURN_NAMES[:12]]
        return (*values, resource_references(metadata.a111_params, metadata.final_hashes), metadata.a111_params)

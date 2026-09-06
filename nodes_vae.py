ENCODE_AS_OPTIONS = ["Individual Images", "Video Frames", "Auto"]
ENCODE_AS_TOOLTIP = (
    "For 3D/video VAEs (e.g. Qwen Image, which reuses the Wan video VAE): 'Individual Images' encodes each image "
    "in the batch as its own single-frame sample, avoiding the batch-to-video truncation bug "
    "(https://github.com/Comfy-Org/ComfyUI/issues/14039). 'Video Frames' keeps the batch as a temporal sequence. "
    "'Auto' uses the VAE's own default. Has no effect on regular 2D image VAEs."
)


class VAEEncodeIndividualImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
                "encode_as": (ENCODE_AS_OPTIONS, {"default": "Individual Images", "tooltip": ENCODE_AS_TOOLTIP}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "ImageSaver/VAE"
    DESCRIPTION = "VAE Encode that can force 3D/video VAEs to treat a batch of images as separate images instead of video frames, without needing ComfyUI core changes."

    def encode(self, vae, pixels, encode_as):
        previous = getattr(vae, "not_video", False)
        if encode_as != "Auto":
            vae.not_video = encode_as == "Individual Images"
        try:
            samples = vae.encode(pixels)
        finally:
            vae.not_video = previous
        return ({"samples": samples},)

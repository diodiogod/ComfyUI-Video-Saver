class VAEEncodeIndividualImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "encode"
    CATEGORY = "ImageSaver/VAE"
    DESCRIPTION = (
        "VAE Encode that forces 3D/video VAEs (e.g. Qwen Image, which reuses the Wan video VAE) to encode each "
        "image in the batch as its own image instead of truncating them into video frames, working around "
        "https://github.com/Comfy-Org/ComfyUI/issues/14039 without needing ComfyUI core changes. "
        "Has no effect on regular 2D image VAEs."
    )

    def encode(self, vae, pixels):
        previous = getattr(vae, "not_video", False)
        vae.not_video = True
        try:
            samples = vae.encode(pixels)
        finally:
            vae.not_video = previous
        return ({"samples": samples},)

import copy


class VAEForceIndividualImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "patch"
    CATEGORY = "ImageSaver/VAE"
    DESCRIPTION = (
        "Returns a copy of the VAE with 3D/video encoding forced off, so any node that encodes with it "
        "(VAE Encode, Ultimate SD Upscale, etc.) treats each image in a batch as its own image instead of "
        "truncating them into video frames. Fixes batch encoding with Qwen Image VAE (which reuses the Wan "
        "video VAE architecture, https://github.com/Comfy-Org/ComfyUI/issues/14039) without needing ComfyUI "
        "core changes. Has no effect on regular 2D image VAEs. The input VAE is left untouched."
    )

    def patch(self, vae):
        patched_vae = copy.copy(vae)
        patched_vae.not_video = True
        return (patched_vae,)

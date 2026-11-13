# Example workflows

## Video Saver example

[`video-saver-example.json`](./video-saver-example.json) demonstrates both supported video paths:

- saving a native ComfyUI `VIDEO` value with `Image Saver — Save Native Video (VIDEO + Audio)`;
- adding metadata to a VideoHelperSuite output with `Image Saver — Add Metadata to VHS Video`.

Drag the JSON onto the ComfyUI canvas, then adapt the generation nodes, models, prompts, and output settings to your installation.

## Video Upscaling — Preserve or Edit Metadata

[`video-upscale-preserve-or-edit-metadata.json`](./video-upscale-preserve-or-edit-metadata.json) loads a native video, reads its generation metadata, resizes the frames by 2x, and saves the video with its original audio and frame rate. Select your own source video in Load Video. The workflow uses this pack and ComfyUI's built-in nodes.

- By default, the saver receives the original `METADATA`, preserving the source prompt, settings, resources, and generation dimensions.
- To write modified metadata, connect the **Modified metadata (optional)** reroute to the saver's `metadata` input. The example carries source resources through `hashes` → `additional_hashes`, supplies a new prompt/settings, and reads the processed frames' dimensions. These controls describe the metadata; replace the resize section with your own workflow to perform an AI upscale pass.
- The unconnected image metadata reader shows the equivalent entry point for PNG/JPEG/WebP source files.

Hi @alexopus,

I tried your `load-metadata` branch and built an image-only prototype directly on top of it. After using your Edit Metadata node, I agree with keeping finished metadata and editing only what changes. The straightforward workflow can stay:

**Load Image with Metadata → Edit Metadata → Image Saver Simple**

I kept your image/mask loading, existing editing controls and the same `Metadata` class. The additions are mostly about making resource editing easier to use:

- **Choose what to remove:** resources appear in a checklist with their names, versions, identifiers and weights. Settings have a separate advanced checklist showing their current values. This avoids having to remember the exact names to type into the removal fields. The underlying inputs still support text selectors and JSON arrays for API workflows.
- **Add resources while preserving existing ones:** `additional_hashes` accepts hashes, Civitai version IDs and AIRs, with optional weights. Matching known identities merge, and an explicitly supplied weight updates the matching resource. The editor does this without rehashing model files or making Civitai lookups. It cannot infer that an unrelated hash and version ID identify the same resource.
- **Connect resource references:** I made `additional_hashes` a connection input by default. It can receive the complete reference string from an **Expand Image Saver Metadata** node, or manually entered text from a STRING node. I also extended the metadata builder's parser to accept those references. Keeping hashes, IDs/AIRs and weights together avoids having to concatenate separate outputs before reconnecting them.
- **Keep Expand optional:** the expander exposes prompts, generation settings and resource references when those values are needed elsewhere in a workflow. Ordinary edits can use your editor directly, without expanding and rebuilding everything.
- **Preview edits live:** the editor uses the same Python editing function for preview and execution. It follows chains of editors back to the image loader and applies their edits in order. Computed inputs require a run and are identified as such. Disconnecting metadata clears the preview and cached result; reconnecting a supported source restores it. Late results cannot refill a disconnected editor.
- **Keep the image still:** Table and Text share a fixed metadata panel height, so changing views no longer resizes or moves the loaded image. Longer content scrolls inside that panel.

The [branch, screenshots and example workflow](https://github.com/diodiogod/ComfyUI-Video-Saver/tree/feat/image-metadata-editor/docs/metadata-editor) are available for review. The example demonstrates metadata editing and resaving an image. I tested adding/removing resources, saved-file metadata, reload persistence, chained editors and disconnect/reconnect behavior, including in a running ComfyUI instance.

![Resource selection and live result preview](https://raw.githubusercontent.com/diodiogod/ComfyUI-Video-Saver/feat/image-metadata-editor/docs/metadata-editor/resource-selection.jpg)

One behavior I left unchanged raises a design question: when a file contains a ComfyUI workflow but no A1111 parameter text, the loader can recover some values for display, but the metadata output still has an empty A1111 parameter string. Passing it to a saver therefore does not write generation parameters recovered from that workflow.

Before preparing a PR, I'd appreciate your thoughts on:

1. Does this checklist-based resource/settings UI fit your direction? Would you keep `additional_hashes` as a connection input by default, or prefer an inline text widget?
2. For workflow-only metadata, would you keep the current display-only fallback, or offer an explicit option to turn recovered values into saver parameters? I haven't implemented conversion because the recovery is best effort.
3. Would you prefer one PR against `load-metadata`, or smaller contributions for resource editing, the expander, and preview/layout changes?

This contribution stays focused on images. I'll continue using your implementation as the base for video support in my fork and handling the upstream merges there. I'm happy to adapt these changes to the direction you prefer.

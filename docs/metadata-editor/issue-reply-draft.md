Hi @alexopus,

I tried your `load-metadata` branch and built an image-only prototype on top of it. Your loader and preview are the base; I added a few ideas around editing and resource reuse:

- `additional_hashes` merges extra resource references into the existing metadata. It accepts hashes, Civitai version IDs/AIRs and weights, without hashing model files or making Civitai requests. An explicitly supplied weight updates a matching resource.
- Instead of typing names for removal, the editor presents checklists populated from the source metadata. Resources show their names, versions, identifiers and weights; settings have a separate advanced checklist with their current values.
- A result preview uses the same editing code as execution. It works before execution for sources traceable to your image loader, including editor chains. Computed inputs need a workflow run, which the UI indicates.
- An **Expand Image Saver Metadata** node exposes individual settings and the complete resource-reference string for wiring into other nodes.

The existing text inputs still support API workflows. The Metadata class is unchanged, and the branch has no video code or dependencies.

The [prototype and example](https://github.com/diodiogod/ComfyUI-Video-Saver/tree/feat/image-metadata-editor/docs/metadata-editor) include screenshots from a running ComfyUI instance. I tested removal/addition, saved-image metadata, and restoring selections after reloading the workflow.

Would this selection-based approach fit the direction you have in mind? I'm happy to adjust it to your preferences and prepare a focused PR if useful.

# ComfyUI Video Saver

Save videos with generation metadata in ComfyUI, while retaining the complete inherited Image Saver toolset.

> [!TIP]
> **Upload compatible videos directly to Civitai—no browser extension required.** Since v1.27.0, both the native ComfyUI VIDEO saver and the VHS metadata node write metadata that Civitai can import automatically. The [browser extension](https://github.com/diodiogod/Civitai-Video-Metadata-Assistant) remains available as a fallback for older videos or metadata Civitai misses.

This is a maintained fork of [alexopus/ComfyUI-Image-Saver](https://github.com/alexopus/ComfyUI-Image-Saver), extended with native ComfyUI and VideoHelperSuite video support. It remains a GitHub fork so upstream changes and attribution stay visible. Existing Image Saver node names are intentionally preserved for workflow compatibility.

The original project was itself forked from [giriss/comfy-image-saver](https://github.com/giriss/comfy-image-saver).

For images, the nodes save **generation metadata** compatible with Civitai geninfo detection. PNG, JPG, and WebP are supported. PNG stores the full ComfyUI workflow plus A1111-style parameters; JPEG/WebP store the A1111-style parameters. Model, LoRA, and embedding hashes are included for resource linking on Civitai.

You can find image examples in [`examples`](./examples) and the native/VHS video template in [`example_workflows`](./example_workflows).
<img width="1288" height="1039" alt="workflow" src="https://github.com/user-attachments/assets/dbbb9f67-afa3-48a2-8cd3-e4116393f8e0" />

You can also add LoRAs to the prompt in \<lora:name:weight\> format, which would be translated into hashes and stored together with the metadata. For this it is recommended to use `ImpactWildcardEncode` from the fantastic [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack). It will allow you to convert the LoRAs directly to proper conditioning without having to worry about avoiding/concatenating lora strings, which have no effect in standard conditioning nodes. Here is an example:
![workflow](https://github.com/user-attachments/assets/61440fac-f1d5-414b-ae69-dbdda9d6d442)

## Video metadata

The video nodes preserve the same positive/negative prompts, settings, model and LoRA hashes, and Civitai resource list used by the image saver. Civitai now reads ComfyUI metadata natively from MP4 and WebM uploads ([Civitai PR #5147](https://github.com/civitai/civitai/pull/5147)). Videos saved with this version can use that native upload flow.

To use native Civitai detection:

1. Update this node pack and restart ComfyUI.
2. Connect `Image Saver Metadata` to the native VIDEO saver or the VHS metadata node. Keep `download_civitai_data` enabled for resolved resource IDs.
3. Save a new MP4 or WebM. For VHS, use the enriched output from `Image Saver — Add Metadata to VHS Video`—the `_civitai` copy by default.
4. Disable the browser extension and upload through Civitai's own media-upload form. Review the detected prompt, settings, and resources before publishing.

For native ComfyUI video workflows, use `Image Saver — Save Native Video (VIDEO + Audio)`. Connect the native `VIDEO` output from `Create Video` or another native video node to `video`, and connect `Image Saver Metadata` to `metadata`. This is the actual saver: it encodes the native frames plus optional audio, then adds metadata. Its filename prefix supports Image Saver placeholders such as `%time`, `%basemodelname`, and `%seed`.

For VideoHelperSuite, keep `save_metadata` enabled on `Video Combine`, then connect its `VHS_FILENAMES` output to `Image Saver — Add Metadata to VHS Video` and connect `Image Saver Metadata` to `metadata`. It enriches the files already written by VHS and creates a suffixed MP4/WebM copy by default, preserving the original and all existing video/audio streams. Upload that enriched copy to Civitai for native metadata detection.

Both paths write a compact ComfyUI compatibility graph to `prompt`, built from the explicit Image Saver Metadata inputs. Civitai reads this graph to recover positive/negative prompts, seed, steps, sampler, scheduler, guidance, dimensions, and resolved resource versions. This graph describes the saved metadata; the original UI workflow remains the workflow to load in ComfyUI.

Keep `download_civitai_data` enabled on Image Saver Metadata to include resolved model-version IDs for native resource attribution. Existing AIR identifiers and version-ID-only records are supported. When lookup data is unavailable or disabled, recognizable model and LoRA filenames are retained, but native resource linking is not guaranteed from filenames alone.

With workflow embedding enabled, the original UI workflow stays in `workflow` and the unmodified execution graph is retained in `prompt_original`. With it disabled, the video still contains the small compatibility graph, but not the original workflow/execution graph. A1111 `parameters`, hashes, `civitaiResources`, and `extraMetadata` are retained for other readers.

The companion [Civitai Video Metadata Assistant](https://github.com/diodiogod/Civitai-Video-Metadata-Assistant) is an optional fallback for older videos, hash-based resource recovery, or metadata Civitai misses. Try the native upload with the extension disabled first; a browser extension is no longer required for the supported native path. Existing videos are not rewritten by updating this node pack.

### Preserve metadata through upscaling or another workflow

Try the [Video Upscaling — Preserve or Edit Metadata example](./example_workflows/video-upscale-preserve-or-edit-metadata.json). It preserves original metadata by default and includes an optional branch for new prompts/settings and carried resource references. See the [example instructions](./example_workflows/README.md) before running it.

Use **Image Saver — Read Video Metadata** to recover the source video's metadata. Upload/select an MP4 or WebM, or connect the original native `Load Video` output to its optional `video` input. Connect the reader's `metadata` output directly to `metadata` on **Save Native Video** or **Add Metadata to VHS Video**; process the frames/audio with your usual nodes. For VHS, select the same source file in the reader and your VHS loader.

For images, use **Image Saver — Read Image Metadata** with a PNG, JPEG, or WebP file. It reads PNG text metadata and JPEG/WebP EXIF, including Image Saver's parameters and embedded ComfyUI graphs. Select the original file: the decoded `IMAGE` output of Load Image does not carry file metadata. Its `metadata` output connects to the image/video savers or the shared expander.

The reader does not decode frames, hash model files, or contact Civitai. It preserves the source generation tags, including prompts, settings, hashes, resource versions, and weights. You do not need the original models installed. The output video's resolution can change while its metadata continues to describe the original generation.

Both media readers output only `METADATA`. To inspect or reuse individual values, connect it to **Read Image Saver Metadata**. It exposes `modelname`, positive/negative prompts, width, height, seed, steps, CFG, sampler, scheduler, denoise, clip skip, hashes, and the original A1111 parameter text. Sampler and scheduler outputs connect directly to compatible sampler inputs. This expander also accepts the existing **Image Saver Metadata** output; **Read Image Saver Pipe** remains the reader for pipe configurations.

The **`hashes`** output carries **all available resource references and weights** in one comma-separated string: literal hashes, Civitai model-version IDs, and AIRs. The metadata creator and savers use the same format. The historical socket name is retained, but a resource stored as an ID/AIR does not need to be converted to a hash. Only information embedded in the source file can be recovered; reading metadata makes no Civitai requests.

To keep the source resources while writing a **new upscale prompt and settings**, connect the expander's **`hashes`** output to **`additional_hashes`** on **Image Saver Metadata**. Supply the new prompt/settings there, then connect its `metadata` output to the native VIDEO saver, VHS metadata saver, or Image Saver Simple. The all-in-one Image Saver also accepts the string directly. To add more resources manually, join the exported text and your extra entries with a comma.

The input accepts a mixture of these formats:

| Reference | Example |
| --- | --- |
| Hash, optionally named and weighted | `ExtraLoRA:ABCDEF1234:0.8` |
| Civitai **model-version ID**, optionally weighted | `civitai:123456:0.65` |
| AIR, optionally weighted | `urn:air:sdxl:lora:civitai:12@123456:0.65` |

Export preserves embedded IDs/AIRs and available weights without requesting hashes from Civitai. When a resource has a saved type but no AIR, the exporter can encode that type in an AIR using model ID `0` as an unknown-model placeholder; the real version ID is retained. Direct IDs/AIRs and their weights are saved even with `download_civitai_data` disabled. Keep that setting enabled to resolve additional hashes into resource IDs/weights. Matching hashes and matching versions are included once, with the current workflow's detected resource taking precedence over a carried reference; repeated manual entries keep the first occurrence. Without a successful lookup, an unrelated hash and an ID cannot always be recognized as the same resource. The existing 30-manual-entry limit applies.

Weighted hashes that cannot be resolved are retained in a `Resource references` parameter field, so their weights can be recovered on the next read even without a Civitai lookup.

This path carries resource attribution into the new metadata. Use the original complete `METADATA` connection instead when you want to preserve the source generation metadata unchanged, including its prompt and workflow history.

With `embed_workflow` enabled on the saver, the current processing workflow is saved in `workflow`, the original generation workflow is retained in `workflow_original`, and the processing execution graph is stored separately in `prompt_postprocess`. The source `prompt`, `prompt_original`, parameters, and resource records remain intact. Disabling workflow embedding omits the workflow history while retaining generation metadata.

Individual field extraction supports A1111 parameters and standard ComfyUI sampler/conditioning graphs, including this pack's Civitai compatibility graph. Unsupported custom graphs are still preserved, but individual values may be unavailable and use the saver defaults. Metadata stripped from a source file cannot be recovered.

If a valid file contains no generation metadata, the reader shows a warning popup and blocks its dependent branch without a traceback or invented metadata. Reload ComfyUI's page after updating to load the notification extension. Corrupt files and invalid metadata still report normal errors.

This would have civitai autodetect all of the resources (assuming the model/lora/embedding hashes match):
![image](https://github.com/alexopus/ComfyUI-Image-Saver/assets/25933468/f0642389-4f34-4a64-89a6-5cf9c33d5ed1)

## How to install?

### Method 1: Manager (Recommended)
Install this fork from its Git URL in *ComfyUI-Manager*:

`https://github.com/diodiogod/ComfyUI-Video-Saver`

Searching for “ComfyUI Image Saver” may install the upstream image-only project instead.

### Method 2: Easy
If you don't have *ComfyUI-Manager*, then:
- Using CLI, go to the ComfyUI folder
- `cd custom_nodes`
- `git clone https://github.com/diodiogod/ComfyUI-Video-Saver.git`
- `cd ComfyUI-Video-Saver`
- `pip install -r requirements.txt`
- Start/restart ComfyUI

## Customization of file/folder names

You can use following placeholders:

- `%date`
- `%time` *– format taken from `time_format`*
- `%time_format<format>` *– custom datetime format using Python strftime codes*
- `%model` *– full name of model file*
- `%basemodelname` *– name of model (without file extension)*
- `%seed`
- `%counter`
- `%counter<padding>` *– zero-padded counter (e.g. `%counter<03>` with counter 1 becomes `001`)*
- `%sampler_name`
- `%scheduler`
- `%steps`
- `%cfg`
- `%denoise`

Example:

| `filename` value | Result file name |
| --- | --- |
| `%time-%basemodelname-%cfg-%steps-%sampler_name-%scheduler-%seed` | `2023-11-16-131331-Anything-v4.5-pruned-mergedVae-7.0-25-dpm_2-normal-1_01.png` |
| `%time_format<%Y%m%d_%H%M%S>-%seed` | `20231116_131331-1.png` |
| `%time_format<%B %d, %Y> %basemodelname` | `November 16, 2023 Anything-v4.5.png` |
| `img_%time_format<%Y-%m-%d>_%seed` | `img_2023-11-16_1.png` |

**Common strftime format codes for `%time_format<format>`:**

| Code | Meaning | Example |
|------|---------|---------|
| `%Y` | Year (4-digit) | 2023 |
| `%y` | Year (2-digit) | 23 |
| `%m` | Month (01-12) | 11 |
| `%B` | Month name (full) | November |
| `%b` | Month name (short) | Nov |
| `%d` | Day (01-31) | 16 |
| `%H` | Hour 24h | 13 |
| `%I` | Hour 12h | 01 |
| `%M` | Minute | 13 |
| `%S` | Second | 31 |
| `%p` | AM/PM | PM |
| `%A` | Weekday (full) | Thursday |
| `%a` | Weekday (short) | Thu |
| `%F` | YYYY-MM-DD | 2023-11-16 |
| `%T` | HH:MM:SS | 13:13:31 |

## Optional: Pipe Integration (Advanced Orchestration)

For orchestrating complicated workflows—such as projects involving dozens of character scenarios each with multiple scenes—the standard wiring can become dense. To help manage this, an optional **Pipe** system is available.

The pipe system bundles all metadata and image saver settings into a single connection, allowing you to pass them through a workflow and branch them efficiently.

### When Should I Need Pipe?

In most cases, the standard `ImageSaverMetadata`, `ImageSaverSimple`, or `ImageSaver` nodes are recommended, as the core features are identical.

The Pipe system is primarily beneficial when your workflow becomes crowded with wires between KSamplers and metadata nodes. It acts as an orchestrator, helping you maintain a cleaner generation pipeline in complex, multi-stage environments.

![Image Saver Pipe Workflow](images/image_saver_pipe.png)

#### Execution Order Observations (For Large Workflows)
Depending on your wiring strategy, ComfyUI may execute multiple KSamplers across the graph before reaching the corresponding `ImageSaverFromPipe` nodes. To optimize this, ensure that the branch containing an `ImageSaverFromPipe` node does not depend on resources from other active branches. This helps ComfyUI complete each branch sequentially, preventing unnecessary memory buildup.

For finer control in extremely large workflows, you can also use custom nodes that provide dedicated execution control.

### Features
* **Simplified Wiring:** Carries generation and saver settings in one pipe connection, useful for large-scale orchestration.
* **Deferred Execution:** Expensive operations like checkpoint hashing and Civitai API lookups are deferred until the final save operation.
* **Non-destructive Editing:** The `Edit Image Saver Pipe` node allows you to branch and modify settings. All string fields support the `[original]` placeholder (e.g., `[original], masterpiece`) to easily append or prepend text.
* **Orchestration Support:** Use the `Read Image Saver Pipe` to extract values for external manipulation before feeding them back into an `Edit` node.
* **Standard Dropdown Menus:** Sampler and scheduler inputs use combo boxes, allowing for easier direct connection to other nodes.

### Included Nodes
* **Make Image Saver Simple Config**: Standalone configuration for filename, path, and file type.
* **Make Image Saver Metadata Config**: Standalone configuration for prompts and generation parameters.
* **Make Image Saver Pipe**: Bundles the above configurations into a single pipe connection.
* **Edit Image Saver Pipe**: Overrides settings in an existing pipe (creates a new branch).
* **Read Image Saver Pipe**: Extracts settings from a pipe for inspection or external use.
* **Image Saver (From Pipe)**: Unpacks the pipe settings and saves the image.

### Workflow Consistency

The pipe system uses the same underlying logic as the standard nodes but is designed for a different workflow style.

* **Compatibility:** Standard nodes and pipe nodes have different input/output structures and are not directly interchangeable.
* **Execution Timing:** It is recommended to use one system consistently within a specific workflow branch. Standard nodes trigger side-effects (like metadata lookups) immediately upon execution. In contrast, pipe nodes defer these actions until the final saving step, allowing you to update metadata properties at multiple stages of your workflow before they are committed to disk.


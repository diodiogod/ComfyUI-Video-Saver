# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI Video Saver is a maintained fork of ComfyUI-Image-Saver that adds video generation metadata support while retaining the complete inherited image toolset. It stores model, LoRA, and embedding hashes for Civitai resource recognition. Existing Image Saver node identifiers remain unchanged for workflow compatibility.

## Development Commands

### Testing
```bash
cd saver && python -m pytest
cd tests && python -m pytest
```
See "Testing" below for what each suite covers.

## Key Features

- **Label vs Custom**: `custom` is embedded in the A1111 metadata string (and also usable via `%custom` in filenames); `label` (Image Saver / Image Saver Simple) is filename/path-only via `%label` and independent of metadata, meant for values shared across multiple Saver nodes in a workflow (e.g. one timestamp fed into several savers)

## Testing

Tests are split into two independent suites, each with its own `pytest.ini`, because `saver/saver.py` has no ComfyUI dependency while `nodes.py`/`utils.py` do:

- `saver/test_saver.py` — image format/metadata writing (`save_image`), no ComfyUI dependency. Run with `cd saver && python -m pytest`.
- `tests/test_paths.py` — output-path/filename containment (`resolve_within_output`, `ImageSaver.save_images`). `tests/conftest.py` stubs ComfyUI's runtime modules to import `nodes.py`/`utils.py` outside ComfyUI. Run with `cd tests && python -m pytest`.
- `tests/compat/` — development-only Node.js tests against Civitai's pinned generation-metadata parser using real MP4/WebM fixtures with audio. See its README for Python/FFmpeg configuration and the optional native container-reader check. These packages are not Python runtime dependencies.

Video metadata changes must keep both the compact compatibility graph and the original workflow/execution graph when embedding is enabled. Acceptance requires parser output assertions (prompts, settings, resource IDs/weights and offline filenames), not only assertions that JSON tags were written. Resource `air` and `type` fields are optional; `modelVersionId` is sufficient identity.

For metadata readers, test both a populated file and a valid file with no metadata. Expected absence must produce an actionable UI warning and silently block dependent nodes, rather than raise an execution error or return invented metadata. Corrupt files remain errors. Check that each reader exposes only METADATA and use Read Image Saver Metadata for individual fields. Frontend notification tests run with `node --test tests/metadata-reader-ui.test.cjs`. With ComfyUI's Python, `python tests/runtime_metadata_readers.py` checks warning delivery, cached UI replay where supported, and downstream blocking in its real executor without opening a server port. Use `--comfy-root <checkout>` to check each supported ComfyUI version, including GET_NODE_INFO_V1 serialization. New optional V3 schema fields must be checked against the installed constructor before passing them; test both strict older and current schema signatures.

## Important Notes

- WebP workflow embedding is experimental
- Resource paths are resolved through ComfyUI's folder_paths system
- Civitai integration can be disabled via `download_civitai_data` parameter

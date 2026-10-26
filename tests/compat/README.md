# Civitai video compatibility tests

These development-only tests write real MP4 and WebM files with audio, then parse their `prompt` and `workflow` tags using the same `@civitai/generation-metadata` version and Civitai plugin as the site's video upload path. The Python saver has no new runtime dependencies.

From this directory, install the pinned development package with `npm ci`, set `VIDEO_SAVER_PYTHON` to the ComfyUI Python executable if needed, and run `npm test`. FFmpeg, FFprobe and Node.js 20+ are required.

For an additional check of the actual native container reader, use Node.js 24+ and set `VIDEO_SAVER_NATIVE_READER` to `src/utils/metadata/video-tags.ts` from a Civitai checkout. The release validation used Civitai commit `2e60deea2e12c04b8477cfec24c7a7538e16ea42`. Without that option, FFprobe reads the container and the tests pass only the two tags supported by that reader to the actual generation parser.

Coverage includes AIR and ID-only resources, missing resource types, embeddings and auxiliary resources, invalid AIR fallback, disabled Civitai lookups, disabled workflow embedding, original graph preservation, prompts/settings, and preserved video/audio codecs. These checks do not submit files to Civitai or test its server resource-linking service.

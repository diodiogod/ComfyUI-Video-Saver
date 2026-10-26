const test = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const { applyPlugins, createParserContext, defaultParsers, generationMetadataSchema } = require('@civitai/generation-metadata');
const { civitai } = require('@civitai/generation-metadata/civitai');

const fixtures = JSON.parse(execFileSync(process.env.VIDEO_SAVER_PYTHON || 'python', ['export_fixtures.py'], {
  cwd: __dirname, encoding: 'utf8', maxBuffer: 8 * 1024 * 1024
}));

function parse(tags) {
  const { parsers, context } = applyPlugins([civitai()], defaultParsers);
  const ctx = createParserContext(context);
  for (const parser of parsers) {
    const state = parser.detect(tags, ctx);
    if (state) return generationMetadataSchema.parse(parser.parse(state, ctx));
  }
  throw new Error('Civitai did not detect the compatibility graph');
}

for (const fixture of fixtures) {
  test(`${fixture.container}: Civitai parses ${fixture.case} without the browser extension`, async () => {
    const allTags = Object.fromEntries(Object.entries(fixture.probe.format.tags).map(([key, value]) => [key.toLowerCase(), value]));
    // Match the deployed reader's allowlist. Optionally run Civitai's actual
    // pinned video-tags.ts as well, rather than relying on FFprobe extraction.
    let tags = { prompt: allTags.prompt, workflow: allTags.workflow };
    if (process.env.VIDEO_SAVER_NATIVE_READER) {
      const { readVideoTags } = await import(pathToFileURL(process.env.VIDEO_SAVER_NATIVE_READER).href);
      tags = await readVideoTags(new Blob([Buffer.from(fixture.bytes, 'base64')]));
    }
    const metadata = parse(tags);
    assert.equal(metadata.prompt, 'ABCD positive');
    assert.equal(metadata.negativePrompt, 'negative');
    assert.equal(metadata.seed, 42);
    assert.equal(metadata.steps, 20);
    assert.equal(metadata.cfgScale, 5.5);
    assert.equal(metadata.sampler, 'DPM++ 2M Karras');
    assert.equal(metadata.denoise, 0.75);
    assert.equal(metadata.width, 832);
    assert.equal(metadata.height, 480);
    const versions = metadata.civitaiResources || [];
    const ids = versions.map((resource) => resource.modelVersionId).sort((a, b) => a - b);
    const expected = { air: [101, 102], id_only: [101, 102], missing_type: [103], embedding: [104], auxiliary: [105], invalid_air: [106], offline: [], no_workflow: [101] };
    assert.deepEqual(ids, expected[fixture.case]);
    if (['air', 'id_only'].includes(fixture.case)) {
      assert.equal(versions.find((resource) => resource.modelVersionId === 102).weight, 0.8);
    }
    if (fixture.case === 'embedding') assert.equal(versions[0].type, 'embedding');
    if (fixture.case === 'offline') {
      assert.equal(metadata.Model, 'original');
      assert.equal(metadata.additionalResources[0].name, 'style.safetensors');
      assert.equal(metadata.additionalResources[0].strength, 0.6);
    }
    assert.match(allTags.parameters, /ABCDEF1234/);
    assert.match(allTags.parameters, /123456ABCD/);
    if (fixture.case !== 'no_workflow') {
      assert.equal(JSON.parse(allTags.prompt_original)['3'].class_type, 'CustomVideoSampler');
      assert.equal(JSON.parse(tags.workflow).extra.note, 'preserve me');
    } else {
      assert.equal(allTags.prompt_original, undefined);
      assert.deepEqual(JSON.parse(tags.workflow), {});
    }
    const streams = fixture.probe.streams;
    assert.equal(streams.find((stream) => stream.codec_type === 'video').codec_name, fixture.container === 'mp4' ? 'h264' : 'vp9');
    assert.equal(streams.find((stream) => stream.codec_type === 'audio').codec_name, fixture.container === 'mp4' ? 'aac' : 'opus');
  });
}

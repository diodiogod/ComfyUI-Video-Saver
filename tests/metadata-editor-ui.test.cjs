const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../js/metadata_editor.js'), 'utf8')
  .replace(/^import .*$/gm, '').replace(/^export /gm, '');
const context = { app: { registerExtension() {} }, api: {} };
vm.createContext(context);
vm.runInContext(source, context);

function graph(nodes, links) {
  const graph = { links, getNodeById(id) { return nodes.find(node => node.id === id); } };
  for (const node of nodes) node.graph = graph;
  return nodes;
}

test('resolves editor chains and reroutes in their execution order', () => {
  const nodes = graph([
    { id: 1, type: 'Load Image with Metadata (Image Saver)', widgets: [{ name: 'image', value: 'source.png' }] },
    { id: 2, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 1 }], widgets: [{ name: 'positive', value: 'first' }] },
    { id: 3, type: 'Reroute', inputs: [{ name: '', link: 2 }] },
    { id: 4, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 3 }], widgets: [{ name: 'additional_hashes', value: 'civitai:123' }] },
  ], { 1: { origin_id: 1 }, 2: { origin_id: 2 }, 3: { origin_id: 3 } });
  const result = context.previewRequest(nodes[3]);
  assert.deepEqual(JSON.parse(JSON.stringify(result)), { filename: 'source.png', edits: [{ positive: 'first' }, { additional_hashes: 'civitai:123' }] });
});

test('unknown computed inputs require execution instead of inventing a preview value', () => {
  const nodes = graph([
    { id: 1, type: 'Load Image with Metadata (Image Saver)', widgets: [{ name: 'image', value: 'source.png' }] },
    { id: 2, type: 'GetImageSize' },
    { id: 3, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 1 }, { name: 'width', link: 2 }], widgets: [] },
  ], { 1: { origin_id: 1 }, 2: { origin_id: 2 } });
  assert.throws(() => context.previewRequest(nodes[2]), /Run the workflow/);
});

test('metadata cycles are rejected', () => {
  const [node] = graph([{ id: 1, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 1 }], widgets: [] }], { 1: { origin_id: 1 } });
  assert.throws(() => context.previewRequest(node), /cycle/);
});

test('resource selection round-trips labels containing commas and quotes', () => {
  const values = ['Style, "soft"', 'civitai:102'];
  assert.deepEqual(Array.from(context.selectionValues(JSON.stringify(values))), values);
});

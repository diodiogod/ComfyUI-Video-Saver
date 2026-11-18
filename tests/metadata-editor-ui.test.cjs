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

test('video preview follows the connected native loader rather than the picker', () => {
  const nodes = graph([
    { id: 1, type: 'LoadVideo', widgets: [{ name: 'file', value: 'native.mp4' }] },
    { id: 2, type: 'Image Saver Read Video Metadata', inputs: [{ name: 'video', link: 1 }],
      widgets: [{ name: 'file', value: 'picker.webm' }] },
    { id: 3, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 2 }], widgets: [] },
  ], { 1: { origin_id: 1 }, 2: { origin_id: 2 } });
  assert.equal(context.previewRequest(nodes[2]).filename, 'native.mp4');
  nodes[0].type = 'CreateVideo';
  assert.throws(() => context.previewRequest(nodes[2]), /Run the workflow/);
  nodes[1].inputs[0].link = null;
  assert.equal(context.previewRequest(nodes[2]).filename, 'picker.webm');
});

test('additional resources can be previewed from a connected STRING node', () => {
  const nodes = graph([
    { id: 1, type: 'Load Image with Metadata (Image Saver)', widgets: [{ name: 'image', value: 'source.png' }] },
    { id: 2, type: 'PrimitiveStringMultiline', widgets: [{ name: 'value', value: 'civitai:103:0.8' }] },
    { id: 3, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 1 }, { name: 'additional_hashes', link: 2 }], widgets: [] },
  ], { 1: { origin_id: 1 }, 2: { origin_id: 2 } });
  assert.deepEqual(JSON.parse(JSON.stringify(context.previewRequest(nodes[2]))), {
    filename: 'source.png', edits: [{ additional_hashes: 'civitai:103:0.8' }],
  });
});

test('metadata cycles are rejected', () => {
  const [node] = graph([{ id: 1, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 1 }], widgets: [] }], { 1: { origin_id: 1 } });
  assert.throws(() => context.previewRequest(node), /cycle/);
});

test('resource selection round-trips labels containing commas and quotes', () => {
  const values = ['Style, "soft"', 'civitai:102'];
  assert.deepEqual(Array.from(context.selectionValues(JSON.stringify(values))), values);
});

test('old widget workflows keep additions and both removal selections when loaded', () => {
  const workflow = {
    nodes: [{ id: 2, type: 'Edit Metadata (Image Saver)', pos: [570, 80],
      inputs: [{ name: 'additional_hashes', widget: { name: 'additional_hashes' }, link: null }],
      widgets_values: ['[original]', '[original], detail', '[original]', 'civitai:103:0.8', '["Version"]', '["civitai:102"]'],
    }], links: [], last_node_id: 2, last_link_id: 0,
  };
  context.migrateAdditionalHashes(workflow);
  assert.deepEqual(Array.from(workflow.nodes[0].widgets_values), ['[original]', '[original], detail', '[original]', '["Version"]', '["civitai:102"]']);
  assert.equal(workflow.nodes[1].type, 'PrimitiveStringMultiline');
  assert.equal(workflow.nodes[1].widgets_values[0], 'civitai:103:0.8');
  assert.equal(workflow.nodes[0].inputs[0].link, workflow.links[0][0]);
  context.migrateAdditionalHashes(workflow);
  assert.equal(workflow.nodes.length, 2, 'reloading a migrated workflow must not add another STRING node');
});

class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.listeners = {}; }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text ?? '') + this.children.map(child => child.textContent).join(''); }
  get childElementCount() { return this.children.length; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ''; this.children = children; }
  addEventListener(type, callback) { this.listeners[type] = callback; }
  setAttribute() {}
}

function editorHarness() {
  let poll;
  const result = { source: { resources: [{ value: 'civitai:101', label: 'Original model' }], settings: [] },
    result: { resources: [{ value: 'civitai:101' }], parameters: 'Executed prompt', rows: [['Positive', 'Executed prompt']] }, warning: '' };
  const api = { fetchApi: async () => ({ ok: true, json: async () => result }) };
  const ctx = { app: { registerExtension() {} }, api, document: { createElement: tag => new Element(tag) },
    AbortController, setInterval(callback) { poll = callback; return 1; }, clearInterval() {} };
  vm.createContext(ctx);
  vm.runInContext(source, ctx);
  const nodes = graph([
    { id: 1, type: 'Load Image with Metadata (Image Saver)', widgets: [{ name: 'image', value: 'source.png' }] },
    { id: 2, type: 'Edit Metadata (Image Saver)', inputs: [{ name: 'metadata', link: 1 }, { name: 'width', link: null }],
      widgets: [{ name: 'remove_resources', value: '' }, { name: 'remove_settings', value: '' }] },
    { id: 3, type: 'GetImageSize' },
  ], { 1: { origin_id: 1 }, 2: { origin_id: 3 } });
  const node = nodes[1];
  const editor = ctx.createEditor(node);
  return { nodes, node, editor, api, result, poll: () => poll(),
    execute: () => node.onExecuted({ metadata_editor: [result] }),
    message: () => editor.root.children[0].textContent,
    preview: () => editor.root.children[6].textContent,
    choices: () => editor.root.children[3].textContent,
    buttons: () => editor.root.children[5].children.slice(1),
  };
}

test('disconnect clears an executed preview and reconnect resumes live updates', async () => {
  const h = editorHarness();
  await h.poll();
  h.execute();
  assert.match(h.preview(), /Executed prompt/);
  h.node.inputs[0].link = null;
  // A computed input must not hide the fact that metadata itself is disconnected.
  h.node.inputs[1].link = 2;
  await h.poll();
  assert.match(h.message(), /Connect metadata/);
  assert.equal(h.preview(), '');
  assert.doesNotMatch(h.choices(), /Original model/);
  assert.ok(h.buttons().every(button => button.disabled));
  h.node.inputs[0].link = 1;
  await h.poll();
  assert.match(h.message(), /Run the workflow/);
  assert.doesNotMatch(h.message(), /Showing last execution/);
  h.node.inputs[1].link = null;
  await h.poll();
  assert.match(h.message(), /Live preview/);
  assert.match(h.preview(), /Executed prompt/);
  h.editor.stop();
});

test('an old execution event cannot repopulate a disconnected editor', async () => {
  const h = editorHarness();
  h.execute();
  h.node.inputs[0].link = null;
  await h.poll();
  h.execute();
  await h.poll();
  assert.match(h.message(), /Connect metadata/);
  assert.equal(h.preview(), '');
  h.editor.stop();
});

test('a preview response arriving after disconnect is ignored', async () => {
  const h = editorHarness();
  let respond;
  h.api.fetchApi = () => new Promise(resolve => { respond = resolve; });
  const pending = h.poll();
  h.node.inputs[0].link = null;
  await h.poll();
  respond({ ok: true, json: async () => h.result });
  await pending;
  assert.equal(h.preview(), '');
  assert.match(h.message(), /Connect metadata/);
  h.editor.stop();
});

test('disconnecting an earlier editor clears the downstream preview too', async () => {
  const h = editorHarness();
  const upstream = { id: 4, type: 'Edit Metadata (Image Saver)', graph: h.node.graph,
    inputs: [{ name: 'metadata', link: 1 }], widgets: [] };
  h.nodes.push(upstream);
  h.node.graph.links[3] = { origin_id: 4 };
  h.node.inputs[0].link = 3;
  await h.poll();
  h.execute();
  upstream.inputs[0].link = null;
  await h.poll();
  assert.equal(h.preview(), '');
  assert.match(h.message(), /Connect metadata/);
  upstream.inputs[0].link = 1;
  await h.poll();
  assert.match(h.message(), /Live preview/);
  assert.match(h.preview(), /Executed prompt/);
  h.editor.stop();
});

test('connected processing inputs can still show their last executed result', async () => {
  const h = editorHarness();
  h.node.inputs[1].link = 2;
  await h.poll();
  assert.match(h.message(), /Run the workflow/);
  h.execute();
  await h.poll();
  assert.match(h.preview(), /Executed prompt/);
  h.node.inputs[1].link = null;
  await h.poll();
  assert.match(h.message(), /Live preview/);
  h.editor.stop();
});

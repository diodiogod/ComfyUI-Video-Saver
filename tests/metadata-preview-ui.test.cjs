const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../js/load_image_metadata.js'), 'utf8')
  .replace(/^import .*$/gm, '').replace(/^export /gm, '');

// Exercise the real view buttons, fetch and layout poll without requiring ComfyUI in Node.
class Element {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.style = {};
    this.classList = { toggle() {} };
    this.offsetHeight = 22;
  }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.children = children; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  get firstElementChild() { return this.children[0]; }
  get scrollHeight() { return this.tag === 'table' ? this.children.length * 24 : 200; }
}

async function previewFor(rows, initialView) {
  let tick;
  let stopped = false;
  const data = { rows, parameters: 'Original positive prompt\nSteps: 24', source: 'parameters' };
  const context = {
    app: { registerExtension() {} },
    api: { fetchApi: async () => ({ ok: true, json: async () => data }) },
    document: { createElement: tag => new Element(tag) },
    setInterval(callback) { tick = callback; return 1; },
    clearInterval() { stopped = true; },
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  const sizes = [];
  const node = {
    properties: { metadataView: initialView },
    size: [460, 450],
    widgets: [{ name: 'image', value: 'example.png' }],
    computeSize() { return [this.size[0], 200 + preview.height()]; },
    setSize(size) { sizes.push(size); this.size = size; },
    setDirtyCanvas() {},
  };
  const preview = context.createPreview(node, 'image');
  const widget = { name: 'metadata_preview', margin: 10 };
  node.widgets.push(widget);
  preview.attach(widget);
  tick();
  await new Promise(setImmediate);
  tick();
  return { preview, node, sizes, tick, isStopped: () => stopped };
}

for (const count of [1, 100]) {
  for (const initialView of ['table', 'text']) {
    test(`Table/Text keeps the image allocation unchanged (${count} rows, starts in ${initialView})`, async () => {
      const { preview, node, sizes, tick, isStopped } = await previewFor(
        Array.from({ length: count }, (_, i) => [`Setting ${i}`, 'A long value that can wrap']), initialView);
      const height = preview.height();
      const size = [...node.size];
      const resizeCount = sizes.length;
      const [tableButton, textButton] = preview.root.children[0].children;
      for (const button of [textButton, tableButton, textButton, tableButton]) {
        button.listeners.click();
        tick();
        assert.equal(preview.height(), height, 'metadata height must not steal space from the image');
        assert.deepEqual(Array.from(node.size), size, 'switching views must not resize the node');
        assert.equal(sizes.length, resizeCount);
      }
      textButton.listeners.click();
      assert.equal(preview.root.children[2].firstElementChild.value, 'Original positive prompt\nSteps: 24');
      node.size[0] = 300;
      tick();
      assert.equal(preview.height(), height, 'text wrapping in a narrower node must use the same allocation');
      preview.stop();
      assert.equal(isStopped(), true);
    });
  }
}

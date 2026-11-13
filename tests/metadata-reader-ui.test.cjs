const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function harness() {
  const messages = [];
  let extension;
  const app = {
    registerExtension(value) { extension = value; },
    extensionManager: { toast: { add(message) { messages.push(message); } } }
  };
  const source = fs.readFileSync(path.join(__dirname, '../js/metadata_reader_notifications.js'), 'utf8')
    .replace('import { app } from "../../scripts/app.js";', '');
  vm.runInNewContext(source, { app });
  return { messages, extension };
}

for (const name of ['Image Saver Read Video Metadata', 'Image Saver Read Image Metadata']) {
  test(`${name}: missing metadata produces an actionable warning popup`, () => {
    const { messages, extension } = harness();
    let originalCalls = 0;
    class Node {
      onExecuted(output) {
        originalCalls += 1;
        assert.equal(this.comfyClass, name);
        return output;
      }
    }
    extension.beforeRegisterNodeDef(Node, { name });
    const node = new Node();
    node.comfyClass = name;
    const output = { metadata_warning: ['No generation metadata was found. Use a source saved with metadata enabled.'] };
    assert.equal(node.onExecuted(output), output);
    assert.equal(originalCalls, 1);
    assert.equal(messages.length, 1);
    assert.equal(messages[0].severity, 'warn');
    assert.match(messages[0].detail, /saved with metadata enabled/);
    assert.equal(messages[0].life, 12000);
    node.onExecuted({});
    assert.equal(messages.length, 1, 'a successful read must not show a warning');
  });
}

test('notifications do not modify unrelated nodes', () => {
  const { messages, extension } = harness();
  class Node {}
  extension.beforeRegisterNodeDef(Node, { name: 'Read Image Saver Metadata' });
  assert.equal(Node.prototype.onExecuted, undefined);
  assert.equal(messages.length, 0);
});

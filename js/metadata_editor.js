import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'
import { copyText } from './load_image_metadata.js'

const EDITOR = "Edit Metadata (Image Saver)";
const LOADER = "Load Image with Metadata (Image Saver)";
const FIELDS = ["modelname", "positive", "negative", "width", "height", "seed", "steps", "cfg",
    "sampler_name", "scheduler_name", "denoise", "clip_skip", "additional_hashes", "remove_resources", "remove_settings"];

// Older prototype workflows stored additions as the fourth widget value. Move that value to a
// STRING node so changing the default to a socket cannot shift the two removal selections.
export function migrateAdditionalHashes(graph) {
    for (const node of [...(graph.nodes ?? [])]) {
        const input = node.inputs?.find(input => input.name === "additional_hashes");
        if (node.type !== EDITOR || !input?.widget || node.widgets_values?.length !== 6) continue;
        const [value] = node.widgets_values.splice(3, 1);
        delete input.widget;
        if (node.widgets_values_named) delete node.widgets_values_named.additional_hashes;
        if (!value || input.link != null) continue;
        const id = Math.max(graph.last_node_id ?? 0, ...graph.nodes.map(node => node.id)) + 1;
        const link = Math.max(graph.last_link_id ?? 0, ...(graph.links ?? []).map(link => link[0])) + 1;
        graph.last_node_id = id;
        graph.last_link_id = link;
        input.link = link;
        (graph.links ??= []).push([link, id, 0, node.id, node.inputs.indexOf(input), "STRING"]);
        graph.nodes.push({ id, type: "PrimitiveStringMultiline", title: "Additional resources", pos: [node.pos[0], node.pos[1] - 210],
            size: [400, 160], flags: {}, order: 0, mode: 0, inputs: [],
            outputs: [{ name: "STRING", type: "STRING", links: [link] }], properties: {}, widgets_values: [value] });
    }
}

function sourceNode(node, inputName) {
    const input = node.inputs?.find(input => input.name === inputName);
    const link = input?.link == null ? null : node.graph?.links?.[input.link];
    return link ? node.graph.getNodeById(link.origin_id) : null;
}

export function selectionValues(value) {
    if (!value) return [];
    if (value.trim().startsWith("[")) {
        try {
            const parsed = JSON.parse(value);
            if (Array.isArray(parsed) && parsed.every(item => typeof item === "string")) return parsed;
        } catch { /* Legacy comma-separated selectors are still supported. */ }
    }
    return value.split(",").map(item => item.trim()).filter(Boolean);
}

function literal(node, name, seen = new Set()) {
    if (seen.has(node.id)) throw new Error("A connected value contains a cycle.");
    seen.add(node.id);
    const upstream = sourceNode(node, name);
    if (!upstream) return node.widgets?.find(widget => widget.name === name)?.value;
    if (upstream.type === "Reroute") return literal(upstream, upstream.inputs?.[0]?.name, seen);
    if (/^Primitive/.test(upstream.comfyClass ?? upstream.type) || /Selector.*Image Saver/.test(upstream.comfyClass ?? upstream.type)) {
        return upstream.widgets?.[0]?.value;
    }
    throw new PreviewRequiresExecution("Run the workflow to preview values from connected processing nodes.");
}

class PreviewRequiresExecution extends Error {}

export function previewRequest(node) {
    const editors = [];
    const seen = new Set();
    for (let current = node; current; current = sourceNode(current, current.type === "Reroute" ? current.inputs?.[0]?.name : "metadata")) {
        if (seen.has(current.id) || seen.size >= 32) throw new Error("The metadata connection contains a cycle or too many editors.");
        seen.add(current.id);
        const type = current.comfyClass ?? current.type;
        if (type === LOADER) return { filename: literal(current, "image"), edits: editors.reverse().map(editor => {
            const changes = {};
            for (const field of FIELDS) {
                const value = literal(editor, field);
                if (value !== undefined) changes[field] = value;
            }
            return changes;
        }) };
        if (type === EDITOR) {
            editors.push(current);
        } else if (type !== "Reroute") {
            throw new PreviewRequiresExecution("Run the workflow to inspect metadata from this source.");
        }
    }
    throw new Error("Connect metadata from Load Image with Metadata to choose resources and settings.");
}

const STYLE = `
.isv-editor {display:flex;flex-direction:column;gap:10px;padding:10px;box-sizing:border-box;height:100%;overflow:auto;
 font:12px/1.4 sans-serif;color:var(--input-text,#ddd);background:var(--comfy-input-bg,#222);border-radius:6px;}
.isv-editor > * {flex-shrink:0;}
.isv-editor h4 {margin:0;font-size:13px;}.isv-editor p {margin:0;opacity:.7;}.isv-editor .message {color:var(--error-text,#e6b364);}
.isv-editor .choices {display:flex;flex-direction:column;gap:5px;max-height:230px;overflow:auto;}
.isv-editor label {display:flex;align-items:flex-start;gap:8px;padding:7px;border:1px solid var(--border-color,#444);border-radius:4px;cursor:pointer;}
.isv-editor label:has(input:checked) {border-color:#c08050;background:#60402540;}
.isv-editor label span {display:flex;flex-direction:column;min-width:0;overflow-wrap:anywhere;}
.isv-editor small {font-size:11px;opacity:.65;overflow-wrap:anywhere;}.isv-editor input {accent-color:#df9955;flex:none;}
.isv-editor summary {cursor:pointer;font-weight:bold;}.isv-editor details .choices {margin-top:8px;}
.isv-editor .bar {display:flex;align-items:center;gap:6px;border-top:1px solid var(--border-color,#444);padding-top:10px;}
.isv-editor .bar strong {flex:1;}.isv-editor button {background:var(--comfy-menu-bg,#333);color:inherit;border:1px solid var(--border-color,#555);border-radius:4px;padding:4px 8px;cursor:pointer;}
.isv-editor table {width:100%;border-collapse:collapse;}.isv-editor td {padding:5px 4px;border-bottom:1px solid var(--border-color,#444);vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere;}
.isv-editor td:first-child {width:85px;opacity:.65;}.isv-editor textarea {min-height:150px;resize:vertical;white-space:pre-wrap;width:100%;box-sizing:border-box;background:transparent;color:inherit;border:0;}
`;

function el(tag, text = "") {
    const result = document.createElement(tag);
    result.textContent = text;
    return result;
}

function createEditor(node) {
    const root = el("div");
    root.className = "isv-editor";
    const message = el("p", "Reading source metadata…");
    message.className = "message";
    root.append(message);
    const resourceTitle = el("h4", "Remove resources");
    const hint = el("p", "All resources are kept unless checked below.");
    const resourceList = el("div");
    resourceList.className = "choices";
    root.append(resourceTitle, hint, resourceList);
    const advanced = el("details");
    const settingTitle = el("summary", "Remove settings (advanced)");
    const settingList = el("div");
    settingList.className = "choices";
    advanced.append(settingTitle, settingList);
    root.append(advanced);
    const bar = el("div");
    bar.className = "bar";
    const tableButton = el("button", "Table");
    const textButton = el("button", "Text");
    const copyButton = el("button", "Copy");
    bar.append(el("strong", "Result preview"), tableButton, textButton, copyButton);
    const resultBody = el("div");
    root.append(bar, resultBody);
    for (const type of ["pointerdown", "mousedown", "wheel"]) root.addEventListener(type, event => event.stopPropagation());
    root.addEventListener("keydown", event => {
        if (!event.ctrlKey && !event.metaKey) event.stopPropagation();
    });

    const widgets = Object.fromEntries(["remove_resources", "remove_settings"].map(name => [name, node.widgets.find(widget => widget.name === name)]));
    for (const widget of Object.values(widgets)) {
        widget.type = "converted-widget";
        widget.computeSize = () => [0, -4];
        widget.draw = () => {};
        if (widget.inputEl) widget.inputEl.style.display = "none";
    }
    const additions = node.widgets.find(widget => widget.name === "additional_hashes");
    if (additions?.inputEl) {
        additions.inputEl.setAttribute("aria-label", "Additional resources");
        additions.inputEl.placeholder = "Additional resources — existing resources are kept\nName:HASH:weight, civitai:VERSION_ID:weight, or AIR:weight";
    }
    let data;
    let mode = "table";
    let signature;
    let request = 0;
    let disposed = false;
    let controller;
    let lastExecution;

    function renderChoices(container, choices, name) {
        container.replaceChildren();
        if (!choices) {
            container.append(el("small", "No metadata to inspect."));
            return;
        }
        const selected = selectionValues(widgets[name].value);
        const connected = node.inputs?.some(input => input.name === name && input.link != null);
        for (const choice of choices ?? []) {
            const row = el("label");
            const checkbox = el("input");
            checkbox.type = "checkbox";
            checkbox.checked = selected.some(value => value.toLowerCase() === choice.value.toLowerCase());
            checkbox.disabled = connected;
            checkbox.setAttribute("aria-label", `Remove ${choice.label}`);
            checkbox.addEventListener("change", () => {
                const values = selectionValues(widgets[name].value).filter(value => value.toLowerCase() !== choice.value.toLowerCase());
                if (checkbox.checked) values.push(choice.value);
                widgets[name].value = JSON.stringify(values);
                node.graph?.change();
                refresh();
            });
            const content = el("span");
            content.append(el("strong", choice.label), el("small", choice.detail));
            row.append(checkbox, content);
            container.append(row);
        }
        const available = new Set((choices ?? []).map(choice => choice.value.toLowerCase()));
        for (const value of selected.filter(value => !available.has(value.toLowerCase()))) {
            const clear = el("button", `Clear unmatched removal: ${value}`);
            clear.disabled = connected;
            clear.addEventListener("click", () => {
                widgets[name].value = JSON.stringify(selected.filter(item => item !== value));
                node.graph?.change();
                refresh();
            });
            container.append(clear);
        }
        if (!container.childElementCount) container.append(el("small", "No entries in the source metadata."));
    }

    function render() {
        renderChoices(resourceList, data?.source?.resources, "remove_resources");
        renderChoices(settingList, data?.source?.settings, "remove_settings");
        const count = data?.result?.resources?.length ?? 0;
        resourceTitle.textContent = data?.result ? `Remove resources · ${count} in result` : "Remove resources";
        resultBody.replaceChildren();
        tableButton.disabled = !data?.result || mode === "table";
        textButton.disabled = !data?.result || mode === "text";
        copyButton.disabled = !data?.result?.parameters;
        copyButton.textContent = "Copy";
        if (!data?.result) return;
        if (mode === "text") {
            const area = el("textarea");
            area.readOnly = true;
            area.value = data.result.parameters;
            resultBody.append(area);
        } else {
            const table = el("table");
            for (const [key, value] of data.result.rows ?? []) {
                const row = el("tr");
                row.append(el("td", key), el("td", value));
                table.append(row);
            }
            resultBody.append(table);
        }
    }
    tableButton.addEventListener("click", () => { mode = "table"; render(); });
    textButton.addEventListener("click", () => { mode = "text"; render(); });
    copyButton.addEventListener("click", async () => {
        if (data?.result?.parameters) {
            copyButton.textContent = await copyText(data.result.parameters) ? "Copied" : "Failed";
        }
    });

    async function refresh() {
        if (disposed) return;
        let payload;
        try { payload = previewRequest(node); }
        catch (error) {
            const state = JSON.stringify({ error: error.message,
                values: node.widgets?.filter(widget => FIELDS.includes(widget.name)).map(widget => widget.value) });
            if (signature === state) return;
            signature = state;
            request++;
            controller?.abort();
            // A disconnected/broken metadata path invalidates the old result. Only a connected
            // source that needs execution may show the explicitly labelled historical result.
            if (!(error instanceof PreviewRequiresExecution)) lastExecution = null;
            data = lastExecution ?? null;
            message.textContent = data ? `${error.message} Showing last execution.` : error.message;
            render();
            return;
        }
        const next = JSON.stringify(payload);
        if (signature === next) return;
        signature = next;
        const id = ++request;
        controller?.abort();
        controller = new AbortController();
        message.textContent = "Updating preview…";
        try {
            const response = await api.fetchApi("/image_saver/edit_metadata_preview", {
                method: "POST", headers: { "Content-Type": "application/json" }, body: next, signal: controller.signal,
            });
            const json = await response.json();
            if (disposed || id !== request) return;
            data = response.ok ? json : null;
            message.textContent = json.error || json.warning || "Live preview — existing information stays unless changed.";
            render();
        } catch (error) {
            if (disposed || id !== request || error.name === "AbortError") return;
            data = null;
            message.textContent = `Could not preview metadata: ${error.message}`;
            render();
        }
    }
    const onExecuted = node.onExecuted;
    node.onExecuted = function (output) {
        const result = onExecuted?.apply(this, arguments);
        if (output?.metadata_editor?.[0]) {
            try { previewRequest(node); }
            catch (error) {
                if (!(error instanceof PreviewRequiresExecution)) {
                    // A run queued before disconnection can finish afterwards.
                    signature = undefined;
                    refresh();
                    return result;
                }
            }
            request++;
            controller?.abort();
            data = lastExecution = output.metadata_editor[0];
            message.textContent = "Result from the last workflow execution.";
            render();
        }
        return result;
    };
    const timer = setInterval(refresh, 400);
    return { root, stop() { disposed = true; request++; clearInterval(timer); controller?.abort(); } };
}

app.registerExtension({
    name: "ImageSaver.MetadataEditor",
    beforeConfigureGraph(graphData) { migrateAdditionalHashes(graphData); },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== EDITOR) return;
        if (!document.getElementById("image-saver-editor-style")) {
            const style = el("style", STYLE);
            style.id = "image-saver-editor-style";
            document.head.append(style);
        }
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = original?.apply(this, arguments);
            const editor = createEditor(this);
            const widget = this.addDOMWidget("metadata_editor", "image_saver_editor", editor.root, {
                serialize: false, getMinHeight: () => 420, getMaxHeight: () => 850,
            });
            widget.serialize = false;
            this.setSize([Math.max(this.size[0], 460), Math.max(this.size[1], 900)]);
            const removed = this.onRemoved;
            this.onRemoved = function () { editor.stop(); return removed?.apply(this, arguments); };
            return result;
        };
    },
});

import { app } from "../../scripts/app.js";

const readers = new Set(["Image Saver Read Video Metadata", "Image Saver Read Image Metadata"]);

app.registerExtension({
    name: "ImageSaver.MetadataReaderNotifications",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!readers.has(nodeData.name)) return;
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = onExecuted?.apply(this, arguments);
            for (const message of output?.metadata_warning || []) {
                app.extensionManager.toast.add({
                    severity: "warn",
                    summary: "Image Saver metadata",
                    detail: message,
                    life: 12000,
                });
            }
            return result;
        };
    },
});

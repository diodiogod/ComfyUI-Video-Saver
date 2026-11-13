"""Run with ComfyUI's Python to check reader UI and blocking in its real executor.

This uses temporary local files and a captured server, without opening a port.
It is separate from the isolated pytest suites because it imports ComfyUI.
"""
import asyncio
import argparse
import json
import sys
import tempfile
from inspect import signature
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--comfy-root", type=Path, default=REPO.parents[1], help="ComfyUI checkout whose real API/executor to test")
COMFY_ROOT = parser.parse_args().comfy_root.resolve()
sys.path.insert(0, str(COMFY_ROOT))
sys.argv = [sys.argv[0], "--cpu"]

import comfy.options
comfy.options.enable_args_parsing()

import av
import execution
import folder_paths
import nodes
from PIL import Image
from comfy_api.latest import io
from comfy_api.latest import InputImpl


class CaptureServer:
    client_id = "metadata-reader-test"
    last_node_id = None

    def __init__(self):
        self.events = []

    def send_sync(self, event, data, sid=None):
        self.events.append((event, data))


class MetadataSink:
    called = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"metadata": ("METADATA",)}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    OUTPUT_NODE = True

    def run(self, metadata):
        MetadataSink.called += 1
        return ()


async def main():
    with tempfile.TemporaryDirectory(prefix="image_saver_metadata_execution_") as temp:
        temp = Path(temp)
        folder_paths.set_input_directory(str(temp))
        Image.new("RGB", (8, 8)).save(temp / "plain.png")
        with av.open(str(temp / "plain.mp4"), "w") as container:
            stream = container.add_stream("libx264", rate=1)
            stream.width = stream.height = 16
            stream.pix_fmt = "yuv420p"
            for packet in stream.encode(av.VideoFrame(16, 16, "yuv420p")):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        assert await nodes.load_custom_node(str(REPO))
        assert Path(execution.__file__).resolve().parent == COMFY_ROOT
        for name in ("Image Saver Read Video Metadata", "Image Saver Read Image Metadata", "Read Image Saver Metadata"):
            info = nodes.NODE_CLASS_MAPPINGS[name].GET_NODE_INFO_V1()
            assert info["name"] == name
            if name == "Read Image Saver Metadata":
                assert len(info["output_name"]) == 14
                assert info["output_name"][12:] == ["hashes", "a1111_params"]
                assert "version ID" in info["output_tooltips"][12]
                assert "weights" in info["output_tooltips"][12]
                assert "additional_hashes" in info["output_tooltips"][12]
        nodes.NODE_CLASS_MAPPINGS["Metadata Test Sink"] = MetadataSink
        results = []
        replays_intermediate_ui = "has_intermediate_output" in signature(io.Schema).parameters
        for reader, file in [("Image Saver Read Video Metadata", "plain.mp4"),
                             ("Image Saver Read Image Metadata", "plain.png")]:
            cls = nodes.NODE_CLASS_MAPPINGS[reader]
            assert len(cls.GET_SCHEMA().outputs) == 1
            server = CaptureServer()
            runner = execution.PromptExecutor(server, cache_type=execution.CacheType.CLASSIC,
                                             cache_args={"ram": 0, "ram_inactive": 0})
            prompt = {
                "1": {"class_type": reader, "inputs": {"file": file}},
                "2": {"class_type": "Metadata Test Sink", "inputs": {"metadata": ["1", 0]}},
            }
            for attempt in range(2):
                await runner.execute_async(prompt, f"negative-{file}-{attempt}",
                                           {"client_id": server.client_id}, ["2"])
                assert runner.success, runner.status_messages
                assert MetadataSink.called == 0
                assert not any(event == "execution_error" for event, _ in server.events), server.events
                warnings = [data for event, data in server.events
                            if event == "executed" and (data.get("output") or {}).get("metadata_warning")]
                expected_warnings = attempt + 1 if replays_intermediate_ui else 1
                assert len(warnings) == expected_warnings, server.events
            results.append({"reader": reader, "single_output": True,
                            "warning_emitted": True, "cached_ui_replay_supported": replays_intermediate_ui,
                            "downstream_blocked": True, "execution_errors": 0})

        folder_paths.set_output_directory(str(temp))
        reader = nodes.NODE_CLASS_MAPPINGS["Image Saver Read Video Metadata"]
        expander = nodes.NODE_CLASS_MAPPINGS["Read Image Saver Metadata"]
        maker = nodes.NODE_CLASS_MAPPINGS["Image Saver Metadata"]()
        module = sys.modules[reader.__module__]
        video_helpers = sys.modules[module.__package__ + ".saver.video"]
        civitai_helpers = sys.modules[module.__package__ + ".utils_civitai"]
        original = maker.get_metadata(positive="Original prompt", negative="old negative", seed_value=42,
            additional_hashes="urn:air:sdxl:checkpoint:civitai:11@101,urn:air:sdxl:lora:civitai:12@102:0.65",
            download_civitai_data=False)[0]
        source = video_helpers.embed_video_metadata(temp / "plain.mp4", video_helpers.build_video_metadata(original, {}, {"workflow": {"nodes": []}}))
        loaded = InputImpl.VideoFromFile(source)
        recovered = reader.execute(video=loaded).result[0]
        references = expander.execute(recovered).result[12] + ",ExtraLoRA:FEEDBEEF12:0.8"
        original_lookup = civitai_helpers.get_civitai_info
        civitai_helpers.get_civitai_info = lambda path, hash: (
            {"id": 103, "name": "v1", "model": {"name": "Extra LoRA", "type": "LORA"}} if hash == "FEEDBEEF12" else None
        )
        try:
            updated = maker.get_metadata(positive="New upscale prompt", negative="new negative", seed_value=99, steps=12,
                additional_hashes=references, download_civitai_data=True)[0]
        finally:
            civitai_helpers.get_civitai_info = original_lookup

        processing = {"999": {"class_type": "UpscaleVideo"}}
        workflow = {"workflow": {"nodes": [{"id": 999, "type": "UpscaleVideo"}]}}
        vhs = nodes.NODE_CLASS_MAPPINGS["Image Saver Video Metadata"]().save_video_metadata(
            (True, [source]), metadata=updated, prompt=processing, extra_pnginfo=workflow)
        native = nodes.NODE_CLASS_MAPPINGS["Image Saver Save Video"]().save_video(
            loaded, filename_prefix="upscale", format="mp4", codec="h264", metadata=updated,
            prompt=processing, extra_pnginfo=workflow)
        outputs = [vhs["result"][0][1][0], str(temp / native["ui"]["images"][0]["filename"])]
        for output in outputs:
            tags, width, height = module.read_video_tags(output)
            metadata = module._metadata_from_tags(tags, width, height)
            expanded = expander.execute(metadata).result
            records = {item["modelVersionId"]: item for item in module.read_resource_records(tags)}
            entries = nodes.NODE_CLASS_MAPPINGS["Image Saver"].parse_manual_hashes(expanded[12], set(), False)
            assert metadata.positive == "New upscale prompt"
            assert metadata.seed == 99
            assert set(records) == {101, 102, 103}
            assert records[102]["weight"] == 0.65
            assert records[103]["weight"] == 0.8
            assert set(entries) == {"civitai:101", "civitai:102", "civitai:103"}
        print(json.dumps({"comfy_root": str(COMFY_ROOT), "node_info_checked": 3, "execution": results,
                          "resource_export_append_and_resave": {"native": True, "vhs": True, "resource_ids": [101, 102, 103]}}))


if __name__ == "__main__":
    asyncio.run(main())

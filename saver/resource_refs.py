"""Comma-separated resource references shared by metadata readers and savers."""

import math
import re
from typing import Any


MAX_HASH_LENGTH = 16
NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
AIR_PATTERN = r"urn:air:[\w/-]+:([\w/-]+):civitai:(\d+)@([1-9]\d*)(?:\+\d+)?(?:\.[\w-]+)?"
REFERENCE_PATTERN = re.compile(
    rf"(?P<identifier>civitai:[1-9]\d*|{AIR_PATTERN})(?::(?P<weight>{NUMBER}))?", re.IGNORECASE
)


def positive_id(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal() and int(value) > 0:
        return int(value)
    return None


def resource_from_identifier(identifier: str) -> dict[str, Any] | None:
    match = re.fullmatch(r"civitai:([1-9]\d*)", identifier, re.IGNORECASE)
    if match:
        return {"modelVersionId": int(match[1])}
    match = re.fullmatch(AIR_PATTERN, identifier, re.IGNORECASE)
    if match:
        return {"modelVersionId": int(match[3]), "type": match[1].lower(), "air": identifier}
    return None


def resource_identity(resource: dict[str, Any]) -> tuple[str, str] | None:
    """Keep a valid AIR or encode a known version/type without a lookup."""
    air = resource.get("air")
    record = resource_from_identifier(air) if isinstance(air, str) else None
    version_id = positive_id(resource.get("modelVersionId"))
    if record and "air" in record and (version_id is None or version_id == record["modelVersionId"]):
        return air, record["type"]
    if version_id is None:
        return None
    kind = resource.get("type")
    kind = kind.lower() if isinstance(kind, str) else "other"
    kind = {"textualinversion": "embedding", "locon": "lycoris"}.get(kind, kind)
    if not re.fullmatch(r"[a-z0-9_/-]+", kind):
        kind = "other"
    model_id = positive_id(resource.get("modelId")) or 0
    return f"urn:air:other:{kind}:civitai:{model_id}@{version_id}", kind


def parse_reference(entry: str) -> tuple[str | None, float | None, str] | None:
    """Return name, weight and a hash or explicitly identified resource."""
    entry = entry.strip()
    match = REFERENCE_PATTERN.fullmatch(entry)
    if match:
        weight = float(match["weight"]) if match["weight"] is not None else None
        if weight is not None and not math.isfinite(weight):
            return None
        identifier = match["identifier"]
        if identifier.lower().startswith("civitai:"):
            identifier = identifier.lower()
        return None, weight, identifier
    if entry.lower().startswith(("civitai:", "urn:air:")):
        return None

    parts = entry.split(":")
    weight = None
    # Preserve names such as LORA:style from A1111 Hashes, including an
    # entirely numeric hash. A bare HASH:weight is also accepted.
    prefixed_hash = len(parts) == 3 and parts[0].lower() in {"lora", "embed"}
    if (len(parts) >= 2 and not prefixed_hash and re.fullmatch(NUMBER, parts[-1].strip())
            and (len(parts) >= 3 or re.fullmatch(r"[a-fA-F0-9]{1,16}", parts[0].strip()))):
        weight = float(parts.pop())
        if not math.isfinite(weight):
            return None
    identifier = parts.pop().strip()
    name = ":".join(parts).strip() or None
    if not identifier or len(identifier) > MAX_HASH_LENGTH or re.search(r"[\s,:]", identifier):
        return None
    return name, weight, identifier


def format_reference(name: str | None, weight: float | None, identifier: str) -> str:
    entry = f"{name}:{identifier}" if name and resource_from_identifier(identifier) is None else identifier
    return f"{entry}:{weight}" if weight is not None else entry


def export_resource_references(resources: Any, hashes: str) -> str:
    """Export existing identities/weights without converting IDs into hashes."""
    entries = []
    seen = set()

    def append(name, weight, identifier):
        record = resource_from_identifier(identifier)
        key = ("version", record["modelVersionId"]) if record else ("hash", identifier.lower())
        if key not in seen:
            seen.add(key)
            entries.append(format_reference(name, weight, identifier))

    for resource in resources if isinstance(resources, list) else []:
        if not isinstance(resource, dict):
            continue
        identity = resource_identity(resource)
        weight = resource.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            weight = None
        if identity:
            identifier = identity[0] if resource.get("air") or resource.get("type") else f"civitai:{positive_id(resource['modelVersionId'])}"
            append(None, weight, identifier)
        elif isinstance(resource.get("hash"), str):
            parsed = parse_reference(resource["hash"])
            if parsed:
                name = resource.get("name")
                append(name if isinstance(name, str) else parsed[0], weight, parsed[2])
    for entry in (hashes or "").replace("\n", ",").split(","):
        parsed = parse_reference(entry)
        if parsed:
            append(*parsed)
    return ",".join(entries)

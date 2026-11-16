"""Resource reuse for image metadata, without model files or Civitai requests."""
import json

from .image_metadata import _format_value, edit_parameters, parse_parameters
from .resource_refs import export_resource_references, format_reference, parse_reference, resource_from_identifier


def selection_values(value: str) -> list[str]:
    """The checklist stores JSON; comma-separated selectors remain valid for API workflows."""
    if value.strip().startswith("["):
        values = json.loads(value)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("Metadata selections must be a list of strings.")
        return values
    return [item.strip() for item in value.split(",") if item.strip()]


def settings_resources(settings):
    value = settings.get("Civitai resources", [])
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def resource_version(record):
    identifier = resource_from_identifier(record.get("air", "")) if isinstance(record.get("air"), str) else None
    return str(record.get("modelVersionId") or (identifier or {}).get("modelVersionId") or "")


def resource_references(parameters: str, fallback: str = "") -> str:
    _, _, settings = parse_parameters(parameters)
    return references_from_settings(settings, fallback)


def references_from_settings(settings, fallback: str = "") -> str:
    records = settings_resources(settings)
    weighted = settings.get("Resource references", [])
    for entry in weighted if isinstance(weighted, list) else []:
        parsed = parse_reference(entry) if isinstance(entry, str) else None
        if parsed:
            name, weight, identifier = parsed
            records.append({"hash": identifier, "name": name, "weight": weight})
    hashes = settings.get("Hashes", {})
    parts = []
    if settings.get("Model hash"):
        parts.append(format_reference(str(settings.get("Model", "model")), None, settings["Model hash"]))
    if isinstance(hashes, dict):
        for name, value in hashes.items():
            if isinstance(value, str):
                name = str(settings.get("Model", "model")) if name == "model" else str(name).split(":", 1)[-1]
                parts.append(format_reference(name, None, value))
    return export_resource_references(records, fallback + "," + ",".join(parts))


def remove_references(references: str, selectors: list[str]) -> str:
    wanted = {value.casefold() for value in selectors}
    kept = []
    for entry in references.split(","):
        parsed = parse_reference(entry)
        if not parsed:
            continue
        name, _, identifier = parsed
        record = resource_from_identifier(identifier)
        values = [identifier, name, (name or "").split(":", 1)[-1]]
        if record:
            values += [str(record["modelVersionId"]), f"civitai:{record['modelVersionId']}"]
        if not any(value and value.casefold() in wanted for value in values):
            kept.append(entry)
    return ",".join(kept)


def merge_resources(parameters: str, references: str, additional: str) -> tuple[str, str]:
    """Merge explicit identities; an explicitly supplied weight wins. No identity is guessed."""
    if not additional.strip():
        return parameters, references
    additions = []
    for entry in additional.replace("\n", ",").split(","):
        if not entry.strip():
            continue
        parsed = parse_reference(entry)
        if parsed is None:
            raise ValueError(f"Invalid additional resource reference: {entry.strip()}")
        additions.append(parsed)
    _, _, settings = parse_parameters(parameters)
    records = settings_resources(settings)
    hashes = dict(settings.get("Hashes") or {})
    all_refs = [parse_reference(entry) for entry in resource_references(parameters, references).split(",") if entry]
    entries = {}

    def identity(identifier):
        record = resource_from_identifier(identifier)
        return ("version", record["modelVersionId"]) if record else ("hash", identifier.casefold())

    for parsed in all_refs:
        if parsed:
            entries.setdefault(identity(parsed[2]), parsed)
    for name, weight, identifier in additions:
        key = identity(identifier)
        previous = entries.get(key)
        if previous:
            name = previous[0] or name
            weight = previous[1] if weight is None else weight
            identifier = previous[2]
        entries[key] = (name, weight, identifier)
        record = resource_from_identifier(identifier)
        if record:
            existing = next((item for item in records if resource_version(item) == str(record["modelVersionId"])), None)
            if existing is None:
                records.append(record)
                existing = record
            if weight is not None:
                existing["weight"] = weight
        else:
            existing_name = next((key for key, value in hashes.items()
                                  if isinstance(value, str) and value.casefold() == identifier.casefold()), None)
            if existing_name is None:
                hash_name = name or f"manual:{identifier}"
                if hash_name in hashes and hashes[hash_name].casefold() != identifier.casefold():
                    hash_name += f":{identifier}"
                hashes[hash_name] = identifier
    changes = {}
    for key, value in (("Hashes", hashes), ("Civitai resources", records)):
        if value != settings.get(key, {} if key == "Hashes" else []):
            changes[key] = json.dumps(value, separators=(",", ":")) if value else None
    weighted = [format_reference(*entry) for entry in entries.values()
                if entry[1] is not None and resource_from_identifier(entry[2]) is None]
    if weighted != settings.get("Resource references", []):
        changes["Resource references"] = json.dumps(weighted, separators=(",", ":")) if weighted else None
    if not parameters:
        raise ValueError("This metadata has no saved parameter text. Create Image Saver Metadata before adding resources.")
    parameters = edit_parameters(parameters, settings=changes) if changes else parameters
    return parameters, ",".join(format_reference(*entry) for entry in entries.values())


def metadata_choices(metadata):
    """Readable choices with exact selectors, so the user never needs storage keys or resource IDs."""
    if metadata is None:
        return {"parameters": "", "resources": [], "settings": [], "rows": []}
    positive, negative, settings = parse_parameters(metadata.a111_params)
    refs = resource_references(metadata.a111_params, metadata.final_hashes)
    records = settings_resources(settings)
    resources = []
    for entry in refs.split(","):
        parsed = parse_reference(entry)
        if not parsed:
            continue
        name, weight, identifier = parsed
        resource = resource_from_identifier(identifier)
        if resource:
            record = next((item for item in records if resource_version(item) == str(resource["modelVersionId"])), {})
            title = " — ".join(str(record[key]) for key in ("modelName", "modelVersionName") if record.get(key))
            title = title or f"Civitai version {resource['modelVersionId']}"
            selector = f"civitai:{resource['modelVersionId']}"
            detail = f"{record.get('type', resource.get('type', 'resource'))} · version {resource['modelVersionId']}"
        else:
            title = name.split(":", 1)[-1] if name else "Unnamed resource"
            selector, detail = identifier, identifier
        if weight is not None:
            detail += f" · weight {weight:g}"
        resources.append({"value": selector, "label": title, "detail": detail})
    labels = {"CFG scale": "CFG", "Size": "Generation dimensions", "Model hash": "Primary model hash",
              "Hashes": "Resource hashes", "Civitai resources": "Civitai resource records"}
    choices = [{"value": key, "label": labels.get(key, key), "detail": _format_value(key, value)} for key, value in settings.items()]
    rows = [("Positive", positive), ("Negative", negative)] + [(key, _format_value(key, value)) for key, value in settings.items()]
    return {"parameters": metadata.a111_params, "resources": resources, "settings": choices, "rows": rows}

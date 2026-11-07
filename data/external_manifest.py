"""Dataset-independent table parsing, field/path mapping and explicit labels."""
import codecs
import copy
import csv
from pathlib import Path
from string import Formatter

import yaml


PRESETS = {"bonafide-spoof": {"bonafide": 0, "spoof": 1},
           "real-fake": {"real": 0, "fake": 1},
           "genuine-synthetic": {"genuine": 0, "synthetic": 1}}


class UniqueMappingLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader, node):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ValueError("Mapping keys must be scalar values") from error
        if duplicate:
            raise ValueError(f"Conflicting/duplicate mapping key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


UniqueMappingLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def validate_label_mapping(mapping):
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("An explicit nonempty label_mapping is required (Real=0, Fake=1)")
    for key, value in mapping.items():
        if not isinstance(key, str) or not key or key != key.strip():
            raise ValueError("Label mapping keys must be nonempty strings; quote numeric labels")
        if type(value) is not int or value not in (0, 1):
            raise ValueError("Label mapping values must be integers Real=0 or Fake=1")
    if set(mapping.values()) != {0, 1}:
        raise ValueError("Label mapping must define both Real=0 and Fake=1")
    return dict(mapping)


def load_label_mapping(reference):
    if reference in PRESETS:
        return dict(PRESETS[reference])
    with Path(reference).open(encoding="utf-8-sig") as handle:
        payload = yaml.load(handle, Loader=UniqueMappingLoader)
    if isinstance(payload, dict) and "label_mapping" in payload:
        if set(payload) != {"label_mapping"}:
            raise ValueError("Unexpected fields alongside label_mapping")
        payload = payload["label_mapping"]
    return validate_label_mapping(payload)


def validate_input_mapping(payload):
    """Validate a small declarative schema, with no dataset names or executable rules."""
    if not isinstance(payload, dict):
        raise ValueError("Input mapping must be a YAML/JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise ValueError("Input mapping option names must be strings")
    allowed = {"format", "columns", "paths", "labels", "label_mapping", "metadata"}
    if set(payload) - allowed:
        raise ValueError(f"Unknown input mapping options: {sorted(set(payload) - allowed)}")
    result = copy.deepcopy(payload)
    schemas = {"format": {"delimiter", "header", "encoding", "column_count"},
               "columns": {"uttid", "audio_path", "label", "attack_id"},
               "paths": {"root", "template"}}
    for section, keys in schemas.items():
        value = result.setdefault(section, {})
        if not isinstance(value, dict) or set(value) - keys:
            raise ValueError(f"Invalid {section} mapping; supported keys: {sorted(keys)}")
    fmt = result["format"]
    if "header" in fmt and type(fmt["header"]) is not bool:
        raise ValueError("format.header must be true or false")
    if "delimiter" in fmt:
        delimiter = fmt["delimiter"]
        if (not isinstance(delimiter, str) or
                (delimiter != "whitespace" and (len(delimiter) != 1 or delimiter in '\r\n\0"'))):
            raise ValueError("format.delimiter must be one character or 'whitespace'")
    if "encoding" in fmt:
        try:
            if not isinstance(fmt["encoding"], str):
                raise LookupError
            codecs.lookup(fmt["encoding"])
        except LookupError as error:
            raise ValueError("Unknown format.encoding") from error
    if "column_count" in fmt and (type(fmt["column_count"]) is not int or fmt["column_count"] <= 0):
        raise ValueError("format.column_count must be a positive integer")
    metadata = result.setdefault("metadata", {})
    if not isinstance(metadata, dict) or any(not isinstance(key, str) or not key.isidentifier()
                                            or key in ("value", "uttid") for key in metadata):
        raise ValueError("metadata must map simple names (except value/uttid) to columns")
    for selector in [*result["columns"].values(), *metadata.values()]:
        if not ((type(selector) is int and selector >= 0) or (isinstance(selector, str) and selector.strip() == selector and selector)):
            raise ValueError("Column selectors must be names or zero-based nonnegative integers")
        if fmt.get("header") is False and not isinstance(selector, int):
            raise ValueError("Headerless input requires zero-based integer column selectors")
    for key, value in result["paths"].items():
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"paths.{key} must be a nonempty string without surrounding whitespace")
    template = result["paths"].get("template", "{value}")
    try:
        for _, field, spec, conversion in Formatter().parse(template):
            if field is not None and (not field.isidentifier() or spec or conversion):
                raise ValueError("Path templates allow only simple {value}, {uttid} or metadata fields")
    except ValueError as error:
        raise ValueError(f"Invalid paths.template: {error}") from error
    if "labels" in result and "label_mapping" in result:
        raise ValueError("Conflicting labels and label_mapping; specify only one")
    if "labels" in result:
        groups = result["labels"]
        if not isinstance(groups, dict) or set(groups) != {"real", "fake"}:
            raise ValueError("labels must contain real and fake lists")
        mapping = {}
        for name, label in (("real", 0), ("fake", 1)):
            values = groups[name]
            if not isinstance(values, list) or not values:
                raise ValueError(f"labels.{name} must be a nonempty list of strings")
            for raw in values:
                if not isinstance(raw, str) or not raw or raw != raw.strip():
                    raise ValueError("Label tokens must be nonempty strings; quote numeric labels")
                if raw in mapping:
                    raise ValueError(f"Conflicting/duplicate label token: {raw!r}")
                mapping[raw] = label
        result.pop("labels")
        result["label_mapping"] = mapping
    if "label_mapping" in result:
        result["label_mapping"] = validate_label_mapping(result["label_mapping"])
    return result


def load_input_mapping(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return validate_input_mapping(yaml.load(handle, Loader=UniqueMappingLoader))


def resolve_input_mapping(path, mapping=None, *, labeled=True, input_mapping=None):
    """Return fully resolved settings, also persisted verbatim in the run report."""
    path = Path(path).resolve()
    options = validate_input_mapping({} if input_mapping is None else input_mapping)
    fmt = options["format"]
    if "delimiter" not in fmt:
        if path.suffix.lower() not in (".csv", ".tsv"):
            raise ValueError("Non-CSV/TSV input requires explicit format.delimiter")
        fmt["delimiter"] = "\t" if path.suffix.lower() == ".tsv" else ","
    fmt.setdefault("header", True)
    fmt.setdefault("encoding", "utf-8-sig")
    if mapping is not None:
        if "label_mapping" in options:
            raise ValueError("Conflicting --label-map and input mapping labels; choose one")
        options["label_mapping"] = validate_label_mapping(mapping)
    if labeled and "label_mapping" not in options:
        raise ValueError("Labeled input requires explicit --label-map or mapping labels")
    if not fmt["header"] and not {"uttid", "audio_path"} <= set(options["columns"]):
        raise ValueError("Headerless input requires explicit uttid/audio_path columns")
    if not fmt["header"] and labeled and "label" not in options["columns"]:
        raise ValueError("Headerless labeled input requires an explicit label column")
    root = Path(options["paths"].get("root", str(path.parent)))
    options["paths"]["root"] = str((path.parent / root).resolve() if not root.is_absolute() else root.resolve())
    options["paths"].setdefault("template", "{value}")
    return options


def read_manifest(path, mapping=None, *, labeled=True, input_mapping=None):
    """Read any configured table into canonical records, without guessing semantics.

    Selectors are exact column names or zero-based integer positions. Relative
    roots are relative to the source table. Unknown Query labels remain None.
    """
    path = Path(path).resolve()
    options = resolve_input_mapping(path, mapping, labeled=labeled, input_mapping=input_mapping)
    fmt, columns = options["format"], options["columns"]
    labels = options.get("label_mapping", {})
    records, ids, paths = [], set(), set()
    header, width = None, fmt.get("column_count")
    with path.open(encoding=fmt["encoding"], newline="") as handle:
        if fmt["delimiter"] == "whitespace":
            stream = ((line, text.split()) for line, text in enumerate(handle, 1))
        else:
            reader = csv.reader(handle, delimiter=fmt["delimiter"], strict=True)
            stream = ((reader.line_num, fields) for fields in reader)
        try:
            for line, fields in stream:
                if not fields:
                    continue
                if width is None:
                    width = len(fields)
                if len(fields) != width:
                    raise ValueError(f"{path}:{line}: expected {width} columns, got {len(fields)}")
                if fmt["header"] and header is None:
                    if len(set(fields)) != len(fields) or any(not field or field != field.strip() for field in fields):
                        raise ValueError(f"{path}:{line}: invalid/duplicate header")
                    header = fields
                    continue
                def column(selector):
                    if isinstance(selector, str):
                        if header is None or selector not in header:
                            raise ValueError(f"{path}:{line}: missing column {selector!r}")
                        selector = header.index(selector)
                    if selector >= len(fields):
                        raise ValueError(f"{path}:{line}: column index {selector} outside {len(fields)} columns")
                    return fields[selector]

                def get(name, required=False):
                    selector = columns.get(name)
                    if selector is None and header is not None and name in header:
                        selector = name
                    if selector is None:
                        if required:
                            raise ValueError(f"{path}:{line}: missing column mapping for {name}")
                        return ""
                    value = column(selector)
                    if value != value.strip() or (required and not value):
                        raise ValueError(f"{path}:{line}: missing/invalid {name}")
                    return value

                uid, raw_path = get("uttid", True), get("audio_path", True)
                raw = get("label", labeled)
                if labeled and raw not in labels:
                    raise ValueError(f"{path}:{line}: unconfigured label {raw!r}")
                if not labeled and (raw or (header is not None and "label" in header and column("label"))):
                    raise ValueError("Unlabeled Query must not provide labels")
                metadata = dict(zip(header or map(str, range(width)), fields))
                for key, selector in options["metadata"].items():
                    value = column(selector)
                    if key in metadata and metadata[key] != value:
                        raise ValueError(f"{path}:{line}: conflicting metadata name {key!r}")
                    metadata[key] = value
                try:
                    audio_text = options["paths"]["template"].format_map(dict(metadata, value=raw_path, uttid=uid))
                except (KeyError, ValueError) as error:
                    raise ValueError(f"{path}:{line}: unresolved path template field: {error}") from error
                if not audio_text or audio_text != audio_text.strip():
                    raise ValueError(f"{path}:{line}: empty/invalid resolved audio path")
                audio = Path(audio_text)
                audio = (Path(options["paths"]["root"]) / audio).resolve() if not audio.is_absolute() else audio.resolve()
                if not audio.is_file():
                    raise FileNotFoundError(f"{path}:{line}: audio does not exist: {audio}")
                if uid in ids or audio in paths:
                    raise ValueError(f"{path}:{line}: duplicate uttid or resolved audio_path: {uid}")
                attack = get("attack_id")
                ids.add(uid)
                paths.add(audio)
                records.append(dict(uttid=uid, audio_path=str(audio), raw_label=raw,
                                    label=labels[raw] if labeled else None, attack_id=attack, metadata=metadata))
        except csv.Error as error:
            raise ValueError(f"{path}: invalid delimited table: {error}") from error
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def directory_records(directory, role, label=None):
    """Stable recursive order and role-qualified relative IDs; decode errors fail visibly."""
    from .audio_benchmark import AUDIO_EXTENSIONS
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError(f"Missing {role} directory: {directory}; 10-shot requires labeled Support")
    rows = [dict(uttid=f"{role}/{path.relative_to(directory).as_posix()}",
                 audio_path=str(path.resolve()), label=label,
                 raw_label="" if label is None else ("real" if label == 0 else "fake"), attack_id="")
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS]
    if not rows:
        raise ValueError(f"No audio in {role} directory: {directory}")
    return rows

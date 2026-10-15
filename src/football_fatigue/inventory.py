"""Read-only discovery and inventory of the raw FIFA World Cup 2022 dataset.

Files are classified from their directory context (the download folder name and,
for event data, the version subfolder) plus file name. File contents are never
opened: this module records only paths and sizes. It makes no claim about schemas,
and it does not decide which Event Data version is authoritative.

Policies:
- Classification inspects the complete directory context. Conflicting family
  folders, or nesting beyond the observed Google Drive layout, are reported as
  unrecognised files rather than guessed.
- Symbolic links and Windows junctions are never followed or counted as dataset
  files; each is reported as a skipped entry.
- Any traversal or metadata error aborts the scan with InventoryScanError, so an
  incomplete scan is never presented as a complete inventory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

TRACKING = "tracking"
EVENTS = "events"
METADATA = "metadata"
ROSTERS = "rosters"
MATCH_MODALITIES = (TRACKING, EVENTS, METADATA, ROSTERS)

SPECIFICATION = "specification"
REFERENCE = "reference"
UNKNOWN = "unknown"

# Event versions are subfolder names. ``None`` means the file sits directly in the
# Event Data folder, which cannot collide with any real folder name.
ROOT_LABEL = "(root)"

# Download folder base names (Google Drive appends "-<timestamp>-<n>-<part>").
_FAMILY_DIRS = {
    "tracking data": TRACKING,
    "event data": EVENTS,
    "metadata": METADATA,
    "rosters": ROSTERS,
}
_DOWNLOAD_SUFFIX = re.compile(r"-\d{8}T\d{6}Z-\d+-\d+$")
# Matched with fullmatch(); [0-9] (not \d) so only ASCII digits are accepted.
_MATCH_FILE = {
    TRACKING: re.compile(r"([0-9]+)\.jsonl\.bz2", re.IGNORECASE),
    EVENTS: re.compile(r"([0-9]+)\.json", re.IGNORECASE),
    METADATA: re.compile(r"([0-9]+)\.json", re.IGNORECASE),
    ROSTERS: re.compile(r"([0-9]+)\.json", re.IGNORECASE),
}
_DOCUMENT_SUFFIXES = {".pdf", ".docx", ".csv"}


@dataclass(frozen=True)
class FileRecord:
    path: str  # POSIX-style, relative to the scanned root
    kind: str  # a modality, SPECIFICATION, REFERENCE, or UNKNOWN
    size: int
    match_id: str | None = None
    version: str | None = None  # events only; None means the Event Data root folder
    note: str | None = None  # why a file was classified as unknown


@dataclass(frozen=True)
class Issue:
    kind: str
    detail: str
    match_id: str | None = None
    paths: tuple[str, ...] = ()


@dataclass
class MatchInventory:
    match_id: str
    tracking: list[str] = field(default_factory=list)
    metadata: list[str] = field(default_factory=list)
    rosters: list[str] = field(default_factory=list)
    events: dict[str | None, list[str]] = field(default_factory=dict)  # version -> paths


class InventoryScanError(Exception):
    """The scan could not enumerate or stat everything, so no inventory is valid."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"{len(errors)} filesystem error(s) during scan")


def version_label(version: str | None) -> str:
    """Human-readable event version label (never used as an identifier)."""
    if version is None:
        return ROOT_LABEL
    if version == ROOT_LABEL:
        return f"'{ROOT_LABEL}' (literal folder)"
    return version


def _version_key(version: str | None) -> tuple[bool, str]:
    return (version is not None, version or "")


@dataclass
class Inventory:
    root: str
    matches: dict[str, MatchInventory]
    files: list[FileRecord]
    issues: list[Issue]

    @property
    def match_ids(self) -> list[str]:
        return sorted(self.matches, key=int)

    @property
    def event_versions(self) -> list[str | None]:
        return sorted({v for m in self.matches.values() for v in m.events}, key=_version_key)

    def coverage(self) -> dict[str, int]:
        """Number of matches with at least one file per modality."""
        counts = {name: 0 for name in MATCH_MODALITIES}
        for m in self.matches.values():
            counts[TRACKING] += bool(m.tracking)
            counts[METADATA] += bool(m.metadata)
            counts[ROSTERS] += bool(m.rosters)
            counts[EVENTS] += bool(m.events)
        return counts

    def event_version_coverage(self) -> dict[str | None, int]:
        return {v: sum(v in m.events for m in self.matches.values()) for v in self.event_versions}

    def files_of_kind(self, kind: str) -> list[FileRecord]:
        return [f for f in self.files if f.kind == kind]


def _family_of(directory_name: str) -> str | None:
    base = _DOWNLOAD_SUFFIX.sub("", directory_name).lower()
    return _FAMILY_DIRS.get(base)


def _is_download_wrapper(directory_name: str) -> bool:
    return _DOWNLOAD_SUFFIX.search(directory_name) is not None


def _unknown(note: str) -> tuple[str, None, None, str]:
    return UNKNOWN, None, None, note


def classify(parts: tuple[str, ...]) -> tuple[str, str | None, str | None, str | None]:
    """Classify a path (relative components, file name last).

    Returns (kind, match_id, event_version, note). Supported layout, as observed:
    ``<Family>[-<download suffix>]/[<Family>/][<event version>/]<file>`` where the
    event version folder applies to Event Data only. Anything else that touches a
    family folder is reported as unknown with an explanatory note.
    """
    name = parts[-1]
    dirs = parts[:-1]
    families = [_family_of(d) for d in dirs]
    present = {f for f in families if f}
    if not present:
        if not dirs and Path(name).suffix.lower() in _DOCUMENT_SUFFIXES:
            return REFERENCE, None, None, None
        return _unknown("not inside a recognised dataset folder")
    if len(present) > 1:
        return _unknown("conflicting family directories: " + ", ".join(sorted(present)))
    family = next(iter(present))
    if families[0] is None:
        return _unknown(f"unsupported nesting: {family} folder is not a top-level directory")
    rest = list(dirs[1:])
    # Drive's inner folder: only the exact bare family name, never a timestamped wrapper.
    if rest and families[1] == family and not _is_download_wrapper(rest[0]):
        rest = rest[1:]
    for directory in rest:
        if _family_of(directory):
            if _is_download_wrapper(directory):
                return _unknown(f"unsupported nesting: nested download wrapper {directory!r}")
            return _unknown(f"unsupported nesting: repeated {family} directory")
    version: str | None = None
    if family == EVENTS:
        if len(rest) > 1:
            return _unknown("unsupported nesting: more than one folder below Event Data")
        version = rest[0] if rest else None
    elif rest:
        return _unknown(f"unsupported nesting: subfolder {'/'.join(rest)} in {family} folder")
    found = _MATCH_FILE[family].fullmatch(name)
    if found:
        return family, found.group(1), version, None
    if Path(name).suffix.lower() == ".pdf":
        return SPECIFICATION, None, None, None
    return _unknown(f"unexpected file name in {family} folder")


def _describe(error: OSError, root: Path) -> str:
    target = getattr(error, "filename", None)
    try:
        target = Path(target).relative_to(root).as_posix() if target else str(root)
    except ValueError:
        target = str(target)
    return f"{target}: {error.strerror or error}"


def scan(root: str | os.PathLike[str]) -> Inventory:
    """Scan ``root`` read-only and build an inventory. Never opens file contents.

    Symbolic links and junctions are not followed and are reported as skipped
    entries. Raises NotADirectoryError for a bad root and InventoryScanError if the
    root is itself a symlink or junction, or if any directory cannot be listed or
    any file's metadata cannot be read.
    """
    root_path = Path(root)
    # Check the root itself with lstat-based tests before anything follows it.
    checked = os.path.normpath(root_path)
    if os.path.islink(checked) or os.path.isjunction(checked):
        raise InventoryScanError(
            [f"{root_path}: the scan root is a symbolic link or junction; "
             "pass the real raw-data directory instead"]
        )
    if not root_path.is_dir():
        raise NotADirectoryError(f"dataset root is not a directory: {root_path}")

    errors: list[str] = []
    skipped: list[Issue] = []
    files: list[FileRecord] = []

    def skip(full: Path, reason: str) -> None:
        relative = full.relative_to(root_path).as_posix()
        skipped.append(Issue("skipped_entry", reason, None, (relative,)))

    def on_walk_error(error: OSError) -> None:
        errors.append("cannot list directory " + _describe(error, root_path))

    for current, dirnames, filenames in os.walk(root_path, onerror=on_walk_error, followlinks=False):
        for dirname in list(dirnames):
            full = Path(current) / dirname
            if os.path.islink(full) or os.path.isjunction(full):
                dirnames.remove(dirname)
                skip(full, "directory link (symlink or junction) not followed")
        dirnames.sort()
        for filename in sorted(filenames):
            full = Path(current) / filename
            try:
                mode = os.lstat(full)
            except OSError as error:
                errors.append("cannot read file metadata " + _describe(error, root_path))
                continue
            if stat.S_ISLNK(mode.st_mode):
                skip(full, "symbolic link not followed")
                continue
            if not stat.S_ISREG(mode.st_mode):
                skip(full, "not a regular file")
                continue
            relative = full.relative_to(root_path)
            kind, match_id, version, note = classify(relative.parts)
            files.append(
                FileRecord(
                    path=relative.as_posix(),
                    kind=kind,
                    size=mode.st_size,
                    match_id=match_id,
                    version=version,
                    note=note,
                )
            )

    if errors:
        raise InventoryScanError(errors)

    matches: dict[str, MatchInventory] = {}
    for record in files:
        if record.match_id is None:
            continue
        match = matches.setdefault(record.match_id, MatchInventory(record.match_id))
        if record.kind == EVENTS:
            match.events.setdefault(record.version, []).append(record.path)
        else:
            getattr(match, record.kind).append(record.path)

    inventory = Inventory(root=str(root_path), matches=matches, files=files, issues=[])
    inventory.issues = _find_issues(inventory) + skipped
    return inventory


def _find_issues(inventory: Inventory) -> list[Issue]:
    issues: list[Issue] = []
    versions = inventory.event_versions
    for match_id in inventory.match_ids:
        m = inventory.matches[match_id]
        for modality in (TRACKING, METADATA, ROSTERS):
            paths = getattr(m, modality)
            if not paths:
                issues.append(Issue("missing", f"no {modality} file", match_id))
            elif len(paths) > 1:
                issues.append(
                    Issue(
                        "duplicate_candidates",
                        f"{len(paths)} {modality} candidates",
                        match_id,
                        tuple(sorted(paths)),
                    )
                )
        if not m.events:
            issues.append(Issue("missing", "no events file in any version", match_id))
        for version in versions:
            paths = m.events.get(version, [])
            label = version_label(version)
            if m.events and not paths:
                issues.append(Issue("missing", f"no events file for version {label}", match_id))
            elif len(paths) > 1:
                issues.append(
                    Issue(
                        "duplicate_candidates",
                        f"{len(paths)} candidates for events version {label}",
                        match_id,
                        tuple(sorted(paths)),
                    )
                )
    for record in inventory.files_of_kind(UNKNOWN):
        issues.append(Issue("unknown_file", record.note or "unclassified", None, (record.path,)))
    return issues


# --- reporting -----------------------------------------------------------------


def _version_json(version: str | None) -> dict:
    """Unambiguous version: root is {"root": true}; a folder named "(root)" is not."""
    return {"root": version is None, "name": version}


def to_dict(inventory: Inventory) -> dict:
    return {
        "root": inventory.root,
        "match_count": len(inventory.matches),
        "coverage": inventory.coverage(),
        "event_versions": [
            {"version": _version_json(v), "display": version_label(v), "matches": count}
            for v, count in inventory.event_version_coverage().items()
        ],
        "matches": {
            mid: {
                "tracking": inventory.matches[mid].tracking,
                "metadata": inventory.matches[mid].metadata,
                "rosters": inventory.matches[mid].rosters,
                "events": [
                    {"version": _version_json(v), "paths": paths}
                    for v, paths in sorted(
                        inventory.matches[mid].events.items(), key=lambda kv: _version_key(kv[0])
                    )
                ],
            }
            for mid in inventory.match_ids
        },
        "files": [
            {
                "path": f.path,
                "kind": f.kind,
                "size": f.size,
                "match_id": f.match_id,
                "event_version": _version_json(f.version) if f.kind == EVENTS else None,
                "note": f.note,
            }
            for f in inventory.files
        ],
        "specifications": [f.path for f in inventory.files_of_kind(SPECIFICATION)],
        "references": [f.path for f in inventory.files_of_kind(REFERENCE)],
        "issues": [
            {"kind": i.kind, "match_id": i.match_id, "detail": i.detail, "paths": list(i.paths)}
            for i in inventory.issues
        ],
    }


def format_report(inventory: Inventory) -> str:
    total = len(inventory.matches)
    lines = [f"Dataset root: {inventory.root}", f"Files scanned: {len(inventory.files)}"]
    lines.append(f"Match IDs discovered: {total}")
    lines.append("Coverage (matches with at least one file):")
    for modality, count in inventory.coverage().items():
        lines.append(f"  {modality:<9}{count}/{total}")
    lines.append("Event versions (no version is selected as authoritative):")
    if inventory.event_versions:
        for version, count in inventory.event_version_coverage().items():
            lines.append(f"  {version_label(version):<20}{count}/{total} matches")
    else:
        lines.append("  none")
    for label, kind in (("Specifications", SPECIFICATION), ("Global reference files", REFERENCE)):
        found = inventory.files_of_kind(kind)
        lines.append(f"{label}: {len(found)}")
        lines.extend(f"  {f.path}" for f in found)

    by_kind: dict[str, list[Issue]] = defaultdict(list)
    for issue in inventory.issues:
        by_kind[issue.kind].append(issue)
    lines.append(f"Issues: {len(inventory.issues)}")
    titles = {
        "missing": "Missing files",
        "duplicate_candidates": "Duplicate or ambiguous candidates",
        "unknown_file": "Unrecognised files",
        "skipped_entry": "Skipped entries",
    }
    for kind, title in titles.items():
        issues = by_kind.get(kind, [])
        if not issues:
            continue
        lines.append(f"  {title} ({len(issues)}):")
        for issue in issues:
            prefix = f"match {issue.match_id}: " if issue.match_id else ""
            lines.append(f"    {prefix}{issue.detail}")
            lines.extend(f"      {p}" for p in issue.paths)
    return "\n".join(lines)


def format_match(inventory: Inventory, match_id: str) -> str:
    match = inventory.matches.get(match_id)
    if match is None:
        return f"Match {match_id}: not found in {inventory.root}"
    lines = [f"Match {match_id}"]
    for modality in (TRACKING, METADATA, ROSTERS):
        paths = getattr(match, modality)
        lines.append(f"  {modality}:" + ("" if paths else " MISSING"))
        lines.extend(f"    {p}" for p in paths)
    lines.append("  events:" + ("" if match.events else " MISSING"))
    for version in inventory.event_versions:
        paths = match.events.get(version, [])
        lines.append(f"    [{version_label(version)}]" + ("" if paths else " MISSING"))
        lines.extend(f"      {p}" for p in paths)
    sizes = {f.path: f.size for f in inventory.files}
    all_paths = [p for m in (match.tracking, match.metadata, match.rosters) for p in m]
    all_paths += [p for ps in match.events.values() for p in ps]
    lines.append(f"  total size: {sum(sizes[p] for p in all_paths):,} bytes")
    issues = [i for i in inventory.issues if i.match_id == match_id]
    lines.append(f"  issues: {len(issues)}")
    lines.extend(f"    {i.kind}: {i.detail}" for i in issues)
    scan_wide = sum(i.match_id is None for i in inventory.issues)
    if scan_wide:
        lines.append(f"  scan-wide issues not tied to a match: {scan_wide} (run without --match)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m football_fatigue.inventory",
        description="Read-only inventory of the raw dataset directory (file names and sizes only).",
    )
    parser.add_argument("root", nargs="?", default="data/raw", help="raw data root (default: data/raw)")
    parser.add_argument("--match", metavar="ID", help="show details for one match ID")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    try:
        inventory = scan(args.root)
    except NotADirectoryError as error:
        print(error, file=sys.stderr)
        return 2
    except InventoryScanError as error:
        print(
            "Inventory scan FAILED; the dataset may be incomplete and no inventory was produced.",
            file=sys.stderr,
        )
        for message in error.errors:
            print(f"  {message}", file=sys.stderr)
        print("Resolve the problem(s) above and rerun.", file=sys.stderr)
        return 3

    if args.format == "json":
        data = to_dict(inventory)
        if args.match:
            data = {
                "match_id": args.match,
                "found": args.match in inventory.matches,
                **data["matches"].get(args.match, {}),
                "files": [f for f in data["files"] if f["match_id"] == args.match],
                "issues": [i for i in data["issues"] if i["match_id"] == args.match],
                "scan_wide_issue_count": sum(i["match_id"] is None for i in data["issues"]),
            }
        print(json.dumps(data, indent=2))
    elif args.match:
        print(format_match(inventory, args.match))
    else:
        print(format_report(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

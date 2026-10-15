"""Inventory tests on small synthetic directory trees (no real dataset needed).

File contents are placeholder bytes; the inventory must never read them.
"""

import builtins
import io
import json
import os
import subprocess
import sys

import pytest

from football_fatigue import inventory as inv

TS = "20000101T000000Z"
ROOT_VERSION = None  # the Event Data root folder


def touch(root, relative, content=b"x"):
    path = root.joinpath(*relative.split("/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def complete_match(root, match_id, *, ts=TS, versions=("", "V1")):
    touch(root, f"Tracking Data-{ts}-1-001/Tracking Data/{match_id}.jsonl.bz2")
    touch(root, f"Metadata-{ts}-1-001/Metadata/{match_id}.json")
    touch(root, f"Rosters-{ts}-1-001/Rosters/{match_id}.json")
    for version in versions:
        sub = f"{version}/" if version else ""
        touch(root, f"Event Data-{ts}-1-001/Event Data/{sub}{match_id}.json")


def issues(inventory, kind):
    return [i for i in inventory.issues if i.kind == kind]


def unknown_paths(inventory):
    return {f.path for f in inventory.files_of_kind(inv.UNKNOWN)}


# --- basic behaviour ----------------------------------------------------------


def test_complete_match_has_no_issues(tmp_path):
    complete_match(tmp_path, 100)
    result = inv.scan(tmp_path)
    assert result.match_ids == ["100"]
    assert result.coverage() == {"tracking": 1, "events": 1, "metadata": 1, "rosters": 1}
    assert result.issues == []
    assert set(result.matches["100"].events) == {ROOT_VERSION, "V1"}


def test_missing_modality_reported_without_crashing(tmp_path):
    complete_match(tmp_path, 100)
    touch(tmp_path, f"Metadata-{TS}-1-001/Metadata/200.json")
    result = inv.scan(tmp_path)
    assert result.match_ids == ["100", "200"]
    missing = {(i.match_id, i.detail) for i in issues(result, "missing")}
    assert ("200", "no tracking file") in missing
    assert ("200", "no rosters file") in missing
    assert ("200", "no events file in any version") in missing
    assert not [i for i in issues(result, "missing") if i.match_id == "100"]


def test_event_versions_stay_separate_and_missing_version_flagged(tmp_path):
    complete_match(tmp_path, 100, versions=("", "March 14, 2025", "May 1, 2025"))
    complete_match(tmp_path, 101, versions=("", "May 1, 2025"))
    result = inv.scan(tmp_path)
    assert result.event_versions == [ROOT_VERSION, "March 14, 2025", "May 1, 2025"]
    assert result.event_version_coverage() == {
        ROOT_VERSION: 2,
        "March 14, 2025": 1,
        "May 1, 2025": 2,
    }
    assert [i.detail for i in issues(result, "missing")] == [
        "no events file for version March 14, 2025"
    ]
    assert len(result.matches["100"].events["May 1, 2025"]) == 1
    assert "authoritative" in inv.format_report(result)


def test_split_download_parts_and_timestamps_combine(tmp_path):
    touch(tmp_path, "Tracking Data-20250101T010101Z-1-001/Tracking Data/1.jsonl.bz2")
    touch(tmp_path, "Tracking Data-20250101T010101Z-1-002/Tracking Data/2.jsonl.bz2")
    touch(tmp_path, "Event Data-20260505T050505Z-1-001/Event Data/1.json")
    touch(tmp_path, "Event Data-20260505T050505Z-1-002/Event Data/May 1, 2025/2.json")
    result = inv.scan(tmp_path)
    assert result.match_ids == ["1", "2"]
    assert result.coverage()[inv.TRACKING] == 2
    assert result.matches["1"].events == {
        ROOT_VERSION: ["Event Data-20260505T050505Z-1-001/Event Data/1.json"]
    }
    assert list(result.matches["2"].events) == ["May 1, 2025"]


def test_empty_directory(tmp_path):
    result = inv.scan(tmp_path)
    assert result.matches == {} and result.issues == [] and result.event_versions == []
    assert "Match IDs discovered: 0" in inv.format_report(result)


def test_missing_root_raises_and_cli_returns_error(tmp_path, capsys):
    with pytest.raises(NotADirectoryError):
        inv.scan(tmp_path / "absent")
    assert inv.main([str(tmp_path / "absent")]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_match_ids_sort_numerically(tmp_path):
    for match_id in (10502, 3812, 3859):
        complete_match(tmp_path, match_id)
    assert inv.scan(tmp_path).match_ids == ["3812", "3859", "10502"]


def test_unknown_and_non_match_files(tmp_path):
    complete_match(tmp_path, 100)
    touch(tmp_path, "notes.txt")
    touch(tmp_path, f"Event Data-{TS}-1-001/Event Data/readme.json")
    touch(tmp_path, "Elsewhere/100.json")
    touch(tmp_path, f"Tracking Data-{TS}-1-001/Tracking Data/100.json")
    touch(tmp_path, "competitions.csv")
    touch(tmp_path, f"Event Data-{TS}-1-001/Event Data/Spec v2.pdf")
    result = inv.scan(tmp_path)
    assert unknown_paths(result) == {
        "notes.txt",
        f"Event Data-{TS}-1-001/Event Data/readme.json",
        "Elsewhere/100.json",
        f"Tracking Data-{TS}-1-001/Tracking Data/100.json",
    }
    assert [f.path for f in result.files_of_kind(inv.REFERENCE)] == ["competitions.csv"]
    assert len(result.files_of_kind(inv.SPECIFICATION)) == 1
    assert result.match_ids == ["100"]  # unknown files never create matches
    assert len(issues(result, "unknown_file")) == 4


# --- duplicates ---------------------------------------------------------------


def test_duplicate_candidates_for_same_match_and_version(tmp_path):
    complete_match(tmp_path, 100)
    touch(tmp_path, f"Event Data-{TS}-1-002/Event Data/V1/100.json")
    touch(tmp_path, f"Tracking Data-20990101T000000Z-1-001/Tracking Data/100.jsonl.bz2")
    result = inv.scan(tmp_path)
    duplicates = issues(result, "duplicate_candidates")
    assert {i.detail for i in duplicates} == {
        "2 candidates for events version V1",
        "2 tracking candidates",
    }
    assert all(len(i.paths) == 2 for i in duplicates)
    # The root-version event file is a legitimate separate version, not a duplicate.
    assert len(result.matches["100"].events[ROOT_VERSION]) == 1


@pytest.mark.parametrize(
    ("family", "modality"), [("Metadata", "metadata"), ("Rosters", "rosters")]
)
def test_duplicate_metadata_and_roster_candidates_across_split_downloads(
    tmp_path, family, modality
):
    touch(tmp_path, f"{family}-{TS}-1-001/{family}/5.json")
    touch(tmp_path, f"{family}-{TS}-1-002/{family}/5.json")
    result = inv.scan(tmp_path)
    assert len(getattr(result.matches["5"], modality)) == 2
    [duplicate] = issues(result, "duplicate_candidates")
    assert duplicate.detail == f"2 {modality} candidates" and duplicate.match_id == "5"
    assert len(duplicate.paths) == 2


# --- directory classification -------------------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "Metadata/Event Data/123.json",
        "Event Data/Metadata/123.json",
        f"Tracking Data-{TS}-1-001/Rosters/123.jsonl.bz2",
        f"Event Data-{TS}-1-001/Event Data/Tracking Data/123.json",
    ],
)
def test_conflicting_family_directories_are_not_assigned(tmp_path, relative):
    touch(tmp_path, relative)
    result = inv.scan(tmp_path)
    assert result.matches == {}
    assert unknown_paths(result) == {relative}
    [record] = result.files
    assert "conflicting family directories" in record.note


@pytest.mark.parametrize(
    "relative",
    [
        "Elsewhere/Tracking Data/1.jsonl.bz2",
        f"Tracking Data-{TS}-1-001/Tracking Data/sub/1.jsonl.bz2",
        f"Metadata-{TS}-1-001/Metadata/Metadata/Metadata/1.json",
        f"Event Data-{TS}-1-001/Event Data/Event Data/Event Data/1.json",
        f"Event Data-{TS}-1-001/Event Data/A/B/1.json",
        f"Rosters-{TS}-1-001/Rosters/extra/1.json",
    ],
)
def test_unsupported_nesting_is_reported_not_guessed(tmp_path, relative):
    touch(tmp_path, relative)
    result = inv.scan(tmp_path)
    assert result.matches == {}
    [record] = result.files
    assert record.kind == inv.UNKNOWN and "unsupported nesting" in record.note


def test_supported_layouts(tmp_path):
    touch(tmp_path, "Tracking Data/1.jsonl.bz2")  # without Drive wrapper
    touch(tmp_path, f"Event Data-{TS}-1-001/Event Data/1.json")
    touch(tmp_path, "Event Data/Event Data/V/1.json")
    touch(tmp_path, "Event Data/W/1.json")
    result = inv.scan(tmp_path)
    assert result.issues != [] and not issues(result, "unknown_file")
    assert set(result.matches["1"].events) == {ROOT_VERSION, "V", "W"}
    assert result.matches["1"].tracking == ["Tracking Data/1.jsonl.bz2"]


# --- strict file names --------------------------------------------------------


@pytest.mark.parametrize(
    "parts",
    [
        (f"Metadata-{TS}-1-001", "Metadata", "1.json\n"),
        (f"Metadata-{TS}-1-001", "Metadata", "1.json.bak"),
        (f"Metadata-{TS}-1-001", "Metadata", "1.JSON "),
        (f"Metadata-{TS}-1-001", "Metadata", "a1.json"),
        (f"Metadata-{TS}-1-001", "Metadata", "١٢.json"),  # non-ASCII digits
        (f"Tracking Data-{TS}-1-001", "Tracking Data", "1.jsonl.bz2.tmp"),
        (f"Tracking Data-{TS}-1-001", "Tracking Data", "1.jsonl.bz2\n"),
        (f"Event Data-{TS}-1-001", "Event Data", "1 .json"),
    ],
)
def test_strict_filename_matching(parts):
    kind, match_id, _version, note = inv.classify(parts)
    assert kind == inv.UNKNOWN and match_id is None and note


def test_filename_matching_accepts_exact_names_only():
    assert inv.classify((f"Metadata-{TS}-1-001", "Metadata", "42.json"))[:2] == ("metadata", "42")
    assert inv.classify(("Tracking Data", "42.JSONL.BZ2"))[:2] == ("tracking", "42")


# --- event root version collision ---------------------------------------------


def test_root_version_distinct_from_literal_root_folder(tmp_path):
    touch(tmp_path, f"Event Data-{TS}-1-001/Event Data/7.json")
    touch(tmp_path, f"Event Data-{TS}-1-001/Event Data/(root)/7.json")
    result = inv.scan(tmp_path)
    events = result.matches["7"].events
    assert set(events) == {None, "(root)"}
    assert len(events[None]) == 1 and len(events["(root)"]) == 1
    assert not issues(result, "duplicate_candidates")

    data = json.loads(json.dumps(inv.to_dict(result)))
    versions = [e["version"] for e in data["matches"]["7"]["events"]]
    assert {"root": True, "name": None} in versions
    assert {"root": False, "name": "(root)"} in versions
    displays = {v["display"] for v in data["event_versions"]}
    assert displays == {"(root)", "'(root)' (literal folder)"}
    by_version = {
        (f["event_version"]["root"], f["event_version"]["name"]): f["path"]
        for f in data["files"]
    }
    assert by_version[(True, None)].endswith("Event Data/7.json")
    assert by_version[(False, "(root)")].endswith("(root)/7.json")


# --- JSON manifest ------------------------------------------------------------


def test_json_includes_file_level_metadata(tmp_path, capsys):
    touch(tmp_path, f"Tracking Data-{TS}-1-001/Tracking Data/5.jsonl.bz2", b"12345")
    touch(tmp_path, f"Event Data-{TS}-1-001/Event Data/May/5.json", b"123")
    touch(tmp_path, "players.csv", b"1234567")
    touch(tmp_path, "stray.txt", b"1")
    assert inv.main([str(tmp_path), "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    files = {f["path"]: f for f in data["files"]}
    tracking = files[f"Tracking Data-{TS}-1-001/Tracking Data/5.jsonl.bz2"]
    assert tracking == {
        "path": f"Tracking Data-{TS}-1-001/Tracking Data/5.jsonl.bz2",
        "kind": "tracking",
        "size": 5,
        "match_id": "5",
        "event_version": None,
        "note": None,
    }
    event = files[f"Event Data-{TS}-1-001/Event Data/May/5.json"]
    assert event["size"] == 3 and event["event_version"] == {"root": False, "name": "May"}
    assert files["players.csv"]["kind"] == "reference" and files["players.csv"]["size"] == 7
    stray = files["stray.txt"]
    assert stray["kind"] == "unknown" and stray["note"] and stray["match_id"] is None
    assert data["match_count"] == 1 and set(data["coverage"]) == set(inv.MATCH_MODALITIES)
    assert data["issues"]


def test_json_match_view_includes_files(tmp_path, capsys):
    complete_match(tmp_path, 3812)
    assert inv.main([str(tmp_path), "--format", "json", "--match", "3812"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["found"] is True and len(data["files"]) == 5  # 3 modalities + 2 event versions
    assert {e["version"]["name"] for e in data["events"]} == {None, "V1"}
    assert inv.main([str(tmp_path), "--format", "json", "--match", "9"]) == 0
    assert json.loads(capsys.readouterr().out)["found"] is False


def test_cli_text_match_view(tmp_path, capsys):
    complete_match(tmp_path, 3812)
    assert inv.main([str(tmp_path), "--match", "3812"]) == 0
    out = capsys.readouterr().out
    assert "Match 3812" in out and "[V1]" in out and "[(root)]" in out and "MISSING" not in out
    assert inv.main([str(tmp_path), "--match", "9999"]) == 0
    assert "not found" in capsys.readouterr().out


# --- filesystem errors --------------------------------------------------------


def test_enumeration_error_fails_the_scan(tmp_path, monkeypatch, capsys):
    complete_match(tmp_path, 100)
    blocked = tmp_path / f"Metadata-{TS}-1-001"
    real_walk = os.walk

    def failing_walk(top, topdown=True, onerror=None, followlinks=False):
        for current, dirs, files in real_walk(top, topdown, onerror, followlinks):
            if current == str(blocked):
                onerror(PermissionError(13, "Permission denied", str(blocked)))
                dirs[:] = []  # os.walk skips a directory it cannot list
                continue
            yield current, dirs, files

    monkeypatch.setattr(inv.os, "walk", failing_walk)
    with pytest.raises(inv.InventoryScanError) as excinfo:
        inv.scan(tmp_path)
    assert any("cannot list directory" in e and "Metadata-" in e for e in excinfo.value.errors)

    assert inv.main([str(tmp_path)]) == 3
    captured = capsys.readouterr()
    assert captured.out == ""  # no inventory is printed for an incomplete scan
    assert "FAILED" in captured.err and "Permission denied" in captured.err
    assert inv.main([str(tmp_path), "--format", "json"]) == 3
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError(2, "No such file or directory"),  # file disappeared mid-scan
        PermissionError(13, "Permission denied"),
    ],
)
def test_file_metadata_error_fails_the_scan(tmp_path, monkeypatch, capsys, error):
    complete_match(tmp_path, 100)
    real_lstat = os.lstat

    def failing_lstat(path, *args, **kwargs):
        if os.path.basename(path) == "100.jsonl.bz2":
            raise type(error)(error.errno, error.strerror, str(path))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(inv.os, "lstat", failing_lstat)
    with pytest.raises(inv.InventoryScanError) as excinfo:
        inv.scan(tmp_path)
    [message] = excinfo.value.errors
    assert "cannot read file metadata" in message and "100.jsonl.bz2" in message
    assert str(tmp_path) not in message  # path is reported relative to the root

    assert inv.main([str(tmp_path)]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and "100.jsonl.bz2" in captured.err


def test_all_scan_errors_are_collected(tmp_path, monkeypatch):
    complete_match(tmp_path, 100)
    real_lstat = os.lstat

    def failing_lstat(path, *args, **kwargs):
        if str(path).endswith(".json"):
            raise PermissionError(13, "Permission denied", str(path))
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(inv.os, "lstat", failing_lstat)
    with pytest.raises(inv.InventoryScanError) as excinfo:
        inv.scan(tmp_path)
    assert len(excinfo.value.errors) == 4  # metadata, rosters, two event versions


# --- nested download wrappers -------------------------------------------------


@pytest.mark.parametrize(
    ("family", "filename"),
    [
        ("Tracking Data", "123.jsonl.bz2"),
        ("Event Data", "123.json"),
        ("Metadata", "123.json"),
        ("Rosters", "123.json"),
    ],
)
def test_nested_download_wrappers_are_not_confident_matches(tmp_path, family, filename):
    nested = f"{family}-{TS}-1-001/{family}-20990101T000000Z-1-002/{filename}"
    nested_with_inner = f"{family}-{TS}-1-001/{family}/{family}-20990101T000000Z-1-002/{filename}"
    touch(tmp_path, nested)
    touch(tmp_path, nested_with_inner)
    result = inv.scan(tmp_path)
    assert result.matches == {} and result.coverage() == {m: 0 for m in inv.MATCH_MODALITIES}
    assert unknown_paths(result) == {nested, nested_with_inner}
    assert all("nested download wrapper" in f.note for f in result.files)
    assert {p for i in issues(result, "unknown_file") for p in i.paths} == {nested, nested_with_inner}


def test_legitimate_drive_layout_still_accepted_with_bare_inner_folder(tmp_path):
    for family, filename in (
        ("Tracking Data", "1.jsonl.bz2"),
        ("Event Data", "1.json"),
        ("Metadata", "1.json"),
        ("Rosters", "1.json"),
    ):
        touch(tmp_path, f"{family}-{TS}-1-001/{family}/{filename}")
    result = inv.scan(tmp_path)
    assert result.coverage() == {m: 1 for m in inv.MATCH_MODALITIES}
    assert not issues(result, "unknown_file")


# --- linked scan roots --------------------------------------------------------


def test_normal_directory_root_is_accepted(tmp_path):
    complete_match(tmp_path, 100)
    assert inv.scan(tmp_path).match_ids == ["100"]
    assert inv.scan(str(tmp_path) + os.sep).match_ids == ["100"]


def assert_linked_root_rejected(link, capsys):
    with pytest.raises(inv.InventoryScanError) as excinfo:
        inv.scan(link)
    assert "scan root is a symbolic link or junction" in excinfo.value.errors[0]
    assert inv.main([str(link)]) == 3
    captured = capsys.readouterr()
    assert captured.out == "" and "FAILED" in captured.err and "junction" in captured.err
    assert inv.main([str(link), "--format", "json"]) == 3
    assert capsys.readouterr().out == ""


def test_directory_symlink_root_is_rejected(tmp_path, capsys):
    complete_match(tmp_path / "real", 100)
    link = tmp_path / "link"
    make_symlink(link, tmp_path / "real", directory=True)
    assert_linked_root_rejected(link, capsys)


def test_dangling_symlink_root_is_rejected(tmp_path, capsys):
    link = tmp_path / "dangling"
    make_symlink(link, tmp_path / "missing", directory=True)
    assert_linked_root_rejected(link, capsys)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-specific")
def test_junction_root_is_rejected(tmp_path, capsys):
    complete_match(tmp_path / "real", 100)
    link = tmp_path / "junction"
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(tmp_path / "real")], capture_output=True
    )
    if done.returncode != 0:
        pytest.skip("could not create a junction")
    assert_linked_root_rejected(link, capsys)


@pytest.mark.parametrize("attribute", ["islink", "isjunction"])
def test_linked_root_rejected_without_traversal_when_simulated(
    tmp_path, monkeypatch, capsys, attribute
):
    complete_match(tmp_path, 100)

    def no_walk(*args, **kwargs):
        raise AssertionError("traversal must not start for a linked root")

    monkeypatch.setattr(inv.os.path, attribute, lambda p: True)
    monkeypatch.setattr(inv.os, "walk", no_walk)
    assert_linked_root_rejected(tmp_path, capsys)


def test_match_view_reports_scan_wide_issues(tmp_path, capsys):
    complete_match(tmp_path, 100)
    touch(tmp_path, "stray.txt")
    assert inv.main([str(tmp_path), "--match", "100"]) == 0
    assert "scan-wide issues not tied to a match: 1" in capsys.readouterr().out
    assert inv.main([str(tmp_path), "--match", "100", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["issues"] == [] and data["scan_wide_issue_count"] == 1


# --- links --------------------------------------------------------------------


def make_symlink(link, target, *, directory=False):
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


def test_file_symlink_to_external_file_is_skipped(tmp_path):
    root = tmp_path / "raw"
    external = touch(tmp_path, "external/9.jsonl.bz2")
    link = root / f"Tracking Data-{TS}-1-001" / "Tracking Data" / "9.jsonl.bz2"
    link.parent.mkdir(parents=True)
    make_symlink(link, external)
    complete_match(root, 100)
    result = inv.scan(root)
    assert result.match_ids == ["100"]
    [skipped] = issues(result, "skipped_entry")
    assert skipped.paths == (link.relative_to(root).as_posix(),)
    assert "symbolic link" in skipped.detail
    assert skipped.paths[0] not in {f.path for f in result.files}


def test_dangling_symlink_is_reported_not_crashing(tmp_path):
    root = tmp_path / "raw"
    link = root / f"Metadata-{TS}-1-001" / "Metadata" / "8.json"
    link.parent.mkdir(parents=True)
    make_symlink(link, tmp_path / "does-not-exist.json")
    result = inv.scan(root)
    assert result.matches == {} and len(issues(result, "skipped_entry")) == 1


def test_directory_symlink_is_not_followed(tmp_path):
    root = tmp_path / "raw"
    complete_match(tmp_path / "outside", 55)
    root.mkdir()
    make_symlink(root / "Linked", tmp_path / "outside", directory=True)
    complete_match(root, 100)
    result = inv.scan(root)
    assert result.match_ids == ["100"]
    skipped = issues(result, "skipped_entry")
    assert [i.paths for i in skipped] == [("Linked",)]
    assert "not followed" in skipped[0].detail


def test_link_detection_with_simulated_links(tmp_path, monkeypatch):
    """Platform-independent check of link handling, for machines that cannot symlink."""
    import stat as stat_module

    complete_match(tmp_path, 100)
    touch(tmp_path, f"Tracking Data-{TS}-1-001/Tracking Data/9.jsonl.bz2")
    touch(tmp_path, "LinkedDir/Metadata/77.json")
    real_lstat, real_islink = os.lstat, os.path.islink

    def fake_lstat(path, *args, **kwargs):
        result = real_lstat(path, *args, **kwargs)
        if os.path.basename(path) == "9.jsonl.bz2":
            fields = list(result)
            fields[0] = stat_module.S_IFLNK | 0o777
            return os.stat_result(fields)
        return result

    monkeypatch.setattr(inv.os, "lstat", fake_lstat)
    monkeypatch.setattr(
        inv.os.path, "islink", lambda p: os.path.basename(p) == "LinkedDir" or real_islink(p)
    )
    result = inv.scan(tmp_path)
    assert result.match_ids == ["100"]  # neither the linked file nor linked dir contributes
    skipped = {i.paths[0]: i.detail for i in issues(result, "skipped_entry")}
    assert set(skipped) == {f"Tracking Data-{TS}-1-001/Tracking Data/9.jsonl.bz2", "LinkedDir"}
    assert "symbolic link" in skipped[f"Tracking Data-{TS}-1-001/Tracking Data/9.jsonl.bz2"]
    assert "LinkedDir/Metadata/77.json" not in {f.path for f in result.files}


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows-specific")
def test_junction_is_not_followed(tmp_path):
    root = tmp_path / "raw"
    complete_match(tmp_path / "outside", 55)
    root.mkdir()
    done = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root / "Junction"), str(tmp_path / "outside")],
        capture_output=True,
    )
    if done.returncode != 0:
        pytest.skip("could not create a junction")
    complete_match(root, 100)
    result = inv.scan(root)
    assert result.match_ids == ["100"]
    assert [i.paths for i in issues(result, "skipped_entry")] == [("Junction",)]


# --- read-only guarantee ------------------------------------------------------


def snapshot(root):
    return sorted(
        (str(p.relative_to(root)), p.is_dir(), p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
    )


def test_scan_never_opens_or_mutates_dataset_files(tmp_path, monkeypatch):
    complete_match(tmp_path, 100)
    touch(tmp_path, "competitions.csv")
    before = snapshot(tmp_path)
    root = os.path.normcase(os.path.abspath(tmp_path))

    def inside(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        candidate = os.path.normcase(os.path.abspath(os.fsdecode(value)))
        return candidate == root or candidate.startswith(root + os.sep)

    def guard(module, name):
        real = getattr(module, name)

        def wrapper(*args, **kwargs):
            # Sources and destinations (e.g. rename/replace) are both checked.
            for value in [*args[:2], kwargs.get("path"), kwargs.get("src"), kwargs.get("dst")]:
                if inside(value):
                    raise AssertionError(f"{name} attempted on dataset path: {value!r}")
            return real(*args, **kwargs)

        monkeypatch.setattr(module, name, wrapper)

    for module, names in (
        (builtins, ["open"]),
        (io, ["open"]),
        (
            os,
            ["open", "remove", "unlink", "rename", "replace", "rmdir", "mkdir", "chmod", "utime", "truncate"],
        ),
    ):
        for name in names:
            guard(module, name)

    inv.scan(tmp_path)
    inv.format_report(inv.scan(tmp_path))
    assert inv.main([str(tmp_path), "--format", "json"]) == 0
    monkeypatch.undo()
    assert snapshot(tmp_path) == before

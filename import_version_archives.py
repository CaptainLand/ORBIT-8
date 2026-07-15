from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

from maimai_ai.simai import parse_fields


ROOT = Path(r"D:\trans\maimai_finale_dataset")
SONGS = ROOT / "songs"
MANIFEST = ROOT / "source_manifest.jsonl"
STAGING = ROOT / ".import_staging"

SOURCES = [
    (Path(r"D:\ORANGE.zip"), "ORANGE", "orange", "standard", False),
    (Path(r"D:\ORANGE PLUS.zip"), "ORANGE PLUS", "orange_plus", "standard", False),
    (Path(r"D:\PiNK.zip"), "PiNK", "pink", "standard", False),
    (Path(r"D:\PiNK PLUS.zip"), "PiNK PLUS", "pink_plus", "standard", False),
    (Path(r"D:\MiLK.zip"), "MiLK", "milk", "standard", False),
    (Path(r"D:\MiLK PLUS.zip"), "MiLK PLUS", "milk_plus", "standard", False),
    (Path(r"D:\BUDDiES.zip"), "BUDDiES", "buddies", "deluxe", True),
]


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode_maidata(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("maidata", data, 0, len(data), "unsupported text encoding")


def safe_name(value: str) -> str:
    value = re.sub(r"^\[(?:ST|DX)]\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return value[:120] or "untitled"


def package_title(package_name: str, maidata: bytes) -> tuple[str, dict[str, str]]:
    fields = parse_fields(decode_maidata(maidata))
    fallback = Path(package_name).stem
    return safe_name(fields.get("title", "").strip() or fallback), fields


def main() -> None:
    if STAGING.exists():
        raise FileExistsError(f"Remove stale staging directory before retrying: {STAGING}")
    existing = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines() if line]
    imported_keys = {(row.get("source"), row.get("source_package")) for row in existing}
    content_keys = {
        (row["files"]["maidata.txt"]["sha256"], row["files"]["track.mp3"]["sha256"])
        for row in existing
        if "maidata.txt" in row.get("files", {}) and "track.mp3" in row.get("files", {})
    }
    new_records: list[dict] = []
    report = {
        "imported": {}, "skipped_existing": [], "duplicates": [],
        "unavailable": [
            {"version": "MURASAKi", "path": r"D:\MURASAKi.zip", "reason": "invalid 69-byte download"},
            {"version": "MURASAKi PLUS", "path": r"D:\MURASAKi PLUS.zip", "reason": "missing"},
        ],
    }
    STAGING.mkdir(parents=True)
    try:
        for archive_path, version, slug, chart_type, touch_supported in SOURCES:
            if not archive_path.exists():
                report["unavailable"].append({"version": version, "path": str(archive_path), "reason": "missing"})
                continue
            version_count = 0
            with zipfile.ZipFile(archive_path) as outer:
                packages = sorted(
                    (entry for entry in outer.infolist() if not entry.is_dir()),
                    key=lambda entry: entry.filename.casefold(),
                )
                for index, package in enumerate(packages, start=1):
                    source_key = (archive_path.name, package.filename)
                    if source_key in imported_keys:
                        report["skipped_existing"].append({"source": archive_path.name, "package": package.filename})
                        continue
                    package_bytes = outer.read(package)
                    if not package_bytes.startswith(b"PK"):
                        raise ValueError(f"Inner package is not ZIP data: {archive_path.name}/{package.filename}")
                    with zipfile.ZipFile(io.BytesIO(package_bytes)) as inner:
                        entries = {Path(entry.filename).name.lower(): entry for entry in inner.infolist() if not entry.is_dir()}
                        if set(entries) != {"maidata.txt", "track.mp3", "bg.png"}:
                            raise ValueError(
                                f"Unexpected files in {archive_path.name}/{package.filename}: {sorted(entries)}"
                            )
                        payload = {name: inner.read(entry) for name, entry in entries.items()}

                    title, fields = package_title(package.filename, payload["maidata.txt"])
                    content_key = (digest_bytes(payload["maidata.txt"]), digest_bytes(payload["track.mp3"]))
                    if content_key in content_keys:
                        report["duplicates"].append({"version": version, "package": package.filename, "title": title})
                        continue

                    song_id = f"{slug}_{index:04d}"
                    song_dir_name = f"{song_id}_{title}"
                    destination = SONGS / song_dir_name
                    if destination.exists():
                        raise FileExistsError(f"Destination already exists without manifest record: {destination}")
                    staged = STAGING / song_dir_name
                    staged.mkdir()
                    for filename, data in payload.items():
                        (staged / filename).write_bytes(data)

                    relative_dir = f"songs/{song_dir_name}"
                    chart_indices = [
                        difficulty for difficulty in range(1, 8)
                        if fields.get(f"inote_{difficulty}", "").strip()
                    ]
                    record = {
                        "song_id": song_id,
                        "version": version,
                        "source": archive_path.name,
                        "source_package": package.filename,
                        "chart_type": chart_type,
                        "touch_supported": touch_supported,
                        "title": title,
                        "directory": relative_dir,
                        "chart_indices": chart_indices,
                        "files": {
                            filename: {
                                "path": f"{relative_dir}/{filename}",
                                "sha256": digest_bytes(data),
                                "bytes": len(data),
                            }
                            for filename, data in sorted(payload.items())
                        },
                    }
                    new_records.append(record)
                    content_keys.add(content_key)
                    version_count += 1
            report["imported"][version] = version_count

        for record in new_records:
            shutil.move(str(STAGING / Path(record["directory"]).name), str(SONGS / Path(record["directory"]).name))

        all_records = existing + new_records
        MANIFEST.write_text(
            "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in all_records),
            encoding="utf-8",
            newline="\n",
        )
        version_counts: dict[str, int] = {}
        for record in all_records:
            version = record.get("version", "FiNALE")
            version_counts[version] = version_counts.get(version, 0) + 1
        summary = {
            "dataset": "maimai_official_multiversion_raw",
            "song_count": len(all_records),
            "versions": version_counts,
            "prepared_snapshots": {"prepared_v1": {"source_version": "FiNALE", "song_count": 56}},
            "notes": "Raw imports only. New versions have not been added to prepared_v1 or used for training.",
        }
        (ROOT / "dataset.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        report.update({"new_records": len(new_records), "total_records": len(all_records)})
        (ROOT / "import_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))
    finally:
        shutil.rmtree(STAGING, ignore_errors=True)


if __name__ == "__main__":
    main()

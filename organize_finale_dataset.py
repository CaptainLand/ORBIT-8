from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path


SOURCE = Path(r"D:\FiNALE.zip")
OUTPUT = Path(r"D:\trans\maimai_finale_dataset")
SONGS = OUTPUT / "songs"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(name: str) -> str:
    name = re.sub(r"^\[ST\]\s*", "", name, flags=re.IGNORECASE)
    name = re.sub(r'[<>:"/\\|?*]', "_", name).strip(" .")
    return name or "untitled"


def extract_file(entry: zipfile.ZipInfo, archive: zipfile.ZipFile, destination: Path) -> None:
    if Path(entry.filename).name != entry.filename:
        raise ValueError(f"Unexpected nested path: {entry.filename}")
    with archive.open(entry) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    SONGS.mkdir(parents=True)
    records = []

    with zipfile.ZipFile(SOURCE) as outer:
        packages = sorted(
            (entry for entry in outer.infolist() if not entry.is_dir()),
            key=lambda entry: entry.filename.casefold(),
        )

        for index, package in enumerate(packages, start=1):
            title = safe_name(Path(package.filename).stem)
            song_id = f"finale_{index:04d}"
            song_dir = SONGS / f"{song_id}_{title}"
            song_dir.mkdir()

            with outer.open(package) as package_stream:
                package_bytes = io.BytesIO(package_stream.read())
            with zipfile.ZipFile(package_bytes) as inner:
                files = [entry for entry in inner.infolist() if not entry.is_dir()]
                expected = {"maidata.txt", "track.mp3", "bg.png"}
                actual = {Path(entry.filename).name for entry in files}
                if actual != expected:
                    raise ValueError(
                        f"Unexpected contents in {package.filename}: {sorted(actual)}"
                    )
                for entry in files:
                    extract_file(entry, inner, song_dir / Path(entry.filename).name)

            relative_dir = song_dir.relative_to(OUTPUT).as_posix()
            records.append(
                {
                    "song_id": song_id,
                    "source": "FiNALE.zip",
                    "source_package": package.filename,
                    "chart_type": "standard",
                    "touch_supported": False,
                    "directory": relative_dir,
                    "files": {
                        filename: {
                            "path": f"{relative_dir}/{filename}",
                            "sha256": sha256(song_dir / filename),
                            "bytes": (song_dir / filename).stat().st_size,
                        }
                        for filename in sorted(expected)
                    },
                }
            )

    with (OUTPUT / "source_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    summary = {
        "dataset": "maimai_finale_standard",
        "source_archive": str(SOURCE),
        "song_count": len(records),
        "chart_type": "standard",
        "touch_supported": False,
    }
    (OUTPUT / "dataset.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(records)} songs to {OUTPUT}")


if __name__ == "__main__":
    main()

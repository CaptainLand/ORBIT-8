from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from shutil import copy2


SOURCE_DIR = Path("2350138 penoreri - Immaculate")
OUTPUT_DIR = Path("immaculate_maimai")
OSU_FILE = SOURCE_DIR / "penoreri - Immaculate (Kozeki-Ui) [Donpa_s Resonance of Life].osu"
TICKS_PER_BEAT = 48
GRID = 192
KPS_LIMIT = 16


@dataclass
class TimingPoint:
    time_ms: float
    beat_length_ms: float
    meter: int

    @property
    def bpm(self) -> float:
        return 60000 / self.beat_length_ms


@dataclass
class Hit:
    x: int
    time_ms: int
    flags: int
    end_ms: int | None


def fmt(value: float, places: int = 6) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def read_osu(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str], list[TimingPoint], list[Hit]]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current:
            sections.setdefault(current, []).append(line)

    def pairs(section: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in sections.get(section, []):
            if ":" in line:
                key, value = line.split(":", 1)
                result[key.strip()] = value.strip()
        return result

    timing: list[TimingPoint] = []
    for line in sections["TimingPoints"]:
        parts = line.split(",")
        if len(parts) >= 7 and parts[6] == "1" and float(parts[1]) > 0:
            timing.append(TimingPoint(float(parts[0]), float(parts[1]), int(parts[2])))
    timing.sort(key=lambda p: p.time_ms)

    hits: list[Hit] = []
    for line in sections["HitObjects"]:
        parts = line.split(",")
        flags = int(parts[3])
        end_ms = None
        if flags & 128:
            end_ms = int(float(parts[5].split(":", 1)[0]))
        hits.append(Hit(int(float(parts[0])), int(float(parts[2])), flags, end_ms))
    hits.sort(key=lambda h: (h.time_ms, h.x))
    return pairs("General"), pairs("Metadata"), pairs("Difficulty"), timing, hits


def beat_at(timing: list[TimingPoint], time_ms: float) -> float:
    beat = 0.0
    for i, point in enumerate(timing):
        next_time = timing[i + 1].time_ms if i + 1 < len(timing) else None
        if time_ms < point.time_ms:
            break
        if next_time is None or time_ms < next_time:
            return beat + (time_ms - point.time_ms) / point.beat_length_ms
        beat += (next_time - point.time_ms) / point.beat_length_ms
    first = timing[0]
    return (time_ms - first.time_ms) / first.beat_length_ms


def active_timing(timing: list[TimingPoint], time_ms: float) -> TimingPoint:
    active = timing[0]
    for point in timing:
        if point.time_ms <= time_ms:
            active = point
        else:
            break
    return active


def lane(hit: Hit) -> int:
    return min(3, max(0, int(hit.x * 4 / 512)))


def is_hold(hit: Hit) -> bool:
    return bool(hit.flags & 128 and hit.end_ms and hit.end_ms > hit.time_ms)


def button_distance(a: int, b: int) -> int:
    d = abs(a - b)
    return min(d, 8 - d)


def count_notes(token: str | None) -> int:
    if not token:
        return 0
    return sum(1 for part in token.split("/") if re.match(r"^[1-8]", part))


def first_note_button(token: str) -> int | None:
    for part in token.split("/"):
        if re.match(r"^[1-8]", part):
            return int(part[0])
    return None


def slide(button: int, phrase: int) -> str:
    if phrase % 4 == 0:
        end = button + 4
        if end > 8:
            end -= 8
        return f"{button}-{end}[8:1]"
    if phrase % 4 == 1:
        end = button + 2
        if end > 8:
            end -= 8
        return f"{button}>{end}[8:1]"
    if phrase % 4 == 2:
        end = button - 4
        if end < 1:
            end += 8
        return f"{button}-{end}[8:1]"
    end = button - 2
    if end < 1:
        end += 8
    return f"{button}<{end}[8:1]"


def source_pattern_token(
    time_ms: int,
    tick: int,
    interval: int | None,
    lanes: list[int],
    has_hold: bool,
    phrase: int,
    event_index: int,
    prev_button: int | None,
    can_hold: bool,
) -> tuple[str | None, int | None, int | None, bool]:
    multiplicity = len(lanes)
    lane_hint = lanes[0] if lanes else 0

    # Preserve original short bursts as natural maimai gestures.
    if interval is not None and interval <= 8:
        sweeps = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
            [3, 4, 5, 6, 7, 8, 1, 2],
            [6, 5, 4, 3, 2, 1, 8, 7],
        ]
        pattern = sweeps[phrase % len(sweeps)]
        button = pattern[event_index % len(pattern)]
        return str(button), button, None, True

    if interval is not None and interval <= 12:
        pairs = [(3, 4), (5, 6), (2, 3), (6, 7), (4, 5), (1, 2), (7, 8)]
        pair = pairs[(phrase + event_index // 16) % len(pairs)]
        button = pair[event_index % 2]
        return str(button), button, None, False

    if has_hold and can_hold:
        hold_buttons = [4, 5, 3, 6, 2, 7]
        button = hold_buttons[(phrase + lane_hint + event_index) % len(hold_buttons)]
        return f"{button}h[#0.294]", button, tick + 48, False

    if tick % GRID == 0 and phrase >= 1 and event_index % 7 == 0:
        buttons = [1, 8, 2, 7, 3, 6, 4, 5]
        button = buttons[(phrase + event_index) % len(buttons)]
        return slide(button, phrase), button, None, True

    if multiplicity >= 3:
        pairs = [(3, 6), (4, 5), (2, 7), (1, 8)]
        pair = pairs[(phrase + lane_hint + event_index) % len(pairs)]
        return f"{pair[0]}/{pair[1]}", None, None, False

    if multiplicity == 2:
        source_pairs = {
            (0, 1): (3, 4),
            (1, 2): (4, 5),
            (2, 3): (5, 6),
            (0, 2): (3, 6),
            (1, 3): (2, 7),
            (0, 3): (1, 8),
        }
        pair = source_pairs.get(tuple(sorted(lanes)), (3, 6))
        return f"{pair[0]}/{pair[1]}", None, None, False

    if interval is not None and interval <= 16:
        rolls = [[3, 4, 5, 4], [5, 6, 5, 4], [2, 3, 4, 5], [7, 6, 5, 4]]
        pattern = rolls[(phrase + event_index // 12) % len(rolls)]
        button = pattern[event_index % len(pattern)]
        return str(button), button, None, False

    melodic = [
        [3, 4, 5, 6, 5, 4],
        [2, 3, 4, 5, 6, 7, 6, 5],
        [6, 5, 4, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7, 8, 7, 6],
    ][phrase % 4]
    button = melodic[(event_index + lane_hint) % len(melodic)]
    if prev_button is not None and button_distance(prev_button, button) > 3:
        button = min(melodic, key=lambda b: (button_distance(prev_button, b), b))
    return str(button), button, None, False


def render() -> str:
    _, metadata, _, timing, hits = read_osu(OSU_FILE)
    anchor_ms = min(hit.time_ms for hit in hits)
    anchor_beat = beat_at(timing, anchor_ms)
    grouped: dict[int, list[Hit]] = defaultdict(list)
    ms_by_tick: dict[int, int] = {}
    for hit in hits:
        tick = round((beat_at(timing, hit.time_ms) - anchor_beat) * TICKS_PER_BEAT)
        if tick < 0:
            continue
        grouped[tick].append(hit)
        ms_by_tick[tick] = hit.time_ms

    bpm_events: dict[int, str] = {0: f"({fmt(active_timing(timing, anchor_ms).bpm)})"}
    for point in timing:
        if point.time_ms > anchor_ms:
            tick = round((beat_at(timing, point.time_ms) - anchor_beat) * TICKS_PER_BEAT)
            bpm_events[tick] = f"({fmt(point.bpm)})"

    max_tick = max(max(grouped), max(bpm_events))
    emitted: deque[tuple[int, int]] = deque()
    prev_button: int | None = None
    prev_event_tick: int | None = None
    event_index = 0
    active_hold_until = -1
    last_hold_start = -999

    cells = [f"{bpm_events[0]}{{{GRID}}}"]
    for tick in range(max_tick + 1):
        parts: list[str] = []
        if tick in bpm_events and tick != 0:
            parts.append(bpm_events[tick])
        if tick in grouped:
            time_ms = ms_by_tick[tick]
            event_hits = sorted(grouped[tick], key=lane)
            event_lanes = [lane(hit) for hit in event_hits]
            phrase = int((anchor_beat + tick / TICKS_PER_BEAT) // 32)
            interval = None if prev_event_tick is None else tick - prev_event_tick
            can_hold = (
                any(is_hold(hit) for hit in event_hits)
                and tick >= active_hold_until + 24
                and tick - last_hold_start >= 144
                and tick % 96 == 0
            )
            token, new_prev, new_hold_until, sweep_exempt = source_pattern_token(
                time_ms,
                tick,
                interval,
                event_lanes,
                any(is_hold(hit) for hit in event_hits),
                phrase,
                event_index,
                prev_button,
                can_hold,
            )
            while emitted and time_ms - emitted[0][0] >= 1000:
                emitted.popleft()
            current_kps = sum(count for _, count in emitted)
            token_count = 0 if sweep_exempt else count_notes(token)
            if current_kps + token_count > KPS_LIMIT and token and "/" in token:
                token = token.split("/", 1)[0]
                token_count = count_notes(token)
            if current_kps + token_count > KPS_LIMIT:
                token = None
                token_count = 0
            if token:
                parts.append(token)
                emitted.append((time_ms, token_count))
                if new_prev is not None:
                    prev_button = new_prev
                    first = first_note_button(token)
                    if first is not None:
                        prev_button = first
                if new_hold_until is not None:
                    active_hold_until = new_hold_until
                    last_hold_start = tick
            prev_event_tick = tick
            event_index += 1

        if tick == 0:
            cells[0] += "".join(parts)
        else:
            cells.append("".join(parts))

    lines: list[str] = []
    row: list[str] = []
    for cell in cells:
        row.append(cell)
        if len(row) >= 96:
            lines.append(",".join(row) + ",")
            row = []
    if row:
        lines.append(",".join(row) + ",")

    title = metadata.get("TitleUnicode") or metadata.get("Title") or "Immaculate"
    artist = metadata.get("Artist") or "penoreri"
    return "\n".join(
        [
            f"&title={title}",
            f"&artist={artist}",
            f"&first={fmt(anchor_ms / 1000)}",
            "&des_5=Codex source-aligned handwrite v2",
            "&lv_5=14.0",
            "&inote_5=",
            *lines,
            "E",
        ]
    ) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "maidata.txt").write_text(render(), encoding="utf-8")
    copy2(SOURCE_DIR / "audio.mp3", OUTPUT_DIR / "track.mp3")
    copy2(SOURCE_DIR / "bg.jpg", OUTPUT_DIR / "bg.jpg")
    for item in OUTPUT_DIR.iterdir():
        if item.name not in {"maidata.txt", "track.mp3", "bg.jpg"}:
            if item.is_file():
                item.unlink()


if __name__ == "__main__":
    main()

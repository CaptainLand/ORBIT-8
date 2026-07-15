from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2


SOURCE_DIR = Path("2350138 penoreri - Immaculate")
OUTPUT_DIR = Path("immaculate_maimai")
TICKS_PER_BEAT = 48
DIVISION = TICKS_PER_BEAT * 4
KPS_LIMIT = 16


@dataclass
class TimingPoint:
    time_ms: float
    beat_length_ms: float
    meter: int

    @property
    def bpm(self) -> float:
        return 60000.0 / self.beat_length_ms


@dataclass
class HitObject:
    x: int
    time_ms: int
    type_flags: int
    end_time_ms: int | None


@dataclass
class OsuChart:
    path: Path
    general: dict[str, str]
    metadata: dict[str, str]
    difficulty: dict[str, str]
    timing_points: list[TimingPoint]
    hit_objects: list[HitObject]


def fmt_float(value: float, places: int = 6) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def read_osu(path: Path) -> OsuChart:
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

    timing_points: list[TimingPoint] = []
    for line in sections.get("TimingPoints", []):
        parts = line.split(",")
        if len(parts) >= 7 and parts[6] == "1" and float(parts[1]) > 0:
            timing_points.append(TimingPoint(float(parts[0]), float(parts[1]), int(parts[2])))
    timing_points.sort(key=lambda point: point.time_ms)
    if not timing_points:
        raise ValueError(f"{path} has no uninherited timing points")

    hit_objects: list[HitObject] = []
    for line in sections.get("HitObjects", []):
        parts = line.split(",")
        if len(parts) < 5:
            continue
        flags = int(parts[3])
        end_time = None
        if flags & 128 and len(parts) >= 6:
            end_time = int(float(parts[5].split(":", 1)[0]))
        hit_objects.append(HitObject(int(float(parts[0])), int(float(parts[2])), flags, end_time))
    hit_objects.sort(key=lambda obj: (obj.time_ms, obj.x))

    return OsuChart(path, pairs("General"), pairs("Metadata"), pairs("Difficulty"), timing_points, hit_objects)


def beat_at_ms(chart: OsuChart, time_ms: float) -> float:
    beat = 0.0
    for index, point in enumerate(chart.timing_points):
        next_time = chart.timing_points[index + 1].time_ms if index + 1 < len(chart.timing_points) else None
        if time_ms < point.time_ms:
            break
        if next_time is None or time_ms < next_time:
            return beat + (time_ms - point.time_ms) / point.beat_length_ms
        beat += (next_time - point.time_ms) / point.beat_length_ms
    first = chart.timing_points[0]
    return (time_ms - first.time_ms) / first.beat_length_ms


def active_timing_point(chart: OsuChart, time_ms: float) -> TimingPoint:
    active = chart.timing_points[0]
    for point in chart.timing_points:
        if point.time_ms <= time_ms:
            active = point
        else:
            break
    return active


def tick_from_anchor(chart: OsuChart, time_ms: float, anchor_ms: float) -> int:
    return round((beat_at_ms(chart, time_ms) - beat_at_ms(chart, anchor_ms)) * TICKS_PER_BEAT)


def lane(chart: OsuChart, obj: HitObject) -> int:
    keys = int(float(chart.difficulty.get("CircleSize", "4")))
    if keys != 4:
        raise ValueError(f"{chart.path.name} is {keys}K, expected 4K")
    return min(keys - 1, max(0, int(obj.x * keys / 512)))


def is_hold(obj: HitObject) -> bool:
    return bool(obj.type_flags & 128 and obj.end_time_ms and obj.end_time_ms > obj.time_ms)


def hold_duration(obj: HitObject) -> float:
    if obj.end_time_ms and obj.end_time_ms > obj.time_ms:
        return max(0.12, min((obj.end_time_ms - obj.time_ms) / 1000.0, 2.4))
    return 0.0


def phrase(beat: float) -> int:
    return int(max(0.0, beat) // 32)


def step(tick: int) -> int:
    return tick // 12


def pick(pattern: list[int], tick: int, lane_id: int, event_index: int) -> int:
    return pattern[(step(tick) + lane_id + event_index) % len(pattern)]


def chord(*buttons: int) -> str:
    return "/".join(str(button) for button in buttons)


def note_count(token: str | None) -> int:
    if not token:
        return 0
    return sum(1 for part in token.split("/") if part and part[0].isdigit())


def soften_token_for_kps(token: str | None) -> str | None:
    if not token or "/" not in token:
        return token
    return token.split("/", 1)[0]


def token_has_hold(token: str | None) -> bool:
    return bool(token and "h[" in token)


def make_star_slide(button: int, current_phrase: int) -> str:
    if current_phrase % 4 == 0:
        end = button + 4
        if end > 8:
            end -= 8
        return f"{button}-{end}[8:1]"
    if current_phrase % 4 == 1:
        end = button + 2
        if end > 8:
            end -= 8
        return f"{button}>{end}[8:1]"
    if current_phrase % 4 == 2:
        end = button - 4
        if end < 1:
            end += 8
        return f"{button}-{end}[8:1]"
    end = button - 2
    if end < 1:
        end += 8
    return f"{button}<{end}[8:1]"


def button_distance(a: int, b: int) -> int:
    raw = abs(a - b)
    return min(raw, 8 - raw)


def gentle_button(pattern: list[int], tick: int, lane_id: int, event_index: int, previous_button: int | None) -> int:
    candidate = pick(pattern, tick, lane_id, event_index)
    if previous_button is None or button_distance(previous_button, candidate) <= 2:
        return candidate
    nearby = sorted(pattern, key=lambda button: (button_distance(previous_button, button), button))
    return nearby[0]


def hold_button_for(style: str, tick: int, lane_id: int, event_index: int, previous_button: int | None) -> int:
    if style == "lyric":
        pattern = [4, 5, 3, 6]
    elif style == "flow":
        pattern = [3, 6, 4, 5, 2, 7]
    else:
        pattern = [2, 7, 3, 6, 4, 5]
    return gentle_button(pattern, tick, lane_id, event_index, previous_button)


def interact_button(held_button: int, previous_button: int | None) -> int:
    candidates = {
        1: [5, 6, 4],
        2: [6, 5, 7],
        3: [7, 6, 4],
        4: [8, 7, 3],
        5: [1, 2, 6],
        6: [2, 1, 7],
        7: [3, 2, 8],
        8: [4, 3, 1],
    }[held_button]
    if previous_button is None:
        return candidates[0]
    non_repeating = [button for button in candidates if button != previous_button]
    return sorted(non_repeating or candidates, key=lambda button: (button_distance(previous_button, button), button))[0]


def run_button(style: str, interval: int | None, current_phrase: int, event_index: int, previous_button: int | None) -> int | None:
    if interval is None:
        return None
    if interval <= 8:
        sweep_patterns = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
            [3, 4, 5, 6, 7, 8, 1, 2],
            [6, 5, 4, 3, 2, 1, 8, 7],
        ]
        pattern = sweep_patterns[current_phrase % len(sweep_patterns)]
        return pattern[event_index % len(pattern)]
    if interval <= 12:
        interaction_pairs = {
            "lyric": [(3, 4), (5, 6), (4, 5)],
            "flow": [(3, 4), (5, 6), (2, 3), (6, 7), (4, 5)],
            "master": [(3, 4), (5, 6), (2, 3), (6, 7), (1, 2), (7, 8), (4, 5)],
        }[style]
        pair = interaction_pairs[(current_phrase + event_index // 16) % len(interaction_pairs)]
        return pair[event_index % 2]
    if interval <= 16:
        rolls = {
            "lyric": [[3, 4, 5, 4], [6, 5, 4, 5]],
            "flow": [[2, 3, 4, 5], [7, 6, 5, 4], [3, 4, 5, 6]],
            "master": [[1, 2, 3, 4], [8, 7, 6, 5], [3, 4, 5, 6], [6, 5, 4, 3]],
        }[style]
        pattern = rolls[(current_phrase + event_index // 12) % len(rolls)]
        return pattern[event_index % len(pattern)]
    return None


def designed_token(
    chart: OsuChart,
    objects: list[HitObject],
    tick: int,
    beat: float,
    style: str,
    event_index: int,
    previous_button: int | None,
    active_hold_button: int | None,
    allow_hold_interact: bool,
    can_start_hold: bool,
    previous_event_interval: int | None,
) -> tuple[str | None, int | None, int | None]:
    objects = sorted(objects, key=lambda obj: (lane(chart, obj), obj.time_ms))
    lead = objects[0]
    lane_id = lane(chart, lead)
    source_holds = [obj for obj in objects if is_hold(obj)]
    hold_seconds = max((hold_duration(obj) for obj in source_holds), default=0.0)
    current_phrase = phrase(beat)
    strong = tick % DIVISION == 0
    half = tick % (DIVISION // 2) == 0
    dense = len(objects) >= 2
    star_moment = (
        style in {"flow", "master"}
        and not dense
        and not hold_seconds
        and active_hold_button is None
        and strong
        and current_phrase >= 2
        and event_index % (10 if style == "flow" else 8) == 0
    )

    if active_hold_button is not None:
        if not allow_hold_interact:
            return None, previous_button, None
        button = interact_button(active_hold_button, previous_button)
        return str(button), button, None

    if hold_seconds and can_start_hold:
        button = hold_button_for(style, tick, lane_id, event_index, previous_button)
        return f"{button}h[#{fmt_float(hold_seconds, 3)}]", button, button

    fast_button = run_button(style, previous_event_interval, current_phrase, event_index, previous_button)
    if fast_button is not None and not dense and not strong:
        return str(fast_button), fast_button, None

    if style == "lyric":
        patterns = [
            [3, 4, 5, 6, 5, 4],
            [2, 3, 4, 5, 6, 7, 6, 5],
            [6, 5, 4, 3, 4, 5],
            [1, 2, 3, 4, 5, 6, 7, 8, 7, 6],
        ]
        button = gentle_button(patterns[current_phrase % len(patterns)], tick, lane_id, event_index, previous_button)
        if star_moment:
            return make_star_slide(button, current_phrase), button, None
        if dense and strong:
            pair = [(3, 6), (4, 5), (2, 7)][current_phrase % 3]
            return chord(*pair), None, None
        return str(button), button, None

    if style == "flow":
        patterns = [
            [1, 2, 3, 4, 5, 6, 7, 8],
            [8, 7, 6, 5, 4, 3, 2, 1],
            [3, 4, 5, 6, 5, 4, 3, 2],
            [6, 5, 4, 3, 4, 5, 6, 7],
        ]
        button = gentle_button(patterns[current_phrase % len(patterns)], tick, lane_id, event_index, previous_button)
        if star_moment:
            return make_star_slide(button, current_phrase), button, None
        if dense or strong:
            pair = [(2, 7), (3, 6), (4, 5), (1, 8), (3, 5), (4, 6)][(current_phrase + event_index + lane_id) % 6]
            return chord(*pair), None, None
        return str(button), button, None

    patterns = [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [8, 7, 6, 5, 4, 3, 2, 1],
        [2, 3, 4, 5, 6, 7, 6, 5, 4, 3],
        [7, 6, 5, 4, 3, 2, 3, 4, 5, 6],
        [3, 4, 5, 6, 5, 4, 2, 3, 4, 5],
    ]
    button = gentle_button(patterns[current_phrase % len(patterns)], tick, lane_id, event_index, previous_button)
    if star_moment:
        return make_star_slide(button, current_phrase), button, None
    if dense or strong:
        pair = [(1, 8), (2, 7), (3, 6), (4, 5), (2, 5), (4, 7), (3, 5), (4, 6)][(current_phrase + event_index + lane_id) % 8]
        return chord(*pair), None, None
    return str(button), button, None


def bpm_token(point: TimingPoint) -> str:
    return f"({fmt_float(point.bpm, 6)})"


def render_chart(chart: OsuChart, anchor_ms: float, style: str) -> str:
    events: dict[int, list[HitObject]] = {}
    bpm_events: dict[int, str] = {0: bpm_token(active_timing_point(chart, anchor_ms))}
    max_tick = 0

    for point in chart.timing_points:
        if point.time_ms > anchor_ms:
            tick = tick_from_anchor(chart, point.time_ms, anchor_ms)
            bpm_events[tick] = bpm_token(point)
            max_tick = max(max_tick, tick)

    for obj in chart.hit_objects:
        tick = tick_from_anchor(chart, obj.time_ms, anchor_ms)
        if tick < 0:
            continue
        events.setdefault(tick, []).append(obj)
        max_tick = max(max_tick, tick)
        if obj.end_time_ms:
            max_tick = max(max_tick, tick_from_anchor(chart, obj.end_time_ms, anchor_ms))

    cells: list[str] = [f"{bpm_events[0]}{{{DIVISION}}}"]
    event_index = 0
    anchor_beat = beat_at_ms(chart, anchor_ms)
    previous_button: int | None = None
    active_hold_until = -1
    active_hold_button: int | None = None
    last_hold_interact_tick = -999
    last_hold_start_tick = -999
    min_hold_gap = {"lyric": 192, "flow": 144, "master": 120}[style]
    previous_event_tick: int | None = None
    emitted_notes: list[tuple[float, int]] = []
    absolute_anchor_ms = anchor_ms

    for tick in range(0, max_tick + 1):
        parts: list[str] = []
        if tick in bpm_events and tick != 0:
            parts.append(bpm_events[tick])
        if tick >= active_hold_until:
            active_hold_button = None
        if tick in events:
            previous_event_interval = None if previous_event_tick is None else tick - previous_event_tick
            allow_hold_interact = tick - last_hold_interact_tick >= 48
            has_source_hold = any(is_hold(obj) for obj in events[tick])
            can_start_hold = (
                has_source_hold
                and active_hold_button is None
                and tick - active_hold_until >= 24
                and tick - last_hold_start_tick >= min_hold_gap
                and (tick % 96 == 0 or tick % 192 == 0)
            )
            token, previous_button, new_hold_button = designed_token(
                chart,
                events[tick],
                tick,
                anchor_beat + tick / TICKS_PER_BEAT,
                style,
                event_index,
                previous_button,
                active_hold_button,
                allow_hold_interact,
                can_start_hold,
                previous_event_interval,
            )
            if token:
                current_ms = absolute_anchor_ms + (tick / TICKS_PER_BEAT) * (60000 / active_timing_point(chart, anchor_ms).bpm)
                # Recompute with chart beats so BPM changes do not distort the KPS window.
                current_beat = anchor_beat + tick / TICKS_PER_BEAT
                current_ms = anchor_ms
                for point_index, point in enumerate(chart.timing_points):
                    next_point = chart.timing_points[point_index + 1] if point_index + 1 < len(chart.timing_points) else None
                    point_beat = beat_at_ms(chart, point.time_ms)
                    next_beat = beat_at_ms(chart, next_point.time_ms) if next_point else None
                    if current_beat >= point_beat and (next_beat is None or current_beat < next_beat):
                        current_ms = point.time_ms + (current_beat - point_beat) * point.beat_length_ms
                        break
                emitted_notes = [(ms, count) for ms, count in emitted_notes if current_ms - ms < 1000]
                current_count = sum(count for _, count in emitted_notes)
                candidate_count = note_count(token)
                if current_count + candidate_count > KPS_LIMIT and not token_has_hold(token):
                    token = soften_token_for_kps(token)
                    candidate_count = note_count(token)
                if current_count + candidate_count > KPS_LIMIT:
                    token = None
                    candidate_count = 0
            if token:
                parts.append(token)
                emitted_notes.append((current_ms, candidate_count))
                if active_hold_button is not None:
                    last_hold_interact_tick = tick
            if new_hold_button is not None:
                hold_ends = [
                    tick_from_anchor(chart, obj.end_time_ms, anchor_ms)
                    for obj in events[tick]
                    if is_hold(obj) and obj.end_time_ms is not None
                ]
                if hold_ends:
                    active_hold_until = min(max(hold_ends), tick + {"lyric": 72, "flow": 96, "master": 96}[style])
                    active_hold_button = new_hold_button
                    last_hold_start_tick = tick
            previous_event_tick = tick
            event_index += 1
        if tick == 0:
            cells[0] += "".join(parts)
        else:
            cells.append("".join(parts))

    lines: list[str] = []
    line: list[str] = []
    for cell in cells:
        line.append(cell)
        if len(line) >= 96:
            lines.append(",".join(line) + ",")
            line = []
    if line:
        lines.append(",".join(line) + ",")
    return "\n".join(lines)


def write_combined(charts: list[OsuChart]) -> None:
    anchor_ms = min(chart.hit_objects[0].time_ms for chart in charts if chart.hit_objects)
    base = charts[0]
    title = base.metadata.get("TitleUnicode") or base.metadata.get("Title") or "Immaculate"
    artist = base.metadata.get("ArtistUnicode") or base.metadata.get("Artist") or ""
    slots = [
        (3, "lyric", "12.4", charts[0]),
        (4, "flow", "13.2", charts[1]),
        (5, "master", "14.2", charts[2]),
    ]

    lines = [
        f"&title={title}",
        f"&artist={artist}",
        f"&first={fmt_float(anchor_ms / 1000.0, 6)}",
    ]

    for slot, style, level, chart in slots:
        version = chart.metadata.get("Version", chart.path.stem)
        creator = chart.metadata.get("Creator", "")
        lines.extend(
            [
                f"&des_{slot}={creator} / Codex {style} rewrite from {version}",
                f"&lv_{slot}={level}",
                f"&inote_{slot}=",
                render_chart(chart, anchor_ms, style),
            ]
        )

    lines.append("E")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "maidata.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def clean_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keep = {"maidata.txt", "track.mp3", "bg.jpg"}
    for item in OUTPUT_DIR.iterdir():
        if item.name not in keep:
            if item.is_dir():
                for child in sorted(item.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                item.rmdir()
            else:
                item.unlink()


def main() -> None:
    charts = [
        read_osu(SOURCE_DIR / "penoreri - Immaculate (Kozeki-Ui) [Acoustic].osu"),
        read_osu(SOURCE_DIR / "penoreri - Immaculate (Kozeki-Ui) [Resurrection].osu"),
        read_osu(SOURCE_DIR / "penoreri - Immaculate (Kozeki-Ui) [Donpa_s Resonance of Life].osu"),
    ]
    clean_output_dir()
    write_combined(charts)
    copy2(SOURCE_DIR / "audio.mp3", OUTPUT_DIR / "track.mp3")
    copy2(SOURCE_DIR / "bg.jpg", OUTPUT_DIR / "bg.jpg")
    print(f"Wrote {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

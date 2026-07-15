from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any


TICKS_PER_MEASURE = 192
FIELD_RE = re.compile(r"(?m)^&([^=\r\n]+)=(.*)$")
CONTROL_RE = re.compile(r"\(([-+]?\d+(?:\.\d+)?)\)|\{(\d+)\}")
HEAD_RE = re.compile(r"^([1-8])([bx]*)", re.IGNORECASE)
BRANCH_RE = re.compile(r"^(pp|qq|[<>^vVpqszw-])([1-8]{1,2})\[([^]]+)\]([bx]*)$", re.IGNORECASE)
CHAIN_PART_RE = re.compile(r"(pp|qq|[<>^vVpqszw-])([1-8]{1,2})")
TOUCH_RE = re.compile(r"^(?:[ABDE][1-8]|C)", re.IGNORECASE)


class SimaiParseError(ValueError):
    pass


@dataclass
class Duration:
    raw: str
    mode: str
    duration_ticks: int | None = None
    duration_seconds: float | None = None
    wait_ticks: int | None = None
    wait_seconds: float | None = None


@dataclass
class SlideBranch:
    operator: str
    path_lanes: list[int]
    end_lane: int
    duration: Duration


@dataclass
class NoteEvent:
    tick: int
    time_ms: float
    slot: int
    lane: int
    note_type: str
    is_break: bool = False
    is_ex: bool = False
    duration: Duration | None = None
    branches: list[SlideBranch] = field(default_factory=list)
    raw: str = ""


@dataclass
class ParsedChart:
    difficulty_index: int
    level_text: str
    designer: str
    first_ms: float
    default_bpm: float
    events: list[NoteEvent]
    tick_times_ms: list[float]
    bpm_changes: list[dict[str, float | int]]
    division_changes: list[dict[str, int]]

    @property
    def total_ticks(self) -> int:
        return len(self.tick_times_ms) - 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["total_ticks"] = self.total_ticks
        return result


def parse_fields(text: str) -> dict[str, str]:
    matches = list(FIELD_RE.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        start = match.start(2)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[match.group(1).strip()] = text[start:end].strip()
    return fields


def _ratio_ticks(value: str) -> int | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", value.strip())
    if not match:
        return None
    denominator = Fraction(match.group(1))
    numerator = Fraction(match.group(2))
    ticks = Fraction(TICKS_PER_MEASURE) * numerator / denominator
    if ticks.denominator != 1:
        raise SimaiParseError(f"Duration does not land on the {TICKS_PER_MEASURE}-tick grid: {value}")
    return int(ticks)


def parse_duration(raw: str, bpm: float) -> Duration:
    value = raw.strip()
    if "##" in value:
        wait_raw, duration_raw = (part.strip() for part in value.split("##", 1))
        wait_ticks = _ratio_ticks(wait_raw)
        duration_ticks = _ratio_ticks(duration_raw)
        wait_seconds = None if wait_ticks is not None else float(wait_raw)
        duration_seconds = None if duration_ticks is not None else float(duration_raw)
        if wait_ticks is None:
            wait_ticks = round(wait_seconds * bpm * TICKS_PER_MEASURE / 240.0)
        if duration_ticks is None:
            duration_ticks = round(duration_seconds * bpm * TICKS_PER_MEASURE / 240.0)
        return Duration(
            raw=value,
            mode="explicit",
            duration_ticks=duration_ticks,
            duration_seconds=duration_seconds,
            wait_ticks=wait_ticks,
            wait_seconds=wait_seconds,
        )

    ratio_ticks = _ratio_ticks(value)
    if ratio_ticks is not None:
        return Duration(raw=value, mode="ratio", duration_ticks=ratio_ticks)

    seconds = float(value.removeprefix("#"))
    ticks = round(seconds * bpm * TICKS_PER_MEASURE / 240.0)
    return Duration(raw=value, mode="seconds", duration_ticks=ticks, duration_seconds=seconds)


def parse_component(
    component: str,
    *,
    tick: int,
    time_ms: float,
    slot: int,
    bpm: float,
) -> NoteEvent:
    component = component.strip()
    if TOUCH_RE.match(component):
        raise SimaiParseError(f"Touch note is outside the FiNALE schema: {component}")

    head = HEAD_RE.match(component)
    if not head:
        raise SimaiParseError(f"Unknown note component: {component}")

    lane = int(head.group(1))
    modifiers = head.group(2).lower()
    rest = component[head.end():]
    common = {
        "tick": tick,
        "time_ms": round(time_ms, 6),
        "slot": slot,
        "lane": lane,
        "is_break": "b" in modifiers,
        "is_ex": "x" in modifiers,
        "raw": component,
    }

    if not rest:
        return NoteEvent(note_type="tap", **common)

    hold = re.fullmatch(r"h([bx]*)\[([^]]+)\]", rest, re.IGNORECASE)
    if hold:
        suffix_modifiers = hold.group(1).lower()
        return NoteEvent(
            note_type="hold",
            duration=parse_duration(hold.group(2), bpm),
            **{
                **common,
                "is_break": common["is_break"] or "b" in suffix_modifiers,
                "is_ex": common["is_ex"] or "x" in suffix_modifiers,
            },
        )

    branches = []
    slide_suffix_modifiers = ""
    for branch_text in rest.split("*"):
        branch = BRANCH_RE.fullmatch(branch_text)
        if not branch:
            chain = re.fullmatch(r"(.+)\[([^]]+)\]([bx]*)", branch_text, re.IGNORECASE)
            if chain:
                parts = list(CHAIN_PART_RE.finditer(chain.group(1)))
                if parts and "".join(part.group(0) for part in parts) == chain.group(1):
                    path_lanes = [int(value) for part in parts for value in part.group(2)]
                    branches.append(
                        SlideBranch(
                            operator=parts[0].group(1),
                            path_lanes=path_lanes,
                            end_lane=path_lanes[-1],
                            duration=parse_duration(chain.group(2), bpm),
                        )
                    )
                    slide_suffix_modifiers += chain.group(3).lower()
                    continue
            raise SimaiParseError(f"Unknown slide branch: {component} -> {branch_text}")
        operator = branch.group(1)
        path_lanes = [int(value) for value in branch.group(2)]
        branches.append(
            SlideBranch(
                operator=operator,
                path_lanes=path_lanes,
                end_lane=path_lanes[-1],
                duration=parse_duration(branch.group(3), bpm),
            )
        )
        slide_suffix_modifiers += branch.group(4).lower()
    return NoteEvent(
        note_type="slide",
        branches=branches,
        **{
            **common,
            "is_break": common["is_break"] or "b" in slide_suffix_modifiers,
            "is_ex": common["is_ex"] or "x" in slide_suffix_modifiers,
        },
    )


def _strip_terminator(chart: str) -> str:
    return re.split(r"(?m)^E\s*$", chart, maxsplit=1)[0].strip().rstrip(",")


def parse_chart(fields: dict[str, str], difficulty_index: int, *, ignore_touch: bool = False) -> ParsedChart:
    key = f"inote_{difficulty_index}"
    if key not in fields or not fields[key].strip():
        raise SimaiParseError(f"Missing chart field: {key}")

    first_ms = float(fields.get("first", "0") or 0) * 1000.0
    default_bpm = float(fields.get("wholebpm", "120") or 120)
    bpm = default_bpm
    division = 4
    tick = 0
    time_ms = first_ms
    events: list[NoteEvent] = []
    tick_times_ms = [first_ms]
    bpm_changes: list[dict[str, float | int]] = []
    division_changes: list[dict[str, int]] = []

    chart = _strip_terminator(fields[key])
    tokens = chart.replace("\r", "").replace("\n", "").split(",") if chart else []

    for slot, token in enumerate(tokens):
        for control in CONTROL_RE.finditer(token):
            if control.group(1) is not None:
                bpm = float(control.group(1))
                bpm_changes.append({"tick": tick, "time_ms": round(time_ms, 6), "bpm": bpm})
            else:
                division = int(control.group(2))
                if TICKS_PER_MEASURE % division:
                    raise SimaiParseError(
                        f"Division {division} cannot be represented with {TICKS_PER_MEASURE} ticks"
                    )
                division_changes.append({"tick": tick, "division": division})

        cleaned = CONTROL_RE.sub("", token).strip()
        if cleaned:
            for component in cleaned.split("/"):
                component = component.strip()
                if component:
                    if ignore_touch and TOUCH_RE.match(component):
                        continue
                    events.append(
                        parse_component(
                            component,
                            tick=tick,
                            time_ms=time_ms,
                            slot=slot,
                            bpm=bpm,
                        )
                    )

        step_ticks = TICKS_PER_MEASURE // division
        step_ms = 240000.0 / (bpm * division)
        for sub_tick in range(1, step_ticks + 1):
            tick_times_ms.append(time_ms + step_ms * sub_tick / step_ticks)
        tick += step_ticks
        time_ms += step_ms

    return ParsedChart(
        difficulty_index=difficulty_index,
        level_text=fields.get(f"lv_{difficulty_index}", ""),
        designer=fields.get(f"des_{difficulty_index}", ""),
        first_ms=first_ms,
        default_bpm=default_bpm,
        events=events,
        tick_times_ms=[round(value, 6) for value in tick_times_ms],
        bpm_changes=bpm_changes,
        division_changes=division_changes,
    )


def relative_lane(start_lane: int, lane: int) -> int:
    return (lane - start_lane) % 8


def canonical_slide_key(event: NoteEvent) -> str:
    if event.note_type != "slide":
        raise ValueError("Only slide events have canonical slide keys")
    branch_keys = []
    for branch in event.branches:
        relative_path = ".".join(str(relative_lane(event.lane, lane)) for lane in branch.path_lanes)
        branch_keys.append(f"{branch.operator}:{relative_path}:{branch.duration.raw}")
    branch_keys.sort()
    return "*".join(branch_keys)

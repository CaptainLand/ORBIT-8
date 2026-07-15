from __future__ import annotations

import copy
from collections import defaultdict


MIRROR_MODES = ("normal", "horizontal", "vertical", "half_turn")
PATTERN_NAMES = ("none", "interaction", "sweep", "jack")
PATTERN_NONE = 0
PATTERN_INTERACTION = 1
PATTERN_SWEEP = 2
PATTERN_JACK = 3
PATTERN_TRAINING_WEIGHTS = (0.1, 2.0, 1.5, 0.2)
MAX_JACK_PATTERN_SHARE = 0.02

LANE_MAPS = {
    "normal": {lane: lane for lane in range(1, 9)},
    "horizontal": {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1},
    "vertical": {1: 4, 2: 3, 3: 2, 4: 1, 5: 8, 6: 7, 7: 6, 8: 5},
    "half_turn": {1: 5, 2: 6, 3: 7, 4: 8, 5: 1, 6: 2, 7: 3, 8: 4},
}

REFLECTED_OPERATORS = {
    "<": ">",
    ">": "<",
    "p": "q",
    "q": "p",
    "pp": "qq",
    "qq": "pp",
    "s": "z",
    "z": "s",
}


def mirror_lane(lane: int, mode: str) -> int:
    return LANE_MAPS[mode][int(lane)]


def mirror_operator(operator: str, mode: str) -> str:
    if mode in {"horizontal", "vertical"}:
        return REFLECTED_OPERATORS.get(operator, operator)
    return operator


def mirror_event(event: dict, mode: str) -> dict:
    if mode == "normal":
        return copy.deepcopy(event)
    result = copy.deepcopy(event)
    result["lane"] = mirror_lane(result["lane"], mode)
    for branch in result.get("branches") or []:
        branch["operator"] = mirror_operator(branch["operator"], mode)
        branch["path_lanes"] = [mirror_lane(lane, mode) for lane in branch["path_lanes"]]
        branch["end_lane"] = mirror_lane(branch["end_lane"], mode)
    return result


def _ring_distance(first: int, second: int) -> int:
    raw = abs(first - second)
    return min(raw, 8 - raw)


def _matches(kind: int, lanes: list[int], gap: int) -> bool:
    if kind == PATTERN_INTERACTION:
        return (
            gap == 12
            and
            len(lanes) >= 4
            and lanes[-1] == lanes[-3]
            and lanes[-2] == lanes[-4]
            and lanes[-1] != lanes[-2]
            and _ring_distance(lanes[-1], lanes[-2]) <= 2
        )
    if kind == PATTERN_SWEEP:
        if gap not in {6, 8, 12} or len(lanes) < 4:
            return False
        deltas = [((right - left) % 8) for left, right in zip(lanes[-4:], lanes[-3:])]
        return len(set(deltas)) == 1 and deltas[0] in {1, 7}
    if kind == PATTERN_JACK:
        return gap == 12 and len(lanes) >= 3 and len(set(lanes[-3:])) == 1
    return False


def detect_pattern_labels(events: list[dict]) -> list[int]:
    labels = [PATTERN_NONE] * len(events)
    by_tick: dict[int, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        by_tick[int(event["tick"])].append(index)

    groups = sorted(by_tick.items())
    eligible = []
    for order, (tick, indices) in enumerate(groups):
        if len(indices) == 1 and events[indices[0]]["note_type"] == "tap":
            eligible.append((order, tick, indices[0], int(events[indices[0]]["lane"])))
        else:
            eligible.append((order, tick, -1, -1))

    candidates: list[tuple[int, int, int, list[int]]] = []
    for start in range(len(eligible)):
        order, tick, event_index, lane = eligible[start]
        if event_index < 0 or start + 2 >= len(eligible):
            continue
        for kind, minimum in (
            (PATTERN_INTERACTION, 4),
            (PATTERN_SWEEP, 4),
            (PATTERN_JACK, 3),
        ):
            run = [eligible[start]]
            gap = None
            cursor = start + 1
            while cursor < len(eligible):
                item = eligible[cursor]
                if item[2] < 0 or item[0] != run[-1][0] + 1:
                    break
                current_gap = item[1] - run[-1][1]
                if current_gap <= 0 or current_gap > 48:
                    break
                if gap is None:
                    gap = current_gap
                if current_gap != gap:
                    break
                trial_lanes = [entry[3] for entry in run] + [item[3]]
                if len(trial_lanes) >= minimum and not _matches(kind, trial_lanes, gap):
                    break
                run.append(item)
                cursor += 1
            if len(run) >= minimum and _matches(kind, [entry[3] for entry in run], int(gap or 0)):
                candidates.append((len(run), -start, kind, [entry[2] for entry in run]))

    for _, _, kind, indices in sorted(candidates, reverse=True):
        if any(labels[index] != PATTERN_NONE for index in indices):
            continue
        for index in indices:
            labels[index] = kind
    return labels

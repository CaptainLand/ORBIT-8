from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from itertools import permutations, product
from typing import Iterable

from generate_maimai import (
    GeneratedEvent,
    _lane_point,
    event_hand_cost,
    slide_corridor_lanes,
    slide_motion_interval,
    slide_tail_lanes,
)
from maimai_ai.patterns import PATTERN_SWEEP


LEFT_SHOULDER = (-0.42, -1.28)
RIGHT_SHOULDER = (0.42, -1.28)
REPAIR_PENALTY = 2.75
BEAM_WIDTH = 256
MAX_HAND_SPEED = 3.0
MAX_FUTURE_REACH_SPEED = 3.25
MIN_POSTURE_CHANGE_TICKS = 24
MAX_HAND_FLOW_DROPS = 32


@dataclass(frozen=True)
class Reservation:
    start: int
    end: int
    start_lane: int
    end_lane: int
    corridor: frozenset[int]
    direction: int
    kind: str


@dataclass(frozen=True)
class HandState:
    lane: int
    last_tick: int
    reservations: tuple[Reservation, ...] = ()


@dataclass(frozen=True)
class FlowState:
    left: HandState
    right: HandState
    cost: float = 0.0
    crossed: bool = False
    last_posture_change: int = -10_000
    crossings: int = 0
    rapid_posture_changes: int = 0
    backhand_actions: int = 0
    max_speed: float = 0.0
    lane_changes: tuple[tuple[int, int], ...] = ()
    assignments: tuple[tuple[int, str], ...] = ()
    sweep_side: str | None = None
    last_sweep_tick: int = -10_000


def circular_distance(first: int, second: int) -> int:
    raw = abs(first - second)
    return min(raw, 8 - raw)


def _normalize_hand(hand: HandState, tick: int) -> HandState:
    lane = hand.lane
    last_tick = hand.last_tick
    remaining = []
    for reservation in hand.reservations:
        if reservation.end <= tick:
            lane = reservation.end_lane
            last_tick = max(last_tick, reservation.end)
        else:
            remaining.append(reservation)
    return HandState(lane, last_tick, tuple(remaining))


def _active_reservation(hand: HandState, tick: int) -> Reservation | None:
    return next((item for item in hand.reservations if item.start <= tick < item.end), None)


def _position_for_hand(hand: HandState, tick: int) -> tuple[float, float]:
    active = _active_reservation(hand, tick)
    if active is None:
        return _lane_point(hand.lane)
    if active.kind == "hold" or active.end <= active.start:
        return _lane_point(active.start_lane)
    progress = min(1.0, max(0.0, (tick - active.start) / (active.end - active.start)))
    start = _lane_point(active.start_lane)
    start_angle = math.atan2(start[1], start[0])
    if active.direction:
        delta = active.direction * circular_distance(active.start_lane, active.end_lane) * math.pi / 4
    else:
        raw = (active.end_lane - active.start_lane) % 8
        signed = raw if raw <= 4 else raw - 8
        delta = -signed * math.pi / 4
    angle = start_angle + delta * progress
    return math.cos(angle), math.sin(angle)


def _orientation(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross(a, b, c, d) -> bool:
    return (
        _orientation(a, b, c) * _orientation(a, b, d) < -1e-6
        and _orientation(c, d, a) * _orientation(c, d, b) < -1e-6
    )


def _posture_crossed(left: HandState, right: HandState, tick: int) -> bool:
    left_point = _position_for_hand(left, tick)
    right_point = _position_for_hand(right, tick)
    return (
        _segments_cross(LEFT_SHOULDER, left_point, RIGHT_SHOULDER, right_point)
        or left_point[0] > right_point[0] + 0.18
    )


def _slide_direction(event: GeneratedEvent) -> int:
    if not event.slide_template:
        return 0
    operator = event.slide_template["operators"][0]
    if operator in {"q", "qq", ">"}:
        return 1
    if operator in {"p", "pp", "<"}:
        return -1
    return 0


def _reservations_for_event(event: GeneratedEvent) -> list[Reservation]:
    if event.kind == "hold":
        return [Reservation(event.tick, event.tick + event.duration + 12, event.lane, event.lane,
                            frozenset({event.lane}), 0, "hold")]
    if event.kind != "slide" or not event.slide_template:
        return []
    interval = slide_motion_interval(event)
    if interval is None:
        return []
    corridor, endpoint = slide_corridor_lanes(event)
    tails = sorted(slide_tail_lanes(event))
    endpoints = tails[:2] if event_hand_cost(event) == 2 and len(tails) >= 2 else [endpoint]
    return [
        Reservation(interval[0], interval[1] + 12, event.lane, tail, frozenset(corridor),
                    _slide_direction(event), "slide")
        for tail in endpoints
    ]


def _reservation_overlap(hand: HandState, reservation: Reservation) -> bool:
    return any(max(item.start, reservation.start) < min(item.end, reservation.end)
               for item in hand.reservations)


def _future_reach_speed(hand: HandState, lane: int, tick: int) -> float:
    future = next((item for item in hand.reservations if item.start > tick), None)
    if future is None:
        return 0.0
    return circular_distance(lane, future.start_lane) * 12.0 / max(1, future.start - tick)


def _move_hand(hand: HandState, lane: int, tick: int, side: str,
               reservation: Reservation | None) -> tuple[HandState, float, float, int] | None:
    if _active_reservation(hand, tick) is not None:
        return None
    distance = circular_distance(hand.lane, lane)
    speed = distance * 12.0 / max(6, tick - hand.last_tick)
    future_speed = _future_reach_speed(hand, lane, tick)
    if speed > MAX_HAND_SPEED or future_speed > MAX_FUTURE_REACH_SPEED:
        return None
    point = _lane_point(lane)
    backhand = int((side == "L" and point[0] > 0.35) or (side == "R" and point[0] < -0.35))
    natural = max(0.0, point[0]) if side == "L" else max(0.0, -point[0])
    future_penalty = 1.4 * max(0.0, future_speed - 3.0) ** 2
    move_cost = 0.32 * speed * speed + 0.22 * distance + 0.55 * natural + future_penalty
    reservations = hand.reservations
    if reservation is not None:
        if _reservation_overlap(hand, reservation):
            return None
        reservations = tuple(sorted((*reservations, reservation), key=lambda item: item.start))
    return HandState(lane, tick, reservations), move_cost, speed, backhand


def _blocked_by_other_hand(other: HandState, lane: int, tick: int) -> bool:
    active = _active_reservation(other, tick)
    return active is not None and lane in active.corridor


def _candidate_lanes(event: GeneratedEvent, allow_repair: bool) -> tuple[int, ...]:
    if not allow_repair or event.kind not in {"tap", "hold", "slide"} or event.pattern_type != 0:
        return (event.lane,)
    alternatives = sorted(
        (lane for lane in range(1, 9) if lane != event.lane),
        key=lambda lane: (circular_distance(event.lane, lane), lane),
    )
    return (event.lane, *alternatives)


def _state_key(state: FlowState, tick: int) -> tuple:
    def hand_key(hand: HandState):
        active = _active_reservation(hand, tick)
        future = tuple(
            (item.start, item.end, item.start_lane, item.end_lane, item.direction, item.kind)
            for item in hand.reservations
        )
        return hand.lane, hand.last_tick, active.end if active else -1, future
    return (
        hand_key(state.left), hand_key(state.right), state.crossed,
        state.sweep_side, state.last_sweep_tick,
    )


def _expand_group(state: FlowState, indexed_events: list[tuple[int, GeneratedEvent]],
                  allow_repair: bool) -> Iterable[FlowState]:
    tick = indexed_events[0][1].tick
    state = replace(state, left=_normalize_hand(state.left, tick), right=_normalize_hand(state.right, tick))
    if len(indexed_events) > 2:
        return
    for lanes in product(*[_candidate_lanes(event, allow_repair) for _, event in indexed_events]):
        repair_factor = {"tap": 1.0, "hold": 1.8, "slide": 3.0}
        repair_cost = sum(REPAIR_PENALTY * repair_factor[event.kind]
                           * circular_distance(event.lane, lane)
                           for (_, event), lane in zip(indexed_events, lanes))
        required_hands = sum(event_hand_cost(event) for _, event in indexed_events)
        single_sweep = (
            len(indexed_events) == 1
            and indexed_events[0][1].kind == "tap"
            and indexed_events[0][1].pattern_type == PATTERN_SWEEP
        )
        if required_hands > 2:
            continue
        if required_hands == 2 and len(indexed_events) == 1:
            hand_orders = [("L", "R")]
        elif (
            single_sweep
            and state.sweep_side is not None
            and tick - state.last_sweep_tick in {6, 8, 12}
        ):
            hand_orders = [(state.sweep_side,)]
        elif len(indexed_events) == 2:
            hand_orders = list(permutations(("L", "R"), 2))
        else:
            hand_orders = [("L",), ("R",)]
        for hand_order in hand_orders:
            left, right = state.left, state.right
            cost, max_speed, backhands = state.cost + repair_cost, state.max_speed, state.backhand_actions
            assignments, changes = list(state.assignments), list(state.lane_changes)
            valid = True
            for position, ((event_index, event), lane) in enumerate(zip(indexed_events, lanes)):
                moved_event = copy.copy(event)
                moved_event.lane = lane
                reservations = _reservations_for_event(moved_event)
                if event_hand_cost(event) == 2:
                    if len(reservations) < 2:
                        reservations = (reservations * 2)[:2]
                    if len(reservations) < 2:
                        valid = False
                        break
                    for side, reservation in zip(("L", "R"), reservations):
                        other = right if side == "L" else left
                        if _blocked_by_other_hand(other, lane, tick):
                            valid = False
                            break
                        moved = _move_hand(left if side == "L" else right, lane, tick, side, reservation)
                        if moved is None:
                            valid = False
                            break
                        hand, move_cost, speed, backhand = moved
                        if side == "L": left = hand
                        else: right = hand
                        cost += move_cost; max_speed = max(max_speed, speed); backhands += backhand
                    assignments.append((event_index, "LR"))
                else:
                    side = hand_order[position]
                    other = right if side == "L" else left
                    if _blocked_by_other_hand(other, lane, tick):
                        valid = False
                        break
                    moved = _move_hand(left if side == "L" else right, lane, tick, side,
                                       reservations[0] if reservations else None)
                    if moved is None:
                        valid = False
                        break
                    hand, move_cost, speed, backhand = moved
                    if side == "L": left = hand
                    else: right = hand
                    cost += move_cost; max_speed = max(max_speed, speed); backhands += backhand
                    assignments.append((event_index, side))
                if lane != event.lane:
                    changes.append((event_index, lane))
            if not valid:
                continue
            crossed = _posture_crossed(left, right, tick)
            crossings, rapid, last_change = state.crossings, state.rapid_posture_changes, state.last_posture_change
            if crossed != state.crossed:
                gap = tick - state.last_posture_change
                if gap < MIN_POSTURE_CHANGE_TICKS:
                    continue
                if _active_reservation(left, tick) or _active_reservation(right, tick):
                    continue
                cost += 1.4 if crossed else 0.55
                if gap < 48:
                    cost += 2.5 * (48 - gap) / 48
                    rapid += 1
                crossings += int(crossed)
                last_change = tick
            elif crossed:
                cost += 0.18
            if crossed and (_active_reservation(left, tick) or _active_reservation(right, tick)):
                cost += 1.15
            sweep_side = hand_order[0] if single_sweep else None
            last_sweep_tick = tick if single_sweep else -10_000
            yield FlowState(left, right, cost, crossed, last_change, crossings, rapid, backhands,
                            max_speed, tuple(changes), tuple(assignments), sweep_side, last_sweep_tick)


def search_handflow(events: list[GeneratedEvent], *, allow_repair: bool,
                    beam_width: int = BEAM_WIDTH) -> tuple[FlowState | None, int | None]:
    indexed = sorted(enumerate(events), key=lambda item: (item[1].tick, item[1].lane))
    groups: list[list[tuple[int, GeneratedEvent]]] = []
    for item in indexed:
        if not groups or groups[-1][0][1].tick != item[1].tick:
            groups.append([item])
        else:
            groups[-1].append(item)
    beam = [FlowState(HandState(7, -96), HandState(2, -96))]
    for group in groups:
        candidates = [candidate for state in beam for candidate in _expand_group(state, group, allow_repair)]
        if not candidates:
            return None, group[0][1].tick
        best_by_key: dict[tuple, FlowState] = {}
        tick = group[0][1].tick
        for candidate in candidates:
            key = _state_key(candidate, tick)
            if key not in best_by_key or candidate.cost < best_by_key[key].cost:
                best_by_key[key] = candidate
        beam = sorted(best_by_key.values(), key=lambda item: item.cost)[:beam_width]
    return min(beam, key=lambda item: item.cost), None


def _metrics(state: FlowState | None, failure_tick: int | None) -> dict:
    return {
        "feasible": state is not None, "failure_tick": failure_tick,
        "cost": round(state.cost, 3) if state else None,
        "crossings": state.crossings if state else None,
        "rapid_posture_changes": state.rapid_posture_changes if state else None,
        "backhand_actions": state.backhand_actions if state else None,
        "max_normalized_hand_speed": round(state.max_speed, 3) if state else None,
        "lane_changes": len(state.lane_changes) if state else None,
    }


def _failure_events(events: list[GeneratedEvent], tick: int | None) -> list[dict]:
    if tick is None:
        return []
    return [
        {
            "lane": event.lane,
            "kind": event.kind,
            "duration": event.duration,
            "pattern_type": event.pattern_type,
            "hand_cost": event_hand_cost(event),
            "operators": event.slide_template["operators"] if event.slide_template else [],
        }
        for event in events if event.tick == tick
    ]


def optimize_handflow(events: list[GeneratedEvent]) -> tuple[list[GeneratedEvent], dict]:
    baseline, baseline_failure = search_handflow(events, allow_repair=False)
    working = copy.deepcopy(events)
    dropped = []
    optimized = None
    optimized_failure = None
    for _ in range(MAX_HAND_FLOW_DROPS + 1):
        optimized, optimized_failure = search_handflow(working, allow_repair=True)
        if optimized is not None:
            break
        candidates = [
            (index, event) for index, event in enumerate(working)
            if event.tick == optimized_failure and event.kind == "tap" and event.pattern_type == 0
        ]
        if not candidates or len(dropped) >= MAX_HAND_FLOW_DROPS:
            break
        remove_index, event = min(candidates, key=lambda item: item[1].score)
        dropped.append({
            "tick": event.tick,
            "lane": event.lane,
            "score": round(event.score, 6),
            "reason": "strict_handflow_failure",
        })
        working.pop(remove_index)
    if optimized is None:
        return events, {
            "baseline": _metrics(baseline, baseline_failure),
            "baseline_failure_events": _failure_events(events, baseline_failure),
            "optimized": _metrics(optimized, optimized_failure),
            "optimized_failure_events": _failure_events(working, optimized_failure),
            "dropped": dropped,
            "applied": False,
        }
    result = copy.deepcopy(working)
    changes = []
    for event_index, lane in optimized.lane_changes:
        changes.append({
            "tick": working[event_index].tick,
            "kind": working[event_index].kind,
            "from_lane": working[event_index].lane,
            "to_lane": lane,
            "distance": circular_distance(working[event_index].lane, lane),
        })
        result[event_index].lane = lane
    final_state, final_failure = search_handflow(result, allow_repair=False)
    return result, {
        "baseline": _metrics(baseline, baseline_failure),
        "baseline_failure_events": _failure_events(events, baseline_failure),
        "optimized": _metrics(optimized, optimized_failure),
        "optimized_failure_events": _failure_events(events, optimized_failure),
        "final_assignment": _metrics(final_state, final_failure),
        "changes": changes,
        "dropped": dropped,
        "applied": bool(optimized.lane_changes or dropped),
    }

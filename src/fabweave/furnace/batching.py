from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from fabweave.core.ids import deterministic_id
from fabweave.core.timebox import TimeBox

FAR_FUTURE = 2**31 - 1

# Rejection reasons are deliberately specific so every unassigned lot has a
# traceable destination rather than a generic failure.
RECIPE_UNSUPPORTED = "recipe_family_unsupported"
CONTAMINATION_NOT_SUPPORTED = "contamination_class_not_supported"
NO_AVAILABLE_WINDOW = "no_available_equipment_window"
NO_QUALIFIED_EQUIPMENT = "no_qualified_equipment"
WINDOW_CLOSED = "wip_due_window_closed"
INSUFFICIENT_COMPATIBLE_LOAD = "insufficient_compatible_load_for_minimum_batch"
DEFERRED_CAPACITY_CONTENTION = "deferred_due_to_capacity_or_material_contention"

# Exhaustive subset search is bounded; larger groups use the deterministic
# order-fill path instead of an exponential global search.
_ENUMERATION_LIMIT = 18
_NODE_BUDGET = 200_000


@dataclass(frozen=True, slots=True)
class FurnaceCandidate:
    """A candidate WIP lot.  Lots are indivisible members of a furnace batch."""

    lot_id: str
    recipe_family: str
    quantity: int
    contamination_class: str = "GENERAL"
    release_at: int = 0
    due_at: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.lot_id, str) or not self.lot_id:
            raise ValueError("candidate lot id is required")
        if not isinstance(self.recipe_family, str) or not self.recipe_family:
            raise ValueError("recipe family is required")
        if not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("candidate quantity must be a positive integer")
        if not isinstance(self.release_at, int):
            raise ValueError("release_at must be an integer")
        if self.due_at is not None and (
            not isinstance(self.due_at, int) or self.due_at <= self.release_at
        ):
            raise ValueError("due_at must be an integer later than release_at")


@dataclass(frozen=True, slots=True)
class FurnaceTube:
    """Load limits and process/equipment constraints for one furnace tube."""

    tube_id: str
    min_load: int
    max_load: int
    recipe_families: frozenset[str]
    allowed_contamination: frozenset[str] | None = None
    available_start: int = 0
    available_end: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tube_id, str) or not self.tube_id:
            raise ValueError("tube id is required")
        if not isinstance(self.min_load, int) or not isinstance(self.max_load, int):
            raise ValueError("load limits must be integers")
        if self.min_load <= 0 or self.max_load < self.min_load:
            raise ValueError("load limits must satisfy 0 < min_load <= max_load")
        if not self.recipe_families:
            raise ValueError("at least one recipe family is required")
        if not isinstance(self.available_start, int):
            raise ValueError("available_start must be an integer")
        if self.available_end is not None and (
            not isinstance(self.available_end, int)
            or self.available_end <= self.available_start
        ):
            raise ValueError("tube availability window is invalid")

    @property
    def window(self) -> TimeBox:
        return TimeBox(
            self.available_start,
            self.available_end if self.available_end is not None else FAR_FUTURE,
        )


@dataclass(frozen=True, slots=True)
class FurnaceBatch:
    batch_id: str
    tube_id: str
    recipe_family: str
    lot_ids: tuple[str, ...]
    total_quantity: int
    window: TimeBox
    contamination_classes: frozenset[str]


@dataclass(frozen=True, slots=True)
class UnassignedWip:
    lot_id: str
    reasons: tuple[str, ...]
    disposition: str = "UNASSIGNED_WIP"


@dataclass(frozen=True, slots=True)
class FurnacePlan:
    request_id: str
    batches: tuple[FurnaceBatch, ...]
    unassigned: tuple[UnassignedWip, ...]
    at: int

    @property
    def batch_ids(self) -> tuple[str, ...]:
        return tuple(item.batch_id for item in self.batches)

    @property
    def assigned_lot_ids(self) -> tuple[str, ...]:
        return tuple(lot_id for batch in self.batches for lot_id in batch.lot_ids)

    @property
    def unassigned_lot_ids(self) -> tuple[str, ...]:
        return tuple(item.lot_id for item in self.unassigned)

    def destination(self, lot_id: str) -> str:
        """Return the batch id, or the explicit disposition for rejected WIP."""

        for batch in self.batches:
            if lot_id in batch.lot_ids:
                return batch.batch_id
        for item in self.unassigned:
            if item.lot_id == lot_id:
                return item.disposition
        raise KeyError(lot_id)


@dataclass(frozen=True, slots=True)
class _ScreenedCandidate:
    candidate: FurnaceCandidate
    start: int
    end: int
    eligible_tubes: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Pack:
    lot_ids: tuple[str, ...]
    load: int
    start: int
    end: int


class _NodeBudgetExceeded(Exception):
    pass


def _normalise_matrix(matrix: Mapping[str, set[str]]) -> dict[str, frozenset[str]]:
    normalized: dict[str, set[str]] = {}
    for left, others in matrix.items():
        normalized.setdefault(left, set()).update(others)
        for right in others:
            normalized.setdefault(right, set()).add(left)
    return {key: frozenset(values) for key, values in normalized.items()}


def _compatible(left: str, right: str, matrix: Mapping[str, frozenset[str]]) -> bool:
    return right not in matrix.get(left, frozenset())


def _due(member: _ScreenedCandidate) -> int:
    due = member.candidate.due_at
    return due if due is not None else FAR_FUTURE


def _screen_candidate(
    candidate: FurnaceCandidate,
    tubes: Sequence[FurnaceTube],
    now: int,
) -> tuple[_ScreenedCandidate | None, tuple[str, ...]]:
    if candidate.due_at is not None and candidate.due_at <= now:
        return None, (WINDOW_CLOSED,)

    start = max(candidate.release_at, now)
    end = candidate.due_at if candidate.due_at is not None else FAR_FUTURE

    eligible: list[str] = []
    failures: set[str] = set()
    for tube in sorted(tubes, key=lambda item: item.tube_id):
        tube_failures: list[str] = []
        if candidate.recipe_family not in tube.recipe_families:
            tube_failures.append(RECIPE_UNSUPPORTED)
        if (
            tube.allowed_contamination is not None
            and candidate.contamination_class not in tube.allowed_contamination
        ):
            tube_failures.append(CONTAMINATION_NOT_SUPPORTED)
        if not (start < end and tube.window.intersection(TimeBox(start, end))):
            tube_failures.append(NO_AVAILABLE_WINDOW)
        if not tube_failures:
            eligible.append(tube.tube_id)
        failures.update(tube_failures)

    if eligible:
        return (
            _ScreenedCandidate(candidate, start, end, frozenset(eligible)),
            (),
        )

    if failures <= {RECIPE_UNSUPPORTED}:
        return None, (RECIPE_UNSUPPORTED,)
    if failures <= {CONTAMINATION_NOT_SUPPORTED}:
        return None, (CONTAMINATION_NOT_SUPPORTED,)
    if failures <= {NO_AVAILABLE_WINDOW}:
        return None, (NO_AVAILABLE_WINDOW,)
    return None, (NO_QUALIFIED_EQUIPMENT, *sorted(failures))


def _order_fill(
    tube: FurnaceTube,
    members: Sequence[_ScreenedCandidate],
    matrix: Mapping[str, frozenset[str]],
) -> _Pack | None:
    """Deterministic, polynomial fallback: fill in business-rank order."""

    chosen: list[_ScreenedCandidate] = []
    load = 0
    start = tube.available_start
    end = tube.available_end if tube.available_end is not None else FAR_FUTURE
    contamination: set[str] = set()

    for member in members:
        klass = member.candidate.contamination_class
        if any(not _compatible(klass, prior, matrix) for prior in contamination):
            continue
        next_start = max(start, member.start)
        next_end = min(end, member.end)
        if next_start >= next_end or load + member.candidate.quantity > tube.max_load:
            continue
        chosen.append(member)
        load += member.candidate.quantity
        start, end, contamination = next_start, next_end, contamination | {klass}

    if load < tube.min_load:
        return None
    return _Pack(tuple(item.candidate.lot_id for item in chosen), load, start, end)


def _best_pack(
    tube: FurnaceTube,
    members: Sequence[_ScreenedCandidate],
    matrix: Mapping[str, frozenset[str]],
) -> _Pack | None:
    """
    Best feasible single-tube pack.  Small groups are solved exactly with a
    bounded branch-and-bound subset search; large groups fall back to the
    deterministic order-fill heuristic.
    """

    if not members or sum(item.candidate.quantity for item in members) < tube.min_load:
        return None
    if len(members) > _ENUMERATION_LIMIT:
        return _order_fill(tube, members, matrix)

    suffix = [0] * (len(members) + 1)
    for index in range(len(members) - 1, -1, -1):
        suffix[index] = suffix[index + 1] + members[index].candidate.quantity

    best: dict[str, object] = {}
    nodes = 0

    def consider(
        chosen: list[int], load: int, start: int, end: int
    ) -> None:
        lot_ids = tuple(members[pos].candidate.lot_id for pos in chosen)
        dues = tuple(sorted(_due(members[pos]) for pos in chosen))
        key = (-load, dues, -len(chosen), lot_ids)
        if not best or key < best["key"]:
            best.update(key=key, pack=_Pack(lot_ids, load, start, end))

    def extend(
        index: int,
        chosen: list[int],
        load: int,
        start: int,
        end: int,
        contamination: frozenset[str],
    ) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _NODE_BUDGET:
            raise _NodeBudgetExceeded

        if load >= tube.min_load:
            consider(chosen, load, start, end)
        if index == len(members) or load == tube.max_load:
            return
        if load + suffix[index] < tube.min_load:
            return
        if best and min(tube.max_load, load + suffix[index]) < -int(best["key"][0]):
            return

        for next_index in range(index, len(members)):
            member = members[next_index]
            klass = member.candidate.contamination_class
            if any(not _compatible(klass, prior, matrix) for prior in contamination):
                continue
            next_start = max(start, member.start)
            next_end = min(end, member.end)
            next_load = load + member.candidate.quantity
            if next_start >= next_end or next_load > tube.max_load:
                continue
            chosen.append(next_index)
            extend(
                next_index + 1,
                chosen,
                next_load,
                next_start,
                next_end,
                contamination | {klass},
            )
            chosen.pop()

    try:
        extend(
            0,
            [],
            0,
            tube.available_start,
            tube.available_end if tube.available_end is not None else FAR_FUTURE,
            frozenset(),
        )
    except _NodeBudgetExceeded:
        return _order_fill(tube, members, matrix)

    return best.get("pack") if best else None


def plan_furnace_batches(
    request_id: str,
    candidates: Sequence[FurnaceCandidate],
    tubes: Sequence[FurnaceTube],
    *,
    at: int = 0,
    contamination_matrix: Mapping[str, set[str]] | None = None,
) -> FurnacePlan:
    """
    Deterministically form load-feasible furnace batches.

    The planner never consumes arrival/input order.  Lots are screened against
    recipe family, contamination class/equipment restrictions and overlapping
    due windows; survivors are ranked by a stable business key (due date,
    release, quantity, id).  Each round every idle tube proposes its best
    feasible batch for one recipe family (exact bounded search, deterministic
    fill fallback), and the globally tightest proposal is committed.  Tubes
    host at most one batch and lots cannot repeat.  Every candidate ends in
    exactly one batch or in the rejected list with a specific reason; hard
    rejects are kept distinct from capacity/contention leftovers.
    """

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("request id is required")
    if not isinstance(at, int):
        raise ValueError("planning time must be an integer")
    if not tubes:
        raise ValueError("at least one furnace tube is required")
    if len({tube.tube_id for tube in tubes}) != len(tubes):
        raise ValueError("tube ids must be unique")

    candidate_ids = [candidate.lot_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate lot ids must be unique")

    matrix = _normalise_matrix(contamination_matrix or {})
    screened: dict[str, _ScreenedCandidate] = {}
    unassigned: list[UnassignedWip] = []

    for candidate in sorted(candidates, key=lambda item: item.lot_id):
        screened_candidate, reasons = _screen_candidate(candidate, tubes, at)
        if screened_candidate is None:
            unassigned.append(UnassignedWip(candidate.lot_id, reasons))
        else:
            screened[candidate.lot_id] = screened_candidate

    tube_order = tuple(sorted(tubes, key=lambda item: item.tube_id))
    recipes = frozenset(item.candidate.recipe_family for item in screened.values())

    def group_members(
        tube: FurnaceTube, recipe: str, available: frozenset[str]
    ) -> list[_ScreenedCandidate]:
        return [
            screened[lot_id]
            for lot_id in sorted(
                available,
                key=lambda ident: (
                    _due(screened[ident]),
                    screened[ident].start,
                    -screened[ident].candidate.quantity,
                    ident,
                ),
            )
            if tube.tube_id in screened[lot_id].eligible_tubes
            and screened[lot_id].candidate.recipe_family == recipe
        ]

    # A lot that can appear in at least one feasible pack from the initial
    # pool is a capacity/contention leftover; anything else is structurally
    # short of compatible load.
    ever_packable: set[str] = set()
    for tube in tube_order:
        for recipe in sorted(recipes):
            members = group_members(tube, recipe, frozenset(screened))
            pack = _best_pack(tube, members, matrix)
            if pack is not None:
                # The best pack alone is not enough; collect all packable lots
                # with one exact pass over this group.
                ever_packable.update(_packable_lot_ids(tube, members, matrix))

    available = frozenset(screened)
    used_tubes: set[str] = set()
    batches: list[FurnaceBatch] = []

    while True:
        proposals: list[tuple[tuple, FurnaceTube, str, _Pack]] = []
        for tube in tube_order:
            if tube.tube_id in used_tubes:
                continue
            for recipe in sorted(recipes):
                members = group_members(tube, recipe, available)
                pack = _best_pack(tube, members, matrix)
                if pack is None:
                    continue
                proposal_key = (
                    pack.end,
                    -pack.load,
                    tuple(sorted(_due(screened[lot_id]) for lot_id in pack.lot_ids)),
                    recipe,
                    tube.tube_id,
                    pack.lot_ids,
                )
                proposals.append((proposal_key, tube, recipe, pack))

        if not proposals:
            break

        _, tube, recipe, pack = min(proposals, key=lambda item: item[0])
        chosen = [screened[lot_id] for lot_id in pack.lot_ids]
        start = max([tube.available_start, *(item.start for item in chosen)])
        end = min(
            [tube.available_end if tube.available_end is not None else FAR_FUTURE,
             *(item.end for item in chosen)]
        )
        batches.append(
            FurnaceBatch(
                batch_id=deterministic_id(
                    "BATCH", request_id, tube.tube_id, recipe, *pack.lot_ids
                ),
                tube_id=tube.tube_id,
                recipe_family=recipe,
                lot_ids=pack.lot_ids,
                total_quantity=pack.load,
                window=TimeBox(start, end),
                contamination_classes=frozenset(
                    item.candidate.contamination_class for item in chosen
                ),
            )
        )
        used_tubes.add(tube.tube_id)
        available -= frozenset(pack.lot_ids)

    batches = tuple(
        sorted(
            batches,
            key=lambda item: (
                item.window.end,
                item.recipe_family,
                item.tube_id,
                item.batch_id,
            ),
        )
    )

    for lot_id in sorted(available):
        reason = (
            DEFERRED_CAPACITY_CONTENTION
            if lot_id in ever_packable
            else INSUFFICIENT_COMPATIBLE_LOAD
        )
        unassigned.append(UnassignedWip(lot_id, (reason,)))

    # ---- fail-closed conservation checks --------------------------------
    all_assigned = [lot_id for batch in batches for lot_id in batch.lot_ids]
    if len(all_assigned) != len(set(all_assigned)):
        raise RuntimeError("furnace planner reused a lot")
    rejected_ids = {item.lot_id for item in unassigned}
    if set(all_assigned) & rejected_ids:
        raise RuntimeError("furnace planner assigned and rejected the same lot")
    if set(all_assigned) | rejected_ids != {candidate.lot_id for candidate in candidates}:
        raise RuntimeError("furnace planner lost a lot")
    input_quantity = sum(candidate.quantity for candidate in candidates)
    unassigned_quantity = sum(
        candidate.quantity
        for candidate in candidates
        if candidate.lot_id in rejected_ids
    )
    if input_quantity != sum(batch.total_quantity for batch in batches) + unassigned_quantity:
        raise RuntimeError("furnace planner quantity conservation failed")

    return FurnacePlan(
        request_id=request_id,
        batches=batches,
        unassigned=tuple(sorted(unassigned, key=lambda item: item.lot_id)),
        at=at,
    )


def _packable_lot_ids(
    tube: FurnaceTube,
    members: Sequence[_ScreenedCandidate],
    matrix: Mapping[str, frozenset[str]],
) -> frozenset[str]:
    """Lot ids that belong to at least one feasible pack within this group."""

    if not members or len(members) > _ENUMERATION_LIMIT:
        # For large groups, the deterministic fill result is the feasibility
        # witness; leftovers there are contention rather than structural gaps.
        pack = _order_fill(tube, members, matrix)
        return frozenset(pack.lot_ids) if pack is not None else frozenset()

    found: set[str] = set()
    nodes = 0

    def extend(
        index: int,
        chosen: list[int],
        load: int,
        start: int,
        end: int,
        contamination: frozenset[str],
    ) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > _NODE_BUDGET:
            raise _NodeBudgetExceeded
        if load >= tube.min_load:
            found.update(members[pos].candidate.lot_id for pos in chosen)
        if index == len(members) or load == tube.max_load:
            return
        for next_index in range(index, len(members)):
            member = members[next_index]
            klass = member.candidate.contamination_class
            if any(not _compatible(klass, prior, matrix) for prior in contamination):
                continue
            next_start = max(start, member.start)
            next_end = min(end, member.end)
            next_load = load + member.candidate.quantity
            if next_start >= next_end or next_load > tube.max_load:
                continue
            chosen.append(next_index)
            extend(
                next_index + 1,
                chosen,
                next_load,
                next_start,
                next_end,
                contamination | {klass},
            )
            chosen.pop()

    try:
        extend(
            0,
            [],
            0,
            tube.available_start,
            tube.available_end if tube.available_end is not None else FAR_FUTURE,
            frozenset(),
        )
    except _NodeBudgetExceeded:
        pack = _order_fill(tube, members, matrix)
        return frozenset(pack.lot_ids) if pack is not None else frozenset()

    return frozenset(found)

"""Furnace-tube batch formation for candidate work-in-process lots.

Lots are never packed in arrival order.  Candidate lots are first arranged by
a deterministic key (due window, then lot id), grouped by recipe family, and
then placed onto tubes whose load limits, accepted recipe families and
equipment qualifications fit.  A batch is only launched once its load reaches
the tube minimum; underfilled batches receive a salvage pass that relocates
their lots into already launchable batches or merges two underfilled batches
on a shared tube.  Every lot is either launched in exactly one batch or
returned in the disposition list with a concrete reason, and the two lists
together conserve lot quantity.
"""

from __future__ import annotations

from dataclasses import dataclass

from fabweave.contamination.segregation import incompatible_pairs
from fabweave.core.decision import Decision, allow, deny
from fabweave.core.ids import deterministic_id
from fabweave.core.quantities import require_conservation


@dataclass(frozen=True, slots=True)
class FurnaceLot:
    lot_id: str
    recipe_family: str
    contamination_class: str
    quantity: int
    window_open: int
    window_close: int
    required_qualifications: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.lot_id:
            raise ValueError("lot id is required")
        if self.quantity <= 0:
            raise ValueError("lot quantity must be positive")
        if self.window_open > self.window_close:
            raise ValueError("due window is inverted")
        if not self.recipe_family or not self.contamination_class:
            raise ValueError("recipe family and contamination class are required")


@dataclass(frozen=True, slots=True)
class FurnaceTube:
    tube_id: str
    min_load: int
    max_load: int
    accepted_families: frozenset[str] = frozenset()
    qualifications: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.tube_id:
            raise ValueError("tube id is required")
        if self.min_load <= 0 or self.max_load < self.min_load:
            raise ValueError("tube load limits are invalid")


@dataclass(frozen=True, slots=True)
class FurnaceBatch:
    batch_id: str
    tube_id: str
    recipe_family: str
    lot_ids: tuple[str, ...]
    total_quantity: int
    contamination_classes: frozenset[str]
    window_open: int
    window_close: int


@dataclass(frozen=True, slots=True)
class LotDisposition:
    lot_id: str
    quantity: int
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BatchPlan:
    batches: tuple[FurnaceBatch, ...]
    unassigned: tuple[LotDisposition, ...]

    @property
    def launched_lot_ids(self) -> frozenset[str]:
        return frozenset(lot_id for batch in self.batches for lot_id in batch.lot_ids)


def tube_eligibility(lot: FurnaceLot, tube: FurnaceTube) -> Decision:
    """Whether a single lot may run on a tube, independent of its companions.

    Contamination conflicts only arise against companions, not against the tube
    itself, so they are enforced at batch assembly time rather than here.
    """
    reasons: list[str] = []
    if tube.accepted_families and lot.recipe_family not in tube.accepted_families:
        reasons.append("recipe_family_rejected")
    if lot.quantity > tube.max_load:
        reasons.append("quantity_exceeds_max_load")
    missing = lot.required_qualifications - tube.qualifications
    if missing:
        reasons.append("tube_qualification_missing")
    if reasons:
        return deny(*reasons, missing_qualifications=tuple(sorted(missing)))
    return allow()


def _compatible_classes(classes: frozenset[str], matrix: dict[str, set[str]]) -> bool:
    return not incompatible_pairs(set(classes), matrix)


@dataclass(slots=True)
class _OpenBatch:
    tube: FurnaceTube
    recipe_family: str
    lot_ids: list[str]
    total_quantity: int
    contamination_classes: set[str]
    window_open: int
    window_close: int

    @classmethod
    def empty(cls, tube: FurnaceTube, family: str) -> "_OpenBatch":
        return cls(tube, family, [], 0, set(), 0, 0)

    def can_take(
        self,
        lot: FurnaceLot,
        matrix: dict[str, set[str]],
    ) -> tuple[bool, tuple[str, ...]]:
        if not self.lot_ids:
            return True, ()
        open_at = max(self.window_open, lot.window_open)
        close_at = min(self.window_close, lot.window_close)
        if open_at > close_at:
            return False, ("due_window_no_overlap",)
        if self.total_quantity + lot.quantity > self.tube.max_load:
            return False, ("capacity_exceeded",)
        if not _compatible_classes(
            frozenset(self.contamination_classes | {lot.contamination_class}), matrix
        ):
            return False, ("contamination_incompatible",)
        return True, ()

    def add(self, lot: FurnaceLot) -> None:
        if self.lot_ids:
            self.window_open = max(self.window_open, lot.window_open)
            self.window_close = min(self.window_close, lot.window_close)
        else:
            self.window_open = lot.window_open
            self.window_close = lot.window_close
        self.lot_ids.append(lot.lot_id)
        self.total_quantity += lot.quantity
        self.contamination_classes.add(lot.contamination_class)

    @property
    def launchable(self) -> bool:
        return bool(self.lot_ids) and self.total_quantity >= self.tube.min_load


def _lot_sort_key(lot: FurnaceLot) -> tuple[int, int, str]:
    return (lot.window_close, lot.window_open, lot.lot_id)


def _eligible_tube_ids(
    lot: FurnaceLot,
    tubes: list[FurnaceTube],
) -> tuple[set[str], dict[str, Decision]]:
    decisions = {tube.tube_id: tube_eligibility(lot, tube) for tube in tubes}
    eligible = {tube_id for tube_id, decision in decisions.items() if decision.allowed}
    return eligible, decisions


def build_batch_plan(
    request_id: str,
    lots: list[FurnaceLot],
    tubes: list[FurnaceTube],
    contamination_matrix: dict[str, set[str]],
) -> BatchPlan:
    """Pack candidate lots into launchable furnace batches deterministically.

    The result is independent of the order of ``lots`` and ``tubes``: lots are
    processed by (window close, window open, lot id) and every tie among tubes
    or batches is broken by id.  Each lot appears in exactly one batch or in
    exactly one unassigned disposition, and quantities are conserved.
    """
    if not request_id:
        raise ValueError("request id is required")
    if not tubes:
        raise ValueError("at least one tube is required")
    if len({lot.lot_id for lot in lots}) != len(lots):
        raise ValueError("the same lot cannot be batched twice")
    if len({tube.tube_id for tube in tubes}) != len(tubes):
        raise ValueError("duplicate tube id")

    tubes_by_id = {tube.tube_id: tube for tube in tubes}
    lots_by_id = {lot.lot_id: lot for lot in lots}
    open_batches: list[_OpenBatch] = []
    eligibility: dict[str, dict[str, Decision]] = {}

    def placement_target(
        lot: FurnaceLot, eligible: set[str]
    ) -> tuple[_OpenBatch | None, str | None]:
        """Return the batch to extend, or the tube id to open a new batch on."""
        candidates = sorted(
            (
                batch
                for batch in open_batches
                if batch.recipe_family == lot.recipe_family
                and batch.tube.tube_id in eligible
            ),
            key=lambda batch: (batch.tube.tube_id, batch.window_close, batch.total_quantity),
        )
        for batch in candidates:
            fits, _ = batch.can_take(lot, contamination_matrix)
            if fits:
                return batch, None
        # Open on the eligible tube with the smallest minimum load so lots
        # concentrate where a launchable batch is easiest to reach.
        openings = [tubes_by_id[tube_id] for tube_id in eligible]
        if not openings:
            return None, None
        tube = min(openings, key=lambda item: (item.min_load, item.tube_id))
        return None, tube.tube_id

    for family in sorted({lot.recipe_family for lot in lots}):
        family_lots = sorted(
            (lot for lot in lots if lot.recipe_family == family),
            key=_lot_sort_key,
        )
        for lot in family_lots:
            eligible, decisions = _eligible_tube_ids(lot, tubes)
            eligibility[lot.lot_id] = decisions
            if not eligible:
                continue
            batch, new_tube_id = placement_target(lot, eligible)
            if batch is None:
                if new_tube_id is None:
                    continue
                batch = _OpenBatch.empty(tubes_by_id[new_tube_id], family)
                open_batches.append(batch)
            batch.add(lot)

    _salvage_underfilled(open_batches, lots_by_id, contamination_matrix)

    launched = sorted(
        (batch for batch in open_batches if batch.launchable),
        key=lambda item: (item.recipe_family, item.tube.tube_id),
    )
    emitted: list[FurnaceBatch] = []
    for index, batch in enumerate(launched):
        emitted.append(
            FurnaceBatch(
                batch_id=deterministic_id(
                    "BATCH", request_id, batch.tube.tube_id, batch.recipe_family, index
                ),
                tube_id=batch.tube.tube_id,
                recipe_family=batch.recipe_family,
                lot_ids=tuple(sorted(batch.lot_ids)),
                total_quantity=batch.total_quantity,
                contamination_classes=frozenset(batch.contamination_classes),
                window_open=batch.window_open,
                window_close=batch.window_close,
            )
        )

    launched_ids = {lot_id for batch in emitted for lot_id in batch.lot_ids}
    leftover_lots = sorted(
        (lot for lot in lots if lot.lot_id not in launched_ids),
        key=_lot_sort_key,
    )
    dispositions = [
        _disposition(
            lot,
            lots,
            launched_ids,
            contamination_matrix,
            eligibility,
            emitted,
            tubes_by_id,
        )
        for lot in leftover_lots
    ]

    plan = BatchPlan(tuple(emitted), tuple(dispositions))
    _require_plan_conservation(lots, plan)
    _validate_emitted(plan, lots, tubes_by_id, contamination_matrix)
    return plan


def _salvage_underfilled(
    open_batches: list[_OpenBatch],
    lots_by_id: dict[str, FurnaceLot],
    matrix: dict[str, set[str]],
) -> None:
    """Pull lots out of underfilled batches until no more movement is possible.

    Each iteration first relocates individual lots into already launchable
    batches, then attempts to dissolve an entire underfilled batch (merging it
    into companions that may themselves be short).  A full dissolution is
    evaluated on a working copy and only committed when every lot fits, so the
    pass never leaves a target in an invalid state.  Repeated until stable.
    """

    def targets_for(lot: FurnaceLot, source: _OpenBatch) -> list[_OpenBatch]:
        return sorted(
            (
                batch
                for batch in open_batches
                if batch is not source
                and batch.recipe_family == source.recipe_family
                and (not batch.tube.accepted_families or lot.recipe_family in batch.tube.accepted_families)
                and lot.required_qualifications <= batch.tube.qualifications
            ),
            key=lambda item: (0 if item.launchable else 1, item.tube.tube_id),
        )

    while True:
        progressed = False

        # Pass 1: move individual lots into batches that already launch.
        for source in sorted(
            (batch for batch in open_batches if not batch.launchable and batch.lot_ids),
            key=lambda item: (item.recipe_family, item.tube.tube_id),
        ):
            for lot_id in list(source.lot_ids):
                lot = lots_by_id[lot_id]
                for target in targets_for(lot, source):
                    if not target.launchable:
                        continue
                    fits, _ = target.can_take(lot, matrix)
                    if fits:
                        target.add(lot)
                        _detach(source, lot_id, lots_by_id)
                        progressed = True
                        break
            if source in open_batches and not source.lot_ids:
                open_batches.remove(source)

        # Pass 2: dissolve an entire short batch atomically.
        for source in sorted(
            (batch for batch in open_batches if not batch.launchable),
            key=lambda item: (item.recipe_family, item.tube.tube_id, item.total_quantity),
        ):
            if source.launchable or not source.lot_ids:
                continue
            # Simulate every move on working copies of the destinations; only
            # commit when the source as a whole can be relocated.
            working: dict[int, tuple[_OpenBatch, _OpenBatch]] = {}
            plan_moves: list[tuple[_OpenBatch, FurnaceLot]] = []
            blocked = False
            for lot_id in source.lot_ids:
                lot = lots_by_id[lot_id]
                destination = None
                for target in targets_for(lot, source):
                    _, scratch = working.setdefault(id(target), (target, _snapshot(target)))
                    fits, _ = scratch.can_take(lot, matrix)
                    if fits:
                        scratch.add(lot)
                        destination = target
                        break
                if destination is None:
                    blocked = True
                    break
                plan_moves.append((destination, lot))
            if blocked:
                continue
            for destination, lot in plan_moves:
                destination.add(lot)
            open_batches.remove(source)
            progressed = True
            break

        if not progressed:
            empties = [batch for batch in open_batches if not batch.lot_ids]
            if empties:
                for batch in empties:
                    open_batches.remove(batch)
                continue
            return


def _snapshot(batch: _OpenBatch) -> _OpenBatch:
    return _OpenBatch(
        tube=batch.tube,
        recipe_family=batch.recipe_family,
        lot_ids=list(batch.lot_ids),
        total_quantity=batch.total_quantity,
        contamination_classes=set(batch.contamination_classes),
        window_open=batch.window_open,
        window_close=batch.window_close,
    )


def _detach(
    batch: _OpenBatch,
    lot_id: str,
    lots_by_id: dict[str, FurnaceLot],
) -> None:
    if lot_id not in batch.lot_ids:
        return
    batch.lot_ids.remove(lot_id)
    lot = lots_by_id[lot_id]
    batch.total_quantity -= lot.quantity
    if batch.lot_ids:
        batch.window_open = max(lots_by_id[item].window_open for item in batch.lot_ids)
        batch.window_close = min(lots_by_id[item].window_close for item in batch.lot_ids)
        batch.contamination_classes = {
            lots_by_id[item].contamination_class for item in batch.lot_ids
        }


def _disposition(
    lot: FurnaceLot,
    all_lots: list[FurnaceLot],
    launched_ids: set[str],
    matrix: dict[str, set[str]],
    eligibility: dict[str, dict[str, Decision]],
    emitted: list[FurnaceBatch],
    tubes_by_id: dict[str, FurnaceTube],
) -> LotDisposition:
    decisions = eligibility[lot.lot_id]
    if not any(decision.allowed for decision in decisions.values()):
        reasons = tuple(
            sorted({reason for decision in decisions.values() for reason in decision.reasons})
        )
        return LotDisposition(lot.lot_id, lot.quantity, ("no_eligible_tube", *reasons))

    same_family = [
        batch
        for batch in emitted
        if batch.recipe_family == lot.recipe_family and lot.lot_id not in batch.lot_ids
    ]
    if same_family:
        blockers: set[str] = set()
        for batch in same_family:
            tube = tubes_by_id[batch.tube_id]
            decision = decisions[batch.tube_id]
            if not decision.allowed:
                blockers.update(decision.reasons)
                continue
            if lot.quantity + batch.total_quantity > tube.max_load:
                blockers.add("capacity_exceeded")
            window_open = max(batch.window_open, lot.window_open)
            window_close = min(batch.window_close, lot.window_close)
            if window_open > window_close:
                blockers.add("due_window_no_overlap")
            classes = batch.contamination_classes | {lot.contamination_class}
            if not _compatible_classes(classes, matrix):
                blockers.add("contamination_incompatible")
        if not blockers:
            blockers.add("no_compatible_batch")
        return LotDisposition(lot.lot_id, lot.quantity, tuple(sorted(blockers)))

    # No launched batch of this family exists: explain why the lot could not be
    # combined with its unlaunched same-family peers into a minimum-load batch.
    peers = [
        other
        for other in all_lots
        if other.lot_id != lot.lot_id
        and other.lot_id not in launched_ids
        and other.recipe_family == lot.recipe_family
    ]
    peer_blockers: set[str] = set()
    for peer in peers:
        shared = {
            tube_id
            for tube_id, decision in decisions.items()
            if decision.allowed and eligibility[peer.lot_id][tube_id].allowed
        }
        if not shared:
            peer_blockers.add("companion_no_shared_qualified_tube")
            continue
        if max(lot.window_open, peer.window_open) > min(lot.window_close, peer.window_close):
            peer_blockers.add("due_window_no_overlap")
        if not _compatible_classes(
            frozenset({lot.contamination_class, peer.contamination_class}), matrix
        ):
            peer_blockers.add("contamination_incompatible")
        if all(
            lot.quantity + peer.quantity > tubes_by_id[tube_id].max_load for tube_id in shared
        ):
            peer_blockers.add("capacity_exceeded")

    # Best achievable load per eligible tube, using every peer that could share it.
    best_pool = lot.quantity
    for tube_id, decision in decisions.items():
        if not decision.allowed:
            continue
        tube = tubes_by_id[tube_id]
        pool = lot.quantity
        for peer in peers:
            if not eligibility[peer.lot_id][tube_id].allowed:
                continue
            if max(lot.window_open, peer.window_open) > min(lot.window_close, peer.window_close):
                continue
            if not _compatible_classes(
                frozenset({lot.contamination_class, peer.contamination_class}), matrix
            ):
                continue
            if pool + peer.quantity <= tube.max_load:
                pool += peer.quantity
        best_pool = max(best_pool, pool)

    reasons = set(peer_blockers)
    if best_pool < min(tubes_by_id[t].min_load for t, d in decisions.items() if d.allowed):
        reasons.add("insufficient_compatible_wip_for_minimum_load")
    if not reasons:
        reasons.add("no_compatible_batch")
    return LotDisposition(lot.lot_id, lot.quantity, tuple(sorted(reasons)))


def _require_plan_conservation(lots: list[FurnaceLot], plan: BatchPlan) -> None:
    before = {lot.lot_id: lot.quantity for lot in lots}
    quantities = {lot.lot_id: lot.quantity for lot in lots}
    after_launched = {
        lot_id: quantities[lot_id]
        for batch in plan.batches
        for lot_id in batch.lot_ids
    }
    after_left = {item.lot_id: item.quantity for item in plan.unassigned}
    require_conservation(before, {**after_launched, **after_left})
    launched = set(after_launched)
    if launched & set(after_left):
        raise ValueError("lot cannot be both launched and unassigned")
    if launched | set(after_left) != set(before):
        raise ValueError("every lot must be launched or assigned a disposition")
    usage: dict[str, int] = {}
    for batch in plan.batches:
        for lot_id in batch.lot_ids:
            usage[lot_id] = usage.get(lot_id, 0) + 1
    if any(count > 1 for count in usage.values()):
        raise ValueError("a lot cannot be reused across batches")


def _validate_emitted(
    plan: BatchPlan,
    lots: list[FurnaceLot],
    tubes_by_id: dict[str, FurnaceTube],
    matrix: dict[str, set[str]],
) -> None:
    """Fail-closed re-check of every invariant a launched batch must satisfy."""
    lots_by_id = {lot.lot_id: lot for lot in lots}
    for batch in plan.batches:
        tube = tubes_by_id[batch.tube_id]
        members = [lots_by_id[lot_id] for lot_id in batch.lot_ids]
        if not tube.min_load <= batch.total_quantity <= tube.max_load:
            raise ValueError("emitted batch violates tube load limits")
        if {lot.recipe_family for lot in members} != {batch.recipe_family}:
            raise ValueError("emitted batch mixes recipe families")
        if tube.accepted_families and batch.recipe_family not in tube.accepted_families:
            raise ValueError("emitted batch on a tube that rejects its family")
        if any(lot.required_qualifications - tube.qualifications for lot in members):
            raise ValueError("emitted batch on an under-qualified tube")
        window_open = max(lot.window_open for lot in members)
        window_close = min(lot.window_close for lot in members)
        if not window_open <= window_close:
            raise ValueError("emitted batch has no common due window")
        if (window_open, window_close) != (batch.window_open, batch.window_close):
            raise ValueError("emitted batch window disagrees with its lots")
        classes = {lot.contamination_class for lot in members}
        if not _compatible_classes(frozenset(classes), matrix):
            raise ValueError("emitted batch mixes incompatible contamination classes")

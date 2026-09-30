from __future__ import annotations

import pytest

from fabweave.furnace.batching import (
    CONTAMINATION_NOT_SUPPORTED,
    DEFERRED_CAPACITY_CONTENTION,
    FurnaceCandidate,
    FurnaceTube,
    INSUFFICIENT_COMPATIBLE_LOAD,
    NO_AVAILABLE_WINDOW,
    RECIPE_UNSUPPORTED,
    WINDOW_CLOSED,
    plan_furnace_batches,
)

TUBE_1 = FurnaceTube("TUBE-1", 10, 25, frozenset({"OX", "NIT"}))


def _lot(lot_id: str, qty: int, *, family: str = "OX", klass: str = "GENERAL",
         release_at: int = 0, due_at: int | None = None) -> FurnaceCandidate:
    return FurnaceCandidate(lot_id, family, qty, klass, release_at, due_at)


def _reasons(plan):
    return {item.lot_id: item.reasons[0] for item in plan.unassigned}


def test_batch_respects_min_and_max_load_and_conserves_quantity():
    lots = [_lot("L-A", 6), _lot("L-B", 7), _lot("L-C", 20)]
    plan = plan_furnace_batches("REQ-1", lots, [TUBE_1])
    assert len(plan.batches) == 1
    batch = plan.batches[0]
    assert 10 <= batch.total_quantity <= 25
    assert batch.recipe_family == "OX"
    assert batch.tube_id == "TUBE-1"
    assigned = sum(batch.total_quantity for batch in plan.batches)
    deferred = sum(lot.quantity for lot in lots if lot.lot_id in plan.unassigned_lot_ids)
    assert assigned + deferred == 33


def test_result_is_independent_of_input_order():
    lots = [
        _lot(f"L-{i}", qty, due_at=due)
        for i, (qty, due) in enumerate(
            [(8, 90), (5, 30), (12, 60), (9, 20), (6, 120), (11, 45)], start=1
        )
    ]
    first = plan_furnace_batches("REQ-1", lots, [TUBE_1])
    shuffled = [lots[4], lots[0], lots[5], lots[2], lots[1], lots[3]]
    second = plan_furnace_batches("REQ-1", shuffled, [TUBE_1])
    assert first.batch_ids == second.batch_ids
    assert [sorted(batch.lot_ids) for batch in first.batches] == [
        sorted(batch.lot_ids) for batch in second.batches
    ]
    assert first.unassigned_lot_ids == second.unassigned_lot_ids


def test_lot_below_minimum_without_compatible_partner_is_explained():
    plan = plan_furnace_batches("REQ-1", [_lot("LONE", 5)], [TUBE_1])
    assert plan.batches == ()
    assert _reasons(plan) == {"LONE": INSUFFICIENT_COMPATIBLE_LOAD}
    assert plan.destination("LONE") == "UNASSIGNED_WIP"


def test_recipe_families_are_never_mixed_in_a_batch():
    tube = FurnaceTube("TUBE-1", 10, 25, frozenset({"OX", "NIT"}))
    lots = [_lot("RA", 8, family="OX"), _lot("RB", 8, family="NIT")]
    plan = plan_furnace_batches("REQ-1", lots, [tube])
    assert plan.batches == ()
    assert set(_reasons(plan)) == {"RA", "RB"}
    assert all(reason == INSUFFICIENT_COMPATIBLE_LOAD for reason in _reasons(plan).values())


def test_unsupported_recipe_is_a_hard_reject():
    lots = [_lot("RA", 12, family="POLY")]
    plan = plan_furnace_batches("REQ-1", lots, [TUBE_1])
    assert _reasons(plan) == {"RA": RECIPE_UNSUPPORTED}


def test_contamination_classes_are_segregated_and_hard_restrictions_reported():
    tube = FurnaceTube(
        "TUBE-1", 10, 25, frozenset({"OX"}), allowed_contamination=frozenset({"GENERAL", "CU"})
    )
    lots = [
        _lot("CU-A", 8, klass="CU"),
        _lot("FE-A", 8, klass="FE"),
        _lot("FE-B", 8, klass="FE"),
    ]
    plan = plan_furnace_batches(
        "REQ-1", lots, [tube], contamination_matrix={"CU": {"FE"}}
    )
    reasons = _reasons(plan)
    assert reasons["FE-A"] == CONTAMINATION_NOT_SUPPORTED
    assert reasons["FE-B"] == CONTAMINATION_NOT_SUPPORTED
    # CU-A alone cannot reach the minimum load.
    assert reasons["CU-A"] == INSUFFICIENT_COMPATIBLE_LOAD


def test_mutually_incompatible_classes_cannot_share_a_batch_even_when_tube_allows_both():
    tube = FurnaceTube(
        "TUBE-1", 10, 25, frozenset({"OX"}),
        allowed_contamination=frozenset({"CU", "FE"}),
    )
    lots = [_lot("CU-A", 6, klass="CU"), _lot("FE-A", 6, klass="FE")]
    plan = plan_furnace_batches(
        "REQ-1", lots, [tube], contamination_matrix={"CU": {"FE"}}
    )
    assert plan.batches == ()
    assert all(reason == INSUFFICIENT_COMPATIBLE_LOAD for reason in _reasons(plan).values())


def test_compatible_classes_can_share_a_batch():
    tube = FurnaceTube(
        "TUBE-1", 10, 25, frozenset({"OX"}),
        allowed_contamination=frozenset({"GENERAL", "CU"}),
    )
    plan = plan_furnace_batches(
        "REQ-1",
        [_lot("G-A", 6), _lot("CU-A", 6, klass="CU")],
        [tube],
        contamination_matrix={"CU": {"FE"}},
    )
    assert len(plan.batches) == 1
    assert plan.batches[0].contamination_classes == frozenset({"GENERAL", "CU"})


def test_due_window_must_have_a_common_processing_interval():
    tube = FurnaceTube("TUBE-1", 10, 25, frozenset({"OX"}), available_start=0, available_end=100)
    lots = [
        _lot("EARLY", 6, release_at=0, due_at=10),
        _lot("LATE", 6, release_at=10, due_at=20),  # windows only touch at 10
    ]
    plan = plan_furnace_batches("REQ-1", lots, [tube])
    assert plan.batches == ()
    assert all(reason == INSUFFICIENT_COMPATIBLE_LOAD for reason in _reasons(plan).values())


def test_closed_due_window_and_no_equipment_window_are_distinct_rejects():
    lots = [
        _lot("PAST", 12, due_at=5),
        _lot("BUSY", 12, release_at=20, due_at=30),
    ]
    tube = FurnaceTube("TUBE-1", 10, 25, frozenset({"OX"}), available_start=0, available_end=20)
    plan = plan_furnace_batches("REQ-1", lots, [tube], at=10)
    reasons = _reasons(plan)
    assert reasons["PAST"] == WINDOW_CLOSED
    assert reasons["BUSY"] == NO_AVAILABLE_WINDOW


def test_batch_window_lies_inside_every_lot_and_tube_window():
    tube = FurnaceTube("TUBE-1", 10, 25, frozenset({"OX"}), available_start=5, available_end=100)
    lots = [
        _lot("A", 6, release_at=0, due_at=80),
        _lot("B", 6, release_at=15, due_at=60),
    ]
    plan = plan_furnace_batches("REQ-1", lots, [tube])
    batch = plan.batches[0]
    assert batch.window.start >= 15 and batch.window.end <= 60
    assert tube.available_start <= batch.window.start
    assert batch.window.end <= tube.available_end


def test_equipment_matching_routes_lots_to_the_qualified_tube():
    tubes = [
        FurnaceTube("TUBE-A", 10, 25, frozenset({"NIT"})),
        FurnaceTube("TUBE-B", 10, 25, frozenset({"OX"})),
    ]
    plan = plan_furnace_batches("REQ-1", [_lot("A", 12, family="OX")], tubes)
    assert plan.batches[0].tube_id == "TUBE-B"


def test_one_batch_per_tube_and_capacity_leftover_is_deferred_with_reason():
    tubes = [
        FurnaceTube("TUBE-A", 10, 15, frozenset({"OX"})),
        FurnaceTube("TUBE-B", 10, 15, frozenset({"OX"})),
    ]
    lots = [_lot(f"L-{i}", 10, due_at=due) for i, due in enumerate((30, 60, 90), start=1)]
    plan = plan_furnace_batches("REQ-1", lots, tubes)
    assert len(plan.batches) == 2
    assert {batch.total_quantity for batch in plan.batches} == {10, 10}
    reasons = _reasons(plan)
    assert reasons == {"L-3": DEFERRED_CAPACITY_CONTENTION}
    all_lots = [lot_id for batch in plan.batches for lot_id in batch.lot_ids]
    assert len(all_lots) == len(set(all_lots)) == 2


def test_planner_protects_earliest_due_wip_rather_than_arrival_order():
    tube = FurnaceTube("TUBE-1", 5, 20, frozenset({"OX"}))
    # Arrival/input order is newest-due first; the optimizer must not honor it.
    lots = [
        _lot("LATE", 10, due_at=200),
        _lot("MID", 10, due_at=100),
        _lot("EARLY", 10, due_at=50),
    ]
    plan = plan_furnace_batches("REQ-1", lots, [tube])
    assert sorted(plan.assigned_lot_ids) == ["EARLY", "MID"]
    assert plan.unassigned_lot_ids == ("LATE",)


def test_destination_is_defined_for_every_candidate():
    lots = [_lot("A", 12), _lot("B", 5), _lot("C", 12, family="POLY")]
    plan = plan_furnace_batches("REQ-1", lots, [TUBE_1])
    for lot in lots:
        destination = plan.destination(lot.lot_id)
        assert destination.startswith("BATCH-") or destination == "UNASSIGNED_WIP"
    with pytest.raises(KeyError):
        plan.destination("MISSING")


def test_invalid_requests_fail_closed():
    with pytest.raises(ValueError):
        plan_furnace_batches("REQ-1", [_lot("A", 1)], [])
    with pytest.raises(ValueError):
        plan_furnace_batches("REQ-1", [_lot("A", 1), _lot("A", 2)], [TUBE_1])
    with pytest.raises(ValueError):
        FurnaceTube("BAD", 25, 10, frozenset({"OX"}))
    with pytest.raises(ValueError):
        _lot("BAD", 0)

from __future__ import annotations

import random

import pytest

from fabweave.furnace.batching import (
    BatchPlan,
    FurnaceLot,
    FurnaceTube,
    build_batch_plan,
    tube_eligibility,
)


def lot(lot_id, family="OX", clazz="FE", qty=25, open_at=0, close_at=100, quals=frozenset()):
    return FurnaceLot(lot_id, family, clazz, qty, open_at, close_at, quals)


def tube(tube_id="T1", minimum=50, maximum=100, families=frozenset(), quals=frozenset()):
    return FurnaceTube(tube_id, minimum, maximum, families, quals)


def _reorder(items, seed):
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def test_basic_plan_launches_batches_within_load_limits():
    lots = [lot("A"), lot("B"), lot("C", qty=10)]
    plan = build_batch_plan("REQ", lots, [tube()], {})
    assert len(plan.batches) == 1
    batch = plan.batches[0]
    assert batch.lot_ids == ("A", "B", "C")
    assert batch.total_quantity == 60
    assert plan.unassigned == ()

    # A single lot below the minimum cannot launch on its own.
    plan = build_batch_plan("REQ", [lot("SOLO", qty=10)], [tube()], {})
    assert plan.batches == ()
    assert plan.unassigned[0].reasons == (
        "insufficient_compatible_wip_for_minimum_load",
    )


def test_plan_is_independent_of_input_order():
    lots = [
        lot("A", close_at=90),
        lot("B", close_at=30),
        lot("C", close_at=60, qty=60),
        lot("D", close_at=30, qty=60),
    ]
    tubes = [tube("T1"), tube("T2")]
    reference = build_batch_plan("REQ", lots, tubes, {})
    for seed in range(10):
        shuffled = build_batch_plan("REQ", _reorder(lots, seed), _reorder(tubes, seed + 1), {})
        assert [batch.batch_id for batch in shuffled.batches] == [
            batch.batch_id for batch in reference.batches
        ]
        assert [batch.lot_ids for batch in shuffled.batches] == [
            batch.lot_ids for batch in reference.batches
        ]
        assert [item.lot_id for item in shuffled.unassigned] == [
            item.lot_id for item in reference.unassigned
        ]


def test_recipe_families_are_never_mixed():
    lots = [lot("A", family="OX", qty=30), lot("B", family="NIT", qty=30)]
    plan = build_batch_plan("REQ", lots, [tube()], {})
    assert plan.batches == ()
    assert {item.reasons[0] for item in plan.unassigned} == {
        "insufficient_compatible_wip_for_minimum_load"
    }

    lots = [lot("A", family="OX", qty=30), lot("B", family="NIT", qty=30),
            lot("C", family="OX", qty=30), lot("D", family="NIT", qty=30)]
    plan = build_batch_plan("REQ", lots, [tube()], {})
    assert {batch.recipe_family for batch in plan.batches} == {"OX", "NIT"}
    assert all(len(batch.lot_ids) == 2 for batch in plan.batches)


def test_contamination_classes_are_segregated():
    matrix = {"CU": {"FE"}}
    lots = [lot("A", clazz="CU", qty=30), lot("B", clazz="FE", qty=30)]
    plan = build_batch_plan("REQ", lots, [tube()], matrix)
    assert plan.batches == ()
    reasons = {item.lot_id: item.reasons for item in plan.unassigned}
    assert "contamination_incompatible" in reasons["A"]
    assert "insufficient_compatible_wip_for_minimum_load" in reasons["A"]
    assert "contamination_incompatible" in reasons["B"]


def test_compatible_contamination_classes_share_a_batch():
    matrix = {"CU": {"FE"}}
    lots = [lot("A", clazz="CU", qty=30), lot("B", clazz="CU", qty=30)]
    plan = build_batch_plan("REQ", lots, [tube()], matrix)
    assert len(plan.batches) == 1
    assert plan.batches[0].contamination_classes == frozenset({"CU"})
    assert plan.unassigned == ()


def test_due_windows_must_share_a_common_interval():
    lots = [
        lot("A", open_at=0, close_at=10, qty=40),
        lot("B", open_at=20, close_at=40, qty=40),
        lot("C", open_at=0, close_at=40, qty=20),
    ]
    plan = build_batch_plan("REQ", lots, [tube()], {})
    # A and B cannot share a batch; C can only rescue one of them to 60.
    launched = plan.launched_lot_ids
    assert "C" in launched
    assert launched & {"A", "B"}
    leftover = next(item for item in plan.unassigned if item.lot_id in {"A", "B"})
    assert "due_window_no_overlap" in leftover.reasons


def test_tube_qualification_and_family_must_match():
    # Both lots are OX but their required qualifications split the tools:
    # NEEDQ demands Q-HIGH (only Q has it) and PLAIN demands P-ONLY (only P
    # has it).  They share no qualified tube and each is below the minimum.
    tubes = [
        tube("Q", families=frozenset({"OX"}), quals=frozenset({"Q-HIGH"})),
        tube("P", families=frozenset({"OX"}), quals=frozenset({"P-ONLY"})),
    ]
    lots = [
        lot("NEEDQ", qty=30, quals=frozenset({"Q-HIGH"})),
        lot("PLAIN", qty=30, quals=frozenset({"P-ONLY"})),
    ]
    plan = build_batch_plan("REQ", lots, tubes, {})
    assert plan.batches == ()
    reasons = {item.lot_id: item.reasons for item in plan.unassigned}
    assert "companion_no_shared_qualified_tube" in reasons["NEEDQ"]
    assert "companion_no_shared_qualified_tube" in reasons["PLAIN"]

    # A lot requiring a qualification no tube carries is rejected outright.
    plan = build_batch_plan(
        "REQ",
        [lot("X", qty=60, quals=frozenset({"MISSING"}))],
        [tube()],
        {},
    )
    assert plan.batches == ()
    assert "no_eligible_tube" in plan.unassigned[0].reasons
    assert "tube_qualification_missing" in plan.unassigned[0].reasons

    # A family no tube accepts is also rejected outright.
    plan = build_batch_plan(
        "REQ",
        [lot("Y", family="NIT", qty=60)],
        [tube("ONLY-OX", families=frozenset({"OX"}))],
        {},
    )
    assert plan.batches == ()
    assert "recipe_family_rejected" in plan.unassigned[0].reasons


def test_lot_above_max_load_cannot_be_launched():
    lots = [lot("BIG", qty=150), lot("A", qty=60)]
    plan = build_batch_plan("REQ", lots, [tube()], {})
    assert plan.launched_lot_ids == {"A"}
    assert plan.unassigned[0].lot_id == "BIG"
    assert "quantity_exceeds_max_load" in plan.unassigned[0].reasons
    assert not tube_eligibility(lot("BIG", qty=150), tube()).allowed


def test_each_lot_is_used_at_most_once_and_quantity_conserved():
    lots = [lot(f"L{i}", qty=20 + (i % 3) * 10) for i in range(12)]
    plan = build_batch_plan("REQ", lots, [tube("T1"), tube("T2", minimum=40)], {})
    seen: dict[str, int] = {}
    launched_quantity = 0
    for batch in plan.batches:
        assert 50 <= batch.total_quantity <= 100 or batch.tube_id == "T2"
        for lot_id in batch.lot_ids:
            seen[lot_id] = seen.get(lot_id, 0) + 1
            launched_quantity += next(lot.quantity for lot in lots if lot.lot_id == lot_id)
    assert all(count == 1 for count in seen.values())
    assert launched_quantity + sum(item.quantity for item in plan.unassigned) == sum(
        lot.quantity for lot in lots
    )
    assert not (set(seen) & {item.lot_id for item in plan.unassigned})


def test_underfilled_batches_are_salvaged_into_launchable_companions():
    lots = [
        lot("A", qty=50),  # reaches the minimum on its own
        lot("B", qty=20),
        lot("C", qty=20),  # B+C alone stay under the minimum
    ]
    plan = build_batch_plan("REQ", lots, [tube()], {})
    assert len(plan.batches) == 1
    assert plan.batches[0].lot_ids == ("A", "B", "C")
    assert plan.batches[0].total_quantity == 90
    assert plan.unassigned == ()


def test_deterministic_batch_ids_for_same_request():
    lots = [lot("A"), lot("B")]
    first = build_batch_plan("REQ-1", lots, [tube()], {})
    second = build_batch_plan("REQ-1", list(reversed(lots)), [tube()], {})
    other = build_batch_plan("REQ-2", lots, [tube()], {})
    assert first.batches[0].batch_id == second.batches[0].batch_id
    assert first.batches[0].batch_id != other.batches[0].batch_id


def test_duplicate_lot_and_tube_ids_are_rejected():
    with pytest.raises(ValueError):
        build_batch_plan("REQ", [lot("A"), lot("A")], [tube()], {})
    with pytest.raises(ValueError):
        build_batch_plan("REQ", [lot("A")], [tube(), tube()], {})


def test_empty_candidate_set_returns_empty_plan():
    plan = build_batch_plan("REQ", [], [tube()], {})
    assert plan.batches == () and plan.unassigned == ()


def test_two_underfilled_batches_on_one_tube_merge_via_salvage():
    # Each lot alone is below the minimum; greedy first opens one batch, then
    # a second short batch, and the salvage pass must dissolve one into the
    # other so a single launchable batch survives.
    lots = [
        lot("A", qty=30, open_at=0, close_at=100),
        lot("B", qty=30, open_at=0, close_at=100),
    ]
    plan = build_batch_plan("REQ", lots, [tube(minimum=50)], {})
    assert len(plan.batches) == 1
    assert plan.batches[0].total_quantity == 60
    assert plan.batches[0].lot_ids == ("A", "B")
    assert plan.unassigned == ()


def test_plan_exposes_every_lot_exactly_once():
    lots = [lot(f"L{i}", family="OX" if i % 2 else "NIT", qty=15 + i) for i in range(8)]
    plan: BatchPlan = build_batch_plan("REQ", lots, [tube(minimum=60)], {})
    accounted = plan.launched_lot_ids | {item.lot_id for item in plan.unassigned}
    assert accounted == {lot.lot_id for lot in lots}

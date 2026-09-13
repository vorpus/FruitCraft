"""Calibrate + tick the fly pipeline on a synthetic graph (GPU, no BW data)."""

import numpy as np
import pytest

from fruitcraft.fly.encoding import SensoryEncoder, pick_populations_synthetic
from fruitcraft.fly.decoding import ActionDecoder, ACTIONS, STAY
from fruitcraft.fly.brainpool import UnitBrainPool
from fruitcraft.fly.controller import build_observation, HivemindDirective
from conftest import requires_cuda, make_unit

pytestmark = requires_cuda


@pytest.fixture(scope="module")
def pipeline(synthetic_flies):
    encoder = SensoryEncoder(pick_populations_synthetic(
        synthetic_flies.graph.n_neurons, pop_size=8))
    decoder = ActionDecoder.calibrate(
        synthetic_flies, encoder, readout_size=12, stay_floor=0.15)
    return synthetic_flies, encoder, decoder


def test_calibration_readouts_disjoint_from_inputs(pipeline):
    flies, encoder, decoder = pipeline
    inputs = set(encoder.all_input_ids.tolist())
    for action in ACTIONS:
        assert len(decoder.readouts[action]) > 0
        assert not inputs & set(decoder.readouts[action].tolist())


def test_prototype_inputs_decode_to_their_action(pipeline):
    """Feeding each action's own prototype should usually win the argmax."""
    from fruitcraft.fly.decoding import PROTOTYPES, _prototype_observation

    flies, encoder, decoder = pipeline
    pool = UnitBrainPool(flies, encoder, decoder, decision_ms=60.0)
    hits = 0
    for action in ACTIONS:
        obs = _prototype_observation(PROTOTYPES[action])
        result = pool.tick({42: obs})
        hits += result[42] == action
    pool.sync_units([])  # release
    assert hits >= 4, f"only {hits}/5 prototypes decoded to their own action"


def test_zero_observation_stays(pipeline):
    """No drive on any channel -> no spikes -> STAY, deterministically.
    (A unit with weapon_ready=1 is NOT zero-drive; use a cooling-down one.)"""
    flies, encoder, decoder = pipeline
    pool = UnitBrainPool(flies, encoder, decoder, decision_ms=60.0)
    unit = make_unit(7, gw_cooldown=5)  # weapon_ready = 0
    result = pool.tick({7: build_observation(unit, [], None)})
    pool.sync_units([])
    assert result[7] == STAY


def test_pool_slot_lifecycle(pipeline):
    flies, encoder, decoder = pipeline
    pool = UnitBrainPool(flies, encoder, decoder)
    obs = build_observation(make_unit(0), [], HivemindDirective(0, 0))
    pool.tick({10: obs, 11: obs, 12: obs})
    assert pool.n_active == 3
    slots_before = dict(pool.slot_of)
    pool.tick({10: obs, 12: obs})  # unit 11 died
    assert pool.n_active == 2
    assert pool.slot_of[10] == slots_before[10]  # survivors keep their brains
    assert 11 not in pool.slot_of
    pool.sync_units([])
    assert pool.n_active == 0

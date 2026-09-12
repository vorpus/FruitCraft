import numpy as np

from fruitcraft.fly.encoding import (
    CHANNELS, CombatObservation, pick_populations_synthetic)
from fruitcraft.fly.controller import build_observation, HivemindDirective
from conftest import make_unit


def test_populations_disjoint_and_complete():
    pops = pick_populations_synthetic(400, pop_size=8)
    assert set(pops) == set(CHANNELS)
    all_ids = np.concatenate(list(pops.values()))
    assert len(all_ids) == len(np.unique(all_ids))


def test_channel_drives_direction_split():
    obs = CombatObservation(enemy_dx=0.8, enemy_dy=-0.5, enemy_near=0.6)
    d = obs.channel_drives()
    assert d["enemy_e"] == 0.8 and d["enemy_w"] == 0
    assert d["enemy_n"] == 0.5 and d["enemy_s"] == 0
    assert d["enemy_near"] == 0.6


def test_no_visible_enemy_gates_direction():
    obs = CombatObservation(enemy_dx=0.8, enemy_near=0.0)
    assert obs.channel_drives()["enemy_e"] == 0


def test_build_observation_geometry():
    unit = make_unit(1, x=1000, y=1000, hp=10, max_hp=40)
    enemy = make_unit(2, x=1256, y=1000, owner=1)
    directive = HivemindDirective(objective_x=1000, objective_y=488)  # due north
    obs = build_observation(unit, [enemy], directive)
    assert obs.enemy_dx == 0.5 and obs.enemy_dy == 0  # 256/512 east
    assert 0.4 < obs.enemy_near <= 0.5
    assert obs.low_health == 0.75
    assert obs.hive_dy == -1.0 and obs.hive_dx == 0  # north = -y
    assert obs.weapon_ready == 1.0


def test_build_observation_no_enemies():
    obs = build_observation(make_unit(1), [], None)
    assert obs.enemy_near == 0 and obs.enemy_dx == 0
    assert obs.hive_attack == 0

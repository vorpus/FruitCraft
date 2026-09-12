import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _has_cuda():
    try:
        import cupy

        cupy.cuda.runtime.getDeviceCount()
        return True
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not _has_cuda(), reason="no CUDA GPU")


@pytest.fixture(scope="session")
def synthetic_flies():
    """A small FlyBatch on a random synthetic graph — no MaleCNS data needed."""
    from fruitloop import ConnectomeGraph
    from fruitloop.simulator import FlyBatch

    rng = np.random.default_rng(7)
    n = 400
    edges = {}
    for _ in range(8000):
        edges[(int(rng.integers(n)), int(rng.integers(n)))] = int(rng.integers(50, 400))
    signs = rng.choice(np.array([1, 1, 1, -1], dtype=np.int8), size=n)
    graph = ConnectomeGraph.from_edges(
        n, [(s, d, c) for (s, d), c in edges.items()], signs=signs)
    return FlyBatch(32, graph=graph)


class FakeGame:
    """Minimal stand-in for BroodWarGame: records commands, serves fixed units."""

    def __init__(self, my_units=None, enemy_units=None):
        self.self_id = 0
        self.enemy_id = 1
        self._my = my_units or []
        self._enemy = enemy_units or []
        self.commands = []
        self.frame = 0

    def my_units(self):
        return self._my

    def enemy_units_visible(self):
        return self._enemy

    def enemy_units_all(self):
        return self._enemy

    def move(self, uid, x, y):
        self.commands.append(("move", uid, x, y))
        return True

    def attack_move(self, uid, x, y):
        self.commands.append(("attack_move", uid, x, y))
        return True

    def attack_unit(self, uid, tid):
        self.commands.append(("attack_unit", uid, tid))
        return True


def make_unit(uid, x=1000, y=1000, unit_type=0, hp=40, max_hp=40, owner=0,
              gw_cooldown=0, under_attack=False):
    return {
        "id": uid, "type": unit_type, "x": x, "y": y, "hp": hp, "max_hp": max_hp,
        "shields": 0, "max_shields": 0, "energy": 0, "owner": owner,
        "completed": True, "flying": False, "idle": True, "moving": False,
        "attacking": False, "under_attack": under_attack,
        "gw_cooldown": gw_cooldown, "aw_cooldown": 0,
        "vel_x": 0.0, "vel_y": 0.0, "order": 0,
    }


@pytest.fixture
def fake_game_factory():
    return FakeGame, make_unit

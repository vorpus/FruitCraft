"""Controller action -> engine command mapping, using a fake game and a stub pool."""

from fruitcraft.fly.controller import FlyCombatController, HivemindDirective
from conftest import FakeGame, make_unit


class StubPool:
    """Returns a scripted action for every observed unit."""

    def __init__(self, action):
        self.action = action
        self.seen = {}

    def tick(self, observations):
        self.seen = observations
        return {tag: self.action for tag in observations}


def _controller(action, my_units, enemy_units):
    game = FakeGame(my_units, enemy_units)
    controller = FlyCombatController(game, StubPool(action), controlled_types={0})
    return game, controller


def test_move_commands_use_bw_axes():
    game, c = _controller("north", [make_unit(1, x=500, y=500)], [])
    c.tick()
    assert game.commands == [("move", 1, 500, 500 - 64)]  # north is -y


def test_attack_targets_nearest_enemy():
    far, near = make_unit(8, x=900, y=500, owner=1), make_unit(9, x=600, y=500, owner=1)
    game, c = _controller("attack", [make_unit(1, x=500, y=500)], [far, near])
    c.tick()
    assert game.commands == [("attack_unit", 1, 9)]


def test_attack_without_visible_enemy_pushes_to_objective():
    game, c = _controller("attack", [make_unit(1, x=500, y=500)], [])
    c.set_directive(HivemindDirective(2000, 1500, attack=1.0))
    c.tick()
    assert game.commands == [("attack_move", 1, 2000, 1500)]


def test_only_controlled_completed_units_get_observations():
    marine = make_unit(1, unit_type=0)
    tank = make_unit(2, unit_type=5)
    incomplete = dict(make_unit(3, unit_type=0), completed=False)
    game = FakeGame([marine, tank, incomplete], [])
    pool = StubPool("stay")
    FlyCombatController(game, pool, controlled_types={0}).tick()
    assert set(pool.seen.keys()) == {1}


def test_directive_reaches_observations():
    game = FakeGame([make_unit(1, x=500, y=500)], [])
    pool = StubPool("stay")
    c = FlyCombatController(game, pool, controlled_types={0})
    c.set_directive(HivemindDirective(500 + 512, 500, attack=0.7))
    c.tick()
    obs = pool.seen[1]
    assert obs.hive_dx == 1.0 and obs.hive_dy == 0.0
    assert obs.hive_attack == 0.7

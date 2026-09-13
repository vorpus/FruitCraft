"""FlyCombatController: ties game state -> observations -> brains -> commands.

The hivemind directive is a per-team suggestion (an objective position and an
aggression mode). It only ever enters the flies as sensory input; each fly
remains free to do its own thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import CombatObservation
from .brainpool import UnitBrainPool

MOVE_STEP = 64          # pixels per movement command
DIRECTION_RANGE = 512.0  # pixels over which a direction signal saturates
ORDER_GUARD = 3          # engine order id for an idle/passive unit (empirical)


@dataclass
class HivemindDirective:
    """A suggestion broadcast to every fly, not an order."""
    objective_x: float
    objective_y: float
    attack: float = 1.0  # 0..1 aggression mode


def build_observation(unit: dict, visible_enemies: list[dict],
                      directive: HivemindDirective | None) -> CombatObservation:
    obs = CombatObservation()
    obs.low_health = 1.0 - unit["hp"] / max(unit["max_hp"], 1)
    obs.under_attack = 1.0 if unit["under_attack"] else 0.0
    obs.weapon_ready = 1.0 if unit["gw_cooldown"] == 0 else 0.0
    if visible_enemies:
        nearest = min(visible_enemies,
                      key=lambda e: (e["x"] - unit["x"]) ** 2 + (e["y"] - unit["y"]) ** 2)
        dx, dy = nearest["x"] - unit["x"], nearest["y"] - unit["y"]
        dist = float(np.hypot(dx, dy))
        obs.enemy_dx = float(np.clip(dx / DIRECTION_RANGE, -1, 1))
        obs.enemy_dy = float(np.clip(dy / DIRECTION_RANGE, -1, 1))
        obs.enemy_near = float(np.clip(1.0 - dist / DIRECTION_RANGE, 0.01, 1))
    if directive is not None:
        dx, dy = directive.objective_x - unit["x"], directive.objective_y - unit["y"]
        obs.hive_dx = float(np.clip(dx / DIRECTION_RANGE, -1, 1))
        obs.hive_dy = float(np.clip(dy / DIRECTION_RANGE, -1, 1))
        obs.hive_attack = directive.attack
    return obs


class FlyCombatController:
    """Issues engine commands for every controlled unit each decision tick."""

    def __init__(self, game, pool: UnitBrainPool, controlled_types: set[int]):
        self.game = game
        self.pool = pool
        self.controlled_types = controlled_types
        self.directive: HivemindDirective | None = None
        self._last_action: dict[int, str] = {}

    def set_directive(self, directive: HivemindDirective | None):
        self.directive = directive

    def tick(self) -> dict[int, str]:
        units = [u for u in self.game.my_units()
                 if u["type"] in self.controlled_types and u["completed"]]
        enemies = self.game.enemy_units_visible()
        observations = {
            u["id"]: build_observation(u, enemies, self.directive) for u in units}
        actions = self.pool.tick(observations)
        by_id = {u["id"]: u for u in units}
        for tag, action in actions.items():
            self._execute(by_id[tag], action, enemies)
        self._last_action = dict(actions)
        return actions

    def _execute(self, unit: dict, action: str, enemies: list[dict]):
        # Command hygiene: this engine has no auto-acquire, and re-issuing a
        # command resets the unit's attack sequence before its damage frame.
        # Issue only when the fly changed its mind or the unit has gone idle;
        # a persisting attack-move keeps the unit fighting on its own.
        unchanged = self._last_action.get(unit["id"]) == action
        is_idle = unit["idle"] or unit["order"] == ORDER_GUARD
        if unchanged and not is_idle:
            return
        x, y = unit["x"], unit["y"]
        if action == "north":
            self.game.move(unit["id"], x, y - MOVE_STEP)
        elif action == "south":
            self.game.move(unit["id"], x, y + MOVE_STEP)
        elif action == "east":
            self.game.move(unit["id"], x + MOVE_STEP, y)
        elif action == "west":
            self.game.move(unit["id"], x - MOVE_STEP, y)
        elif action == "attack":
            # attack-move (not attack-unit): the unit engages at weapon range
            # instead of chasing into melee, and keeps fighting when its
            # target dies
            if enemies:
                nearest = min(enemies, key=lambda e: (e["x"] - x) ** 2 + (e["y"] - y) ** 2)
                self.game.attack_move(unit["id"], nearest["x"], nearest["y"])
            elif self.directive is not None:
                self.game.attack_move(unit["id"], int(self.directive.objective_x),
                                      int(self.directive.objective_y))
        # stay: deliberately no command — a move-to-self would cancel any
        # ongoing attack (and nothing auto-acquires in this engine)

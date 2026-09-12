"""Micro-combat scenarios: fly-controlled army vs a scripted opponent army.

Base building is out of scope for now: OpenSnowstorm's computer AI does not yet
execute aiscript build orders, so scenarios spawn armies directly via the
engine's trigger-spawn path and script the opponent with periodic attack-move
pulses (commands can be issued as any unit's owner).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import unittypes


@dataclass
class MicroConfig:
    our_army: list[tuple[int, int]] = field(
        default_factory=lambda: [(unittypes.MARINE, 8)])
    enemy_army: list[tuple[int, int]] = field(
        default_factory=lambda: [(unittypes.ZERGLING, 10)])
    separation_px: int = 480      # spawn distance between the two armies
    spread_px: int = 96
    enemy_pulse_frames: int = 48  # how often scripted enemies re-attack-move
    max_frames: int = 24 * 90     # 90 game-seconds


class MicroScenario:
    """Spawns both armies mid-map and referees one battle episode."""

    def __init__(self, game, config: MicroConfig | None = None, seed=0):
        self.game = game
        self.config = config or MicroConfig()
        self.rng = np.random.default_rng(seed)
        self.our_ids: set[int] = set()
        self.enemy_ids: set[int] = set()
        self.start_frame = 0

    def reset(self):
        w, h = self.game.map_pixel_size()
        cx, cy = w // 2, h // 2
        half = self.config.separation_px // 2
        self.our_ids = self._spawn_army(
            self.game.self_id, self.config.our_army, cx - half, cy)
        self.enemy_ids = self._spawn_army(
            self.game.enemy_id, self.config.enemy_army, cx + half, cy)
        self.start_frame = self.game.frame
        return (cx + half, cy)  # a natural hivemind objective: the enemy camp

    def _spawn_army(self, player_id, composition, cx, cy) -> set[int]:
        ids = set()
        for unit_type, count in composition:
            for _ in range(count):
                x = cx + int(self.rng.integers(-self.config.spread_px, self.config.spread_px))
                y = cy + int(self.rng.integers(-self.config.spread_px, self.config.spread_px))
                uid = self.game.spawn(player_id, unit_type, x, y)
                if uid >= 0:
                    ids.add(uid)
        return ids

    def script_enemy(self):
        """Scripted opponent: periodically attack-move everyone at our army."""
        if self.game.frame % self.config.enemy_pulse_frames != 0:
            return
        ours = [u for u in self.game.my_units() if u["id"] in self.our_ids]
        if not ours:
            return
        tx = int(np.mean([u["x"] for u in ours]))
        ty = int(np.mean([u["y"] for u in ours]))
        for u in self.game.enemy_units_all():
            if u["id"] in self.enemy_ids:
                self.game.attack_move(u["id"], tx, ty)

    def status(self) -> dict:
        ours = [u for u in self.game.my_units() if u["id"] in self.our_ids]
        theirs = [u for u in self.game.enemy_units_all() if u["id"] in self.enemy_ids]
        elapsed = self.game.frame - self.start_frame
        done = (not ours or not theirs or elapsed >= self.config.max_frames)
        return {
            "ours_alive": len(ours),
            "enemies_alive": len(theirs),
            "our_hp": sum(u["hp"] for u in ours),
            "enemy_hp": sum(u["hp"] for u in theirs),
            "elapsed_frames": elapsed,
            "done": done,
            "won": bool(done and ours and not theirs),
        }

    def cleanup(self):
        for uid in self.our_ids | self.enemy_ids:
            self.game.kill(uid)
        self.our_ids.clear()
        self.enemy_ids.clear()

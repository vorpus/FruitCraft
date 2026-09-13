"""Record micro battles — positions, hp, and each fly brain's introspection
(channel drives, action logits, decoded action, issued command) — as compact
JSON for the web battle viewer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import unittypes
from .micro import MicroScenario, MicroConfig
from .fly.brainpool import UnitBrainPool
from .fly.controller import FlyCombatController, HivemindDirective
from .fly.encoding import CHANNELS
from .fly.decoding import ACTIONS

FRAME_SAMPLE = 2  # record unit state every N frames


def record_episode(game, flies, encoder, decoder, config: MicroConfig,
                   seed: int, label: str, decision_ms: float = 20.0,
                   decision_frames: int = 12) -> dict:
    scenario = MicroScenario(game, config, seed=seed)
    pool = UnitBrainPool(flies, encoder, decoder, decision_ms=decision_ms)
    ctrl = FlyCombatController(game, pool,
                               controlled_types={t for t, _ in config.our_army})
    objective = scenario.reset()
    ctrl.set_directive(HivemindDirective(*objective, attack=1.0))
    start_frame = game.frame

    frames, decisions, events = [], [], []
    known = {}

    def capture_units():
        rows = []
        for u in game.my_units() + game.enemy_units_all():
            if u["id"] not in scenario.our_ids | scenario.enemy_ids:
                continue
            known[u["id"]] = u
            rows.append([u["id"], u["owner"], u["type"], u["x"], u["y"], u["hp"]])
        return rows

    alive_before = set(scenario.our_ids | scenario.enemy_ids)
    while True:
        scenario.script_enemy()
        rel_frame = game.frame - start_frame
        if game.frame % decision_frames == 0:
            actions = ctrl.tick()
            per_unit = {}
            for tag, action in actions.items():
                drives = ctrl.last_observations[tag].channel_drives()
                per_unit[str(tag)] = {
                    "a": action,
                    "l": [round(float(v), 3) for v in pool.last_logits[tag]],
                    "d": [round(float(drives[c]), 2) for c in CHANNELS],
                    "c": list(ctrl.last_commands.get(tag) or ["stay"]),
                }
            decisions.append({"f": rel_frame, "u": per_unit})
        if rel_frame % FRAME_SAMPLE == 0:
            frames.append({"f": rel_frame, "u": capture_units()})
        game.step(1)
        now_alive = {u["id"] for u in game.my_units() + game.enemy_units_all()}
        for dead in sorted(alive_before - now_alive):
            if dead in scenario.our_ids | scenario.enemy_ids:
                events.append({"f": game.frame - start_frame, "t": "death", "id": dead,
                               "owner": known.get(dead, {}).get("owner", -1)})
        alive_before = now_alive
        status = scenario.status()
        if status["done"] or game.result is not None:
            break
    pool.sync_units([])
    scenario.cleanup()
    game.step(24)

    return {
        "label": label,
        "seed": seed,
        "objective": [int(objective[0]), int(objective[1])],
        "our_ids": sorted(int(i) for i in scenario.our_ids),
        "enemy_ids": sorted(int(i) for i in scenario.enemy_ids),
        "unit_types": {str(i): known[i]["type"] for i in known},
        "channels": CHANNELS,
        "actions": ACTIONS,
        "stay_floor": decoder.stay_floor,
        "decision_frames": decision_frames,
        "frames": frames,
        "decisions": decisions,
        "events": events,
        "result": {k: int(v) for k, v in status.items()},
    }


def type_name(type_id: int) -> str:
    return unittypes.NAMES.get(type_id, f"type{type_id}")

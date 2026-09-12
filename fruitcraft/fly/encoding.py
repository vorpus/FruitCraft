"""Sensory encoding: per-unit combat observations -> fly neuron stimulation.

Each observation channel owns a disjoint population of real MaleCNS sensory
neurons. A channel's scalar value in [0, 1] drives its whole population with
value * amplitude. The hivemind directive is deliberately encoded the same way
as any other sense — it is a suggestion the fly integrates, not an override.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# channel -> drive in [0, 1]
CHANNELS = [
    "enemy_e", "enemy_w", "enemy_n", "enemy_s",   # direction to nearest visible enemy
    "enemy_near",                                  # proximity of that enemy
    "low_health", "under_attack", "weapon_ready",  # own state
    "hive_e", "hive_w", "hive_n", "hive_s",        # hivemind: direction to objective
    "hive_attack",                                 # hivemind: aggression mode
]


@dataclass
class CombatObservation:
    """Normalized per-unit view. BW pixel coords: +x east, +y south."""
    enemy_dx: float = 0.0     # [-1, 1], unit -> nearest visible enemy (0 if none)
    enemy_dy: float = 0.0
    enemy_near: float = 0.0   # [0, 1], 0 = no visible enemy
    low_health: float = 0.0   # 1 - hp_fraction
    under_attack: float = 0.0
    weapon_ready: float = 0.0
    hive_dx: float = 0.0      # [-1, 1], unit -> hivemind objective
    hive_dy: float = 0.0
    hive_attack: float = 0.0

    def channel_drives(self) -> dict[str, float]:
        return {
            "enemy_e": max(self.enemy_dx, 0.0) * (self.enemy_near > 0),
            "enemy_w": max(-self.enemy_dx, 0.0) * (self.enemy_near > 0),
            "enemy_s": max(self.enemy_dy, 0.0) * (self.enemy_near > 0),
            "enemy_n": max(-self.enemy_dy, 0.0) * (self.enemy_near > 0),
            "enemy_near": self.enemy_near,
            "low_health": self.low_health,
            "under_attack": self.under_attack,
            "weapon_ready": self.weapon_ready,
            "hive_e": max(self.hive_dx, 0.0),
            "hive_w": max(-self.hive_dx, 0.0),
            "hive_s": max(self.hive_dy, 0.0),
            "hive_n": max(-self.hive_dy, 0.0),
            "hive_attack": self.hive_attack,
        }


def pick_populations_malecns(neurons, pop_size=120, seed=0) -> dict[str, np.ndarray]:
    """Disjoint sensory populations from real MaleCNS annotations."""
    rng = np.random.default_rng(seed)
    candidates = neurons.query(
        superclass=["ol_sensory", "visual_projection", "cb_sensory"])
    chosen = rng.choice(candidates, size=pop_size * len(CHANNELS), replace=False)
    return {c: np.sort(chosen[i * pop_size:(i + 1) * pop_size])
            for i, c in enumerate(CHANNELS)}


def pick_populations_synthetic(n_neurons, pop_size=8, seed=0) -> dict[str, np.ndarray]:
    """Disjoint populations for tiny synthetic test graphs."""
    rng = np.random.default_rng(seed)
    need = pop_size * len(CHANNELS)
    if need > n_neurons:
        raise ValueError("graph too small for requested populations")
    chosen = rng.choice(n_neurons, size=need, replace=False).astype(np.uint32)
    return {c: np.sort(chosen[i * pop_size:(i + 1) * pop_size])
            for i, c in enumerate(CHANNELS)}


class SensoryEncoder:
    def __init__(self, populations: dict[str, np.ndarray], amplitude=1.5):
        assert set(populations) == set(CHANNELS)
        self.populations = populations
        self.amplitude = amplitude
        self.all_input_ids = np.unique(np.concatenate(list(populations.values())))

    def apply(self, flies, slot_ids: np.ndarray, observations: list[CombatObservation]):
        """Overwrite stimulation for the given fly slots from their observations."""
        flies.clear_stimulation(slot_ids)
        drives = np.array(
            [[obs.channel_drives()[c] for obs in observations] for c in CHANNELS],
            dtype=np.float32) * self.amplitude
        for ci, channel in enumerate(CHANNELS):
            values = drives[ci]
            if not values.any():
                continue
            flies.stimulate_population(slot_ids, self.populations[channel],
                                       values[:, None])

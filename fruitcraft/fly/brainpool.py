"""UnitBrainPool: one independent fly brain per StarCraft unit, batched.

The pool owns a FlyBatch, the unit-tag <-> fly-slot mapping, and the
encode -> simulate -> decode tick. Everything is batched: never one fly at a
time (FruitLoop PRD §26).
"""

from __future__ import annotations

import numpy as np

from .encoding import CombatObservation, SensoryEncoder
from .decoding import ActionDecoder


class UnitBrainPool:
    def __init__(self, flies, encoder: SensoryEncoder, decoder: ActionDecoder,
                 decision_ms: float = 20.0):
        self.flies = flies
        self.encoder = encoder
        self.decoder = decoder
        self.decision_ms = decision_ms
        self.slot_of: dict[int, int] = {}  # unit tag -> fly slot

    def sync_units(self, alive_tags) -> None:
        """Allocate brains for new units, release brains of dead ones."""
        alive = set(alive_tags)
        for tag in [t for t in self.slot_of if t not in alive]:
            self.flies.release(self.slot_of.pop(tag))
        for tag in alive:
            if tag not in self.slot_of:
                self.slot_of[tag] = self.flies.allocate()

    def tick(self, observations: dict[int, CombatObservation]) -> dict[int, str]:
        """One batched decision for every observed unit: tag -> action name."""
        self.sync_units(observations.keys())
        if not observations:
            return {}
        tags = list(observations.keys())
        slots = np.array([self.slot_of[t] for t in tags])
        self.encoder.apply(self.flies, slots, [observations[t] for t in tags])
        self.flies.reset_spike_counts(slots)
        self.flies.advance_ms(self.decision_ms)
        actions = self.decoder.decode(self.flies, slots)
        return dict(zip(tags, actions))

    @property
    def n_active(self) -> int:
        return len(self.slot_of)

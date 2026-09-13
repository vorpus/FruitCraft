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
    """reset_each_tick: both the raw MaleCNS graph (at lif_v1 gain) and dense
    synthetic graphs are excitatory-dominant enough that strong stimulation
    latches them into a saturated attractor that never decays. Resetting each
    fly at the start of its decision makes every decision a clean transient
    matching the calibration condition. Set False to keep cross-tick neural
    state (the long-term research mode; needs a sub-critical gain to be
    informative rather than saturated)."""

    def __init__(self, flies, encoder: SensoryEncoder, decoder: ActionDecoder,
                 decision_ms: float = 20.0, reset_each_tick: bool = True):
        self.flies = flies
        self.encoder = encoder
        self.decoder = decoder
        self.decision_ms = decision_ms
        self.reset_each_tick = reset_each_tick
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
        if self.reset_each_tick:
            self.flies.reset(slots)  # must precede apply: reset clears i_ext too
        self.encoder.apply(self.flies, slots, [observations[t] for t in tags])
        self.flies.reset_spike_counts(slots)
        self.flies.advance_ms(self.decision_ms)
        actions = self.decoder.decode(self.flies, slots, window_ms=self.decision_ms)
        return dict(zip(tags, actions))

    @property
    def n_active(self) -> int:
        return len(self.slot_of)

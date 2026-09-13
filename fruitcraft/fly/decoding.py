"""Action decoding: spike counts in calibrated readout populations -> actions.

Calibration presents each action's prototype sensory pattern to one fly and
selects downstream neurons that respond discriminatively to it (excluding the
input populations themselves, so signals must pass through the connectome).
At run time the action with the strongest readout wins; if nothing clears the
floor the unit STAYs. This is an engineered mapping (PRD: 'engineered first,
later learned/evolved').
"""

from __future__ import annotations

import numpy as np

from .encoding import CHANNELS, CombatObservation, SensoryEncoder

ACTIONS = ["north", "south", "east", "west", "attack"]
STAY = "stay"

PROTOTYPES: dict[str, dict[str, float]] = {
    "north": {"hive_n": 1.0, "enemy_n": 0.5},
    "south": {"hive_s": 1.0, "enemy_s": 0.5},
    "east": {"hive_e": 1.0, "enemy_e": 0.5},
    "west": {"hive_w": 1.0, "enemy_w": 0.5},
    "attack": {"enemy_near": 1.0, "weapon_ready": 1.0, "hive_attack": 1.0},
}


class ActionDecoder:
    """Readout populations differ in excitability, so raw spike counts are not
    comparable across actions. Each action's logit is therefore normalized by
    that readout's own response to its prototype at calibration time:
    logit ~ 1.0 means 'responding as strongly as during calibration'."""

    def __init__(self, readouts: dict[str, np.ndarray],
                 calib_rate: dict[str, float], stay_floor: float = 0.15):
        self.readouts = readouts
        self.calib_rate = calib_rate  # calibration spikes/neuron/ms per action
        self.stay_floor = stay_floor  # STAY below this fraction of calib response
        self._flat = np.concatenate([readouts[a] for a in ACTIONS])
        self._sizes = [len(readouts[a]) for a in ACTIONS]
        self._rates = np.array([calib_rate[a] for a in ACTIONS])

    @classmethod
    def calibrate(cls, flies, encoder: SensoryEncoder, readout_size=150,
                  probe_ms=100.0, stay_floor: float = 0.15) -> "ActionDecoder":
        """Probe each action prototype on fly slot 0 and pick discriminative neurons."""
        responses = {}
        for action in ACTIONS:
            flies.reset()
            drives = PROTOTYPES[action]
            obs = _prototype_observation(drives)
            encoder.apply(flies, np.array([0]), [obs])
            flies.advance_ms(probe_ms)
            responses[action] = flies.read_spike_counts(batch_ids=[0])[0].astype(np.int64)
        flies.reset()

        exclude = np.zeros(len(next(iter(responses.values()))), dtype=bool)
        exclude[encoder.all_input_ids] = True
        readouts, calib_rate = {}, {}
        for action in ACTIONS:
            others = np.max([responses[a] for a in ACTIONS if a != action], axis=0)
            score = responses[action] - others
            score[exclude] = np.iinfo(np.int64).min
            top = np.argsort(score)[-readout_size:][::-1]
            top = top[score[top] > 0]
            if len(top) == 0:
                raise RuntimeError(f"calibration found no discriminative neurons for {action!r}")
            readouts[action] = np.sort(top).astype(np.uint32)
            calib_rate[action] = float(responses[action][readouts[action]].mean()) / probe_ms
        return cls(readouts, calib_rate, stay_floor=stay_floor)

    def logits_for(self, flies, slot_ids: np.ndarray, window_ms: float) -> np.ndarray:
        """[n_slots, n_actions] normalized readout activations (1.0 ~ calibration level)."""
        counts = flies.read_spike_counts(neuron_ids=self._flat, batch_ids=slot_ids)
        rows = []
        for row in counts:
            means, offset = [], 0
            for size in self._sizes:
                means.append(row[offset:offset + size].mean())
                offset += size
            rows.append(np.asarray(means) / (self._rates * window_ms))
        return np.asarray(rows)

    def decode_from_logits(self, logits: np.ndarray) -> list[str]:
        actions = []
        for row in logits:
            best = int(np.argmax(row))
            actions.append(ACTIONS[best] if row[best] >= self.stay_floor else STAY)
        return actions

    def decode(self, flies, slot_ids: np.ndarray, window_ms: float) -> list[str]:
        """One action per fly slot from spike counts accumulated over window_ms."""
        return self.decode_from_logits(self.logits_for(flies, slot_ids, window_ms))


def _prototype_observation(drives: dict[str, float]) -> CombatObservation:
    """Build an observation whose channel_drives() reproduce the prototype."""
    obs = CombatObservation()
    obs.enemy_near = drives.get("enemy_near", 0.0)
    obs.low_health = drives.get("low_health", 0.0)
    obs.under_attack = drives.get("under_attack", 0.0)
    obs.weapon_ready = drives.get("weapon_ready", 0.0)
    obs.hive_attack = drives.get("hive_attack", 0.0)
    obs.hive_dx = drives.get("hive_e", 0.0) - drives.get("hive_w", 0.0)
    obs.hive_dy = drives.get("hive_s", 0.0) - drives.get("hive_n", 0.0)
    enemy_dx = drives.get("enemy_e", 0.0) - drives.get("enemy_w", 0.0)
    enemy_dy = drives.get("enemy_s", 0.0) - drives.get("enemy_n", 0.0)
    if enemy_dx or enemy_dy:
        obs.enemy_dx, obs.enemy_dy = enemy_dx, enemy_dy
        # direction channels are gated on a visible enemy; keep the gate open
        # without polluting the prototype with a strong enemy_near drive
        obs.enemy_near = max(obs.enemy_near, 0.01)
    return obs

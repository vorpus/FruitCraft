"""Live fly-brain telemetry window, shown beside the engine's game window.

For one followed marine ("the hero"), every decision tick shows:
  - what it senses: an egocentric compass (enemy direction scaled by proximity,
    hivemind pull) plus its state senses (health, under attack, weapon)
  - its inputs: the 13 population-coded channel drives
  - its brain modes: the 5 evolved action readouts (+ implicit STAY), live
    activations normalized to calibration level, against the stay floor
  - the outcome: decoded action and the command issued to the game
"""

from __future__ import annotations

import numpy as np

from .fly.encoding import CHANNELS
from .fly.decoding import ACTIONS, STAY

GROUND = "#101418"
PANEL = "#1a2027"
INK = "#e8ecef"
INK2 = "#8b98a5"
OURS = "#4aa3ff"
THEIRS = "#e06a3c"
SELECT = "#ffd166"
WIN = "#35c07e"
LOSS = "#e05252"

CH_SHORT = ["enemy E", "enemy W", "enemy N", "enemy S", "enemy near", "low health",
            "under attack", "weapon ready", "hive E", "hive W", "hive N", "hive S",
            "hive attack"]


class BrainView:
    def __init__(self, stay_floor: float, position=(820, 40)):
        import matplotlib

        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.stay_floor = stay_floor
        plt.ion()
        self.fig = plt.figure(figsize=(5.6, 6.0), facecolor=GROUND)
        self.fig.canvas.manager.set_window_title("Fly Brain — live")
        try:
            self.fig.canvas.manager.window.move(*position)
        except Exception:
            pass
        gs = self.fig.add_gridspec(3, 1, height_ratios=[1.15, 1.25, 1.0],
                                   hspace=0.55, left=0.30, right=0.95, top=0.90, bottom=0.06)
        self.ax_eye = self.fig.add_subplot(gs[0])
        self.ax_in = self.fig.add_subplot(gs[1])
        self.ax_out = self.fig.add_subplot(gs[2])
        self.title = self.fig.suptitle("", color=INK, fontsize=13, fontweight="bold")
        self._build_eye()
        self._build_bars()
        self.fig.canvas.draw()
        plt.show(block=False)

    # ------------------------------------------------------------ construction
    def _build_eye(self):
        ax = self.ax_eye
        ax.set_facecolor(PANEL)
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_color("#2c3641")
        ax.set_title("what it senses (egocentric)", color=INK2, fontsize=9, pad=6)
        for r in (0.5, 1.0):
            ax.add_patch(self.plt.Circle((0, 0), r, fill=False, color="#2c3641", lw=0.8))
        for label, (x, y) in {"N": (0, 1.12), "S": (0, -1.16), "E": (1.12, 0), "W": (-1.16, 0)}.items():
            ax.text(x, y, label, color=INK2, fontsize=8, ha="center", va="center")
        ax.plot(0, 0, "o", color=OURS, ms=10, zorder=5)
        self.q_enemy = ax.annotate("", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=THEIRS, lw=2.2), zorder=4)
        self.q_hive = ax.annotate("", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=SELECT, lw=1.8, linestyle="--"), zorder=3)
        self.eye_legend = ax.text(-1.2, -1.45, "→ enemy (length = proximity)    ⇢ hivemind pull",
                                  color=INK2, fontsize=7.5, transform=ax.transData)
        self.chips = {}
        for i, name in enumerate(["low health", "under attack", "weapon ready"]):
            self.chips[name] = ax.text(1.45, 0.85 - i * 0.38, name, color=INK2, fontsize=8,
                                       ha="left", va="center",
                                       bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec="#2c3641"))

    def _build_bars(self):
        ax = self.ax_in
        ax.set_facecolor(PANEL)
        y = np.arange(len(CHANNELS))[::-1]
        colors = [SELECT if c.startswith("hive") else OURS for c in CHANNELS]
        self.in_bars = ax.barh(y, [0] * len(CHANNELS), color=colors, height=0.62)
        ax.set_yticks(y, CH_SHORT, fontsize=8, color=INK2)
        ax.set_xlim(0, 1.0); ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(colors=INK2, labelsize=7)
        for s in ax.spines.values(): s.set_color("#2c3641")
        ax.set_title("inputs — fraction of each sensory population stimulated",
                     color=INK2, fontsize=9, pad=6)

        ax = self.ax_out
        ax.set_facecolor(PANEL)
        names = ACTIONS + [STAY]
        y = np.arange(len(names))[::-1]
        self.out_bars = ax.barh(y, [0] * len(names), color=["#5c6873"] * len(names), height=0.6)
        ax.set_yticks(y, names, fontsize=9, color=INK2)
        ax.set_xlim(0, 1.6); ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.tick_params(colors=INK2, labelsize=7)
        for s in ax.spines.values(): s.set_color("#2c3641")
        ax.axvline(self.stay_floor, color=LOSS, lw=1.2, ls="--")
        ax.text(self.stay_floor, len(names) - 0.25, " stay floor", color=LOSS, fontsize=7, va="bottom")
        ax.set_title("brain modes — action readouts (1.0 = calibration level)",
                     color=INK2, fontsize=9, pad=6)
        self.cmd_text = ax.text(0.0, -1.45, "", color=INK, fontsize=9, family="monospace")

    # ----------------------------------------------------------------- update
    def update(self, hero_id, obs, logits, action, command):
        self.title.set_text(f"marine #{hero_id} — fly brain (166,700 neurons)")
        # compass: BW +y is south, so flip for display
        e_len = obs.enemy_near
        norm = float(np.hypot(obs.enemy_dx, obs.enemy_dy)) or 1.0
        self.q_enemy.xy = ((obs.enemy_dx / norm) * e_len, (-obs.enemy_dy / norm) * e_len) if e_len else (0, 0)
        hnorm = float(np.hypot(obs.hive_dx, obs.hive_dy)) or 1.0
        hlen = min(1.0, hnorm)
        self.q_hive.xy = ((obs.hive_dx / hnorm) * hlen, (-obs.hive_dy / hnorm) * hlen)
        for name, key in [("low health", obs.low_health > 0.4),
                          ("under attack", obs.under_attack > 0),
                          ("weapon ready", obs.weapon_ready > 0)]:
            chip = self.chips[name]
            chip.set_color(INK if key else "#3a4550")
            chip.get_bbox_patch().set_edgecolor(SELECT if key else "#2c3641")
        drives = obs.channel_drives()
        for bar, c in zip(self.in_bars, CHANNELS):
            bar.set_width(drives[c])
        values = list(logits) + [0.0]
        chosen = action
        for bar, name, v in zip(self.out_bars, ACTIONS + [STAY], values):
            bar.set_width(v if name != STAY else (self.stay_floor * 0.9 if chosen == STAY else 0))
            bar.set_color(WIN if name == chosen else "#5c6873")
        if command is None:
            cmd = "no command (stay)"
        elif command[0] == "continue":
            cmd = "continue current order"
        else:
            cmd = f"{command[0].replace('_', '-')} → ({command[1]}, {command[2]})"
        self.cmd_text.set_text(f"decoded: {chosen.upper():7s}  |  {cmd}")
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def alive(self) -> bool:
        return bool(self.plt.fignum_exists(self.fig.number))

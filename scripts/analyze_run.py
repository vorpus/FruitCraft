#!/usr/bin/env python3
"""Plot an evolution run's fitness history: best and population mean per generation."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES_BEST = "#2a78d6"   # slot 1 blue
SERIES_MEAN = "#eb6834"   # slot 2 orange


def main():
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "evolution_runs/run01")
    history = json.loads((run_dir / "history.json").read_text())
    gens = [h["gen"] for h in history]
    best = [h["best"] for h in history]
    mean = [h["mean"] for h in history]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot(gens, best, color=SERIES_BEST, lw=2, marker="o", ms=4, label="best genome")
    ax.plot(gens, mean, color=SERIES_MEAN, lw=2, marker="o", ms=4, label="population mean")
    ax.axhline(4.0, color=GRID, lw=1, ls="--")
    ax.text(gens[-1], 4.02, "max (win, all kills, all survive)",
            ha="right", va="bottom", fontsize=8, color=TEXT_2)
    ax.text(gens[-1], best[-1], f"  {best[-1]:.2f}", color=TEXT, fontsize=9, va="center")
    ax.text(gens[-1], mean[-1], f"  {mean[-1]:.2f}", color=TEXT, fontsize=9, va="center")
    ax.set_xlabel("generation", color=TEXT_2)
    ax.set_ylabel("battle fitness", color=TEXT_2)
    ax.set_title("Evolving the fly↔game interface (8 marines vs 10 zerglings)",
                 color=TEXT, fontsize=11)
    ax.grid(True, color=GRID, lw=0.6)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.legend(frameon=False, labelcolor=TEXT, fontsize=9, loc="lower right")
    fig.tight_layout()
    out = run_dir / "fitness.png"
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

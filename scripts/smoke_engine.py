#!/usr/bin/env python3
"""Phase 0 smoke test: boot Brood War headless, spawn a marine, move it A->B."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitcraft import unittypes
from fruitcraft.engine import BroodWarGame

BWDATA = Path(__file__).resolve().parents[1] / "bwdata"


def main():
    map_path = sys.argv[1] if len(sys.argv) > 1 else "maps/(2)Benzene.scx"
    print(f"booting {map_path} headless...")
    game = BroodWarGame(BWDATA, map_path, seed=1234)
    print(f"match started: frame={game.frame} self={game.self_id} enemy={game.enemy_id}")
    print(f"map: {game.map_pixel_size()} px")

    game.step(24)
    mine = game.my_units()
    print(f"my starting units ({len(mine)}):")
    for u in mine:
        print(f"  id={u['id']} type={u['type']} at ({u['x']},{u['y']}) hp={u['hp']}")

    w, h = game.map_pixel_size()
    cx, cy = w // 2, h // 2
    uid = game.spawn(game.self_id, unittypes.MARINE, cx, cy)
    game.step(1)
    marine = next(u for u in game.my_units() if u["id"] == uid)
    start = (marine["x"], marine["y"])
    print(f"spawned marine id={uid} at {start}")

    target = (cx + 200, cy)
    assert game.move(uid, *target)
    game.step(120)  # 5 game-seconds
    marine = next(u for u in game.my_units() if u["id"] == uid)
    end = (marine["x"], marine["y"])
    moved = abs(end[0] - start[0])
    print(f"after move command: at {end} (moved {moved}px east)")
    assert moved > 100, "marine did not move under program control"
    print(f"frame={game.frame}, result={game.result}")
    print("PHASE 0 OK: worker^W marine moves from A to B under program control")


if __name__ == "__main__":
    main()

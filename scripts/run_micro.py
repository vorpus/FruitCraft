#!/usr/bin/env python3
"""End-to-end micro battle: fly-brain marines vs scripted zerglings.

Requires:
  - Brood War data files (STARDAT.MPQ, BROODAT.MPQ, patch_rt.mpq) in --bw-data
  - a 2-player melee map (path relative to --bw-data, e.g. maps/(2)Astral Balance.scm)
  - the FruitLoop processed MaleCNS graph (built in ../FruitLoop-CUDA)

Usage:
  python scripts/run_micro.py --bw-data ~/broodwar --map "maps/(2)Astral Balance.scm"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitcraft import unittypes
from fruitcraft.engine import BroodWarGame
from fruitcraft.micro import MicroScenario, MicroConfig
from fruitcraft.fly.encoding import SensoryEncoder, pick_populations_malecns
from fruitcraft.fly.decoding import ActionDecoder
from fruitcraft.fly.brainpool import UnitBrainPool
from fruitcraft.fly.controller import FlyCombatController, HivemindDirective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw-data", required=True, help="dir with Brood War MPQs")
    parser.add_argument("--map", required=True, help="melee map path relative to bw-data")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--decision-frames", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    print("loading fly brains (MaleCNS graph)...")
    from fruitloop import ConnectomeGraph
    from fruitloop.simulator import FlyBatch

    graph = ConnectomeGraph.load()
    flies = FlyBatch(args.batch, graph=graph)
    encoder = SensoryEncoder(pick_populations_malecns(graph.neurons))
    print("calibrating action readouts...")
    decoder = ActionDecoder.calibrate(flies, encoder)
    pool = UnitBrainPool(flies, encoder, decoder)

    print("starting Brood War...")
    game = BroodWarGame(args.bw_data, args.map, seed=1234)
    controller = FlyCombatController(game, pool, controlled_types={unittypes.MARINE})
    scenario = MicroScenario(game, MicroConfig())

    wins = 0
    for episode in range(args.episodes):
        objective = scenario.reset()
        controller.set_directive(HivemindDirective(*objective, attack=1.0))
        t0 = time.perf_counter()
        ticks = 0
        while True:
            scenario.script_enemy()
            if game.frame % args.decision_frames == 0:
                controller.tick()
                ticks += 1
            game.step(1)
            status = scenario.status()
            if status["done"] or game.result is not None:
                break
        wall = time.perf_counter() - t0
        wins += status["won"]
        print(f"episode {episode}: {'WON' if status['won'] else 'lost'}  "
              f"ours {status['ours_alive']} vs enemies {status['enemies_alive']}  "
              f"({status['elapsed_frames']} frames, {ticks} decisions, {wall:.1f}s wall)")
        scenario.cleanup()
        game.step(24)  # let corpses clear

    print(f"\n{wins}/{args.episodes} episodes won by the fly swarm")


if __name__ == "__main__":
    main()

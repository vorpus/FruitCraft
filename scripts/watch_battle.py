#!/usr/bin/env python3
"""Watch fly-brain battles live in the engine's own SDL window (real BW graphics).

Requires an engine build with the renderer (scripts/build_engine.sh does this
automatically when pysdl2-dll is installed) and a display.

Usage:
  python scripts/watch_battle.py [--genome genomes/champion_run01.npz]
      [--episodes 3] [--speed 1.0]
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # engine mutes anyway

from fruitcraft.engine import BroodWarGame
from fruitcraft.micro import MicroScenario, MicroConfig
from fruitcraft.evolve import BattleEvaluator, Genome
from fruitcraft.fly.brainpool import UnitBrainPool
from fruitcraft.fly.controller import FlyCombatController, HivemindDirective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw-data", default="bwdata")
    parser.add_argument("--map", default="maps/(2)Benzene.scx")
    parser.add_argument("--genome", default="genomes/champion_run01.npz")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0, help="game speed multiplier")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    genome_path = Path(args.genome).resolve()
    print("loading fly brains...")
    from fruitloop import ConnectomeGraph
    from fruitloop.simulator import FlyBatch

    graph = ConnectomeGraph.load()
    flies = FlyBatch(16, graph=graph)

    print("starting Brood War with renderer...")
    game = BroodWarGame(Path(args.bw_data).resolve(), args.map, seed=4242, gui=True)
    config = MicroConfig()
    evaluator = BattleEvaluator(game, flies, config)
    genome = Genome.load(genome_path)
    encoder, decoder = evaluator.build(genome)
    rng = np.random.default_rng(args.seed)

    for episode in range(args.episodes):
        scenario = MicroScenario(game, config, seed=int(rng.integers(1 << 30)))
        pool = UnitBrainPool(flies, encoder, decoder, decision_ms=genome.decision_ms)
        ctrl = FlyCombatController(game, pool, controlled_types={t for t, _ in config.our_army})
        objective = scenario.reset()
        ctrl.set_directive(HivemindDirective(*objective, attack=1.0))
        print(f"episode {episode}: watch the window (marines = fly brains)")
        t0 = time.perf_counter()
        frame0 = game.frame
        while True:
            scenario.script_enemy()
            if game.frame % 12 == 0:
                ctrl.tick()
            game.step(1)
            # pace to real time x speed
            target = (game.frame - frame0) / (24 * args.speed)
            behind = target - (time.perf_counter() - t0)
            if behind > 0:
                time.sleep(behind)
            status = scenario.status()
            if status["done"] or game.result is not None:
                break
        print(f"  -> {'WON' if status['won'] else 'lost'} "
              f"({status['ours_alive']} marines vs {status['enemies_alive']} zerglings left)")
        pool.sync_units([])
        scenario.cleanup()
        game.step(24)
        time.sleep(1.0)


if __name__ == "__main__":
    main()

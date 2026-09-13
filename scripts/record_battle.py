#!/bin/sh
'''exec' "$(dirname "$0")/../.venv/bin/python" "$0" "$@" #'''
"""Record battles (champion and/or default interface) to JSON for the viewer.

Usage:
  python scripts/record_battle.py --genome genomes/champion_run01.npz \
      --episodes 3 --default-episodes 2 --out battles.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitcraft.engine import BroodWarGame
from fruitcraft.micro import MicroConfig
from fruitcraft.evolve import BattleEvaluator, Genome
from fruitcraft.record import record_episode
from fruitcraft.fly.encoding import SensoryEncoder, pick_populations_malecns
from fruitcraft.fly.decoding import ActionDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw-data", default="bwdata")
    parser.add_argument("--map", default="maps/(2)Benzene.scx")
    parser.add_argument("--genome", default="genomes/champion_run01.npz")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--default-episodes", type=int, default=2)
    parser.add_argument("--out", default="battles.json")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    out_path = Path(args.out).resolve()
    genome_path = Path(args.genome).resolve()

    print("loading fly brains...")
    from fruitloop import ConnectomeGraph
    from fruitloop.simulator import FlyBatch

    graph = ConnectomeGraph.load()
    flies = FlyBatch(16, graph=graph)

    print("starting Brood War...")
    game = BroodWarGame(Path(args.bw_data).resolve(), args.map, seed=4242)
    config = MicroConfig()
    evaluator = BattleEvaluator(game, flies, config)
    rng = np.random.default_rng(args.seed)

    episodes = []
    genome = Genome.load(genome_path)
    encoder, decoder = evaluator.build(genome)
    for i in range(args.episodes):
        seed = int(rng.integers(1 << 30))
        t0 = time.perf_counter()
        ep = record_episode(game, flies, encoder, decoder, config, seed,
                            label=f"champion #{i + 1}",
                            decision_ms=genome.decision_ms)
        print(f"champion ep{i}: won={ep['result']['won']} "
              f"({time.perf_counter() - t0:.1f}s, {len(ep['frames'])} samples)")
        episodes.append(ep)

    if args.default_episodes:
        encoder = SensoryEncoder(pick_populations_malecns(graph.neurons))
        decoder = ActionDecoder.calibrate(flies, encoder)
        for i in range(args.default_episodes):
            seed = int(rng.integers(1 << 30))
            ep = record_episode(game, flies, encoder, decoder, config, seed,
                                label=f"pre-evolution #{i + 1}")
            print(f"default ep{i}: won={ep['result']['won']}")
            episodes.append(ep)

    out_path.write_text(json.dumps({"episodes": episodes}, separators=(",", ":")))
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

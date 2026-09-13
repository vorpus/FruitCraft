#!/usr/bin/env python3
"""Evolve the fly<->game interface against micro-battle outcomes.

Usage:
  python scripts/evolve_micro.py --bw-data bwdata --map "maps/(2)Benzene.scx" \
      --generations 15 --pop-size 16 --episodes 4 [--seed-genome path.npz]

Then evaluate a champion over many episodes:
  python scripts/evolve_micro.py ... --eval evolution_runs/<run>/best_final.npz
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fruitcraft.engine import BroodWarGame
from fruitcraft.micro import MicroConfig
from fruitcraft.evolve import BattleEvaluator, Genome, evolve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw-data", default="bwdata")
    parser.add_argument("--map", default="maps/(2)Benzene.scx")
    parser.add_argument("--generations", type=int, default=15)
    parser.add_argument("--pop-size", type=int, default=16)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--elite", type=int, default=4)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--seed-genome", help="npz genome to seed the population")
    parser.add_argument("--eval", help="evaluate this genome over --eval-episodes and exit")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    print("loading fly brains...")
    from fruitloop import ConnectomeGraph
    from fruitloop.simulator import FlyBatch

    graph = ConnectomeGraph.load()
    flies = FlyBatch(args.batch, graph=graph)
    candidates = graph.neurons.query(
        superclass=["ol_sensory", "visual_projection", "cb_sensory"])

    # resolve all paths BEFORE BroodWarGame chdirs into the data directory
    out_dir = Path(args.out or f"evolution_runs/{time.strftime('%Y%m%d-%H%M%S')}").resolve()
    eval_path = Path(args.eval).resolve() if args.eval else None
    seed_path = Path(args.seed_genome).resolve() if args.seed_genome else None

    print("starting Brood War...")
    bw_data = Path(args.bw_data).resolve()
    game = BroodWarGame(bw_data, args.map, seed=4242)
    evaluator = BattleEvaluator(game, flies, MicroConfig())

    if eval_path:
        genome = Genome.load(eval_path)
        rng = np.random.default_rng(123)
        seeds = [int(rng.integers(1 << 30)) for _ in range(args.eval_episodes)]
        encoder, decoder = evaluator.build(genome)
        wins, scores = 0, []
        for s in seeds:
            score = evaluator._episode(genome, encoder, decoder, s)
            wins += score >= 2.0
            scores.append(score)
        print(f"{args.eval}: {wins}/{len(seeds)} wins, mean score {np.mean(scores):.3f}")
        return

    print(f"evolving into {out_dir} ...")
    seed_genome = Genome.load(seed_path) if seed_path else None
    best, history = evolve(
        evaluator, candidates, out_dir,
        generations=args.generations, pop_size=args.pop_size,
        episodes=args.episodes, elite=args.elite, seed=args.rng_seed,
        seed_genome=seed_genome)
    print(f"done. best fitness {best.fitness:.3f} -> {out_dir}/best_final.npz")


if __name__ == "__main__":
    main()

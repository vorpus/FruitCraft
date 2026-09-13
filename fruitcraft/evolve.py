"""Evolution of the fly<->game interface against battle outcomes.

The connectome and neuron model are never touched: the genome is the
INTERFACE — which MaleCNS sensory neurons each channel stimulates, the
calibration prototypes that define action readouts, and scalar knobs
(amplitude, decision window, stay floor). Selection pressure comes from
micro-battle results. This is the PRD's 'engineered first, later
learned/evolved' step made literal.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import unittypes
from .micro import MicroScenario, MicroConfig
from .fly.encoding import CHANNELS, SensoryEncoder
from .fly.decoding import ACTIONS, PROTOTYPES, ActionDecoder
from .fly.brainpool import UnitBrainPool
from .fly.controller import FlyCombatController, HivemindDirective

POP_SIZE = 120  # neurons per sensory channel


@dataclass
class Genome:
    populations: dict[str, np.ndarray]           # channel -> neuron ids (disjoint)
    prototypes: dict[str, dict[str, float]]      # action -> channel -> weight
    amplitude: float = 1.5
    decision_ms: float = 20.0
    stay_floor: float = 0.15
    fitness: float | None = None
    lineage: str = "seed"

    def save(self, path: Path):
        np.savez_compressed(
            path,
            **{f"pop_{c}": self.populations[c] for c in CHANNELS},
            meta=json.dumps({
                "prototypes": self.prototypes, "amplitude": self.amplitude,
                "decision_ms": self.decision_ms, "stay_floor": self.stay_floor,
                "fitness": self.fitness, "lineage": self.lineage,
            }))

    @classmethod
    def load(cls, path: Path) -> "Genome":
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return cls(
            populations={c: z[f"pop_{c}"] for c in CHANNELS},
            prototypes=meta["prototypes"], amplitude=meta["amplitude"],
            decision_ms=meta["decision_ms"], stay_floor=meta["stay_floor"],
            fitness=meta["fitness"], lineage=meta["lineage"])


def random_genome(candidates: np.ndarray, rng) -> Genome:
    chosen = rng.choice(candidates, size=POP_SIZE * len(CHANNELS), replace=False)
    pops = {c: np.sort(chosen[i * POP_SIZE:(i + 1) * POP_SIZE])
            for i, c in enumerate(CHANNELS)}
    protos = {a: dict(PROTOTYPES[a]) for a in ACTIONS}
    return Genome(populations=pops, prototypes=protos,
                  amplitude=float(rng.uniform(1.0, 2.0)),
                  decision_ms=float(rng.uniform(15, 30)),
                  stay_floor=float(rng.uniform(0.1, 0.25)))


def mutate(genome: Genome, candidates: np.ndarray, rng,
           swap_frac=0.10, scalar_sigma=0.15) -> Genome:
    used = np.concatenate(list(genome.populations.values()))
    free = np.setdiff1d(candidates, used)
    pops = {}
    for c in CHANNELS:
        pop = genome.populations[c].copy()
        k = max(1, int(len(pop) * swap_frac * rng.uniform(0, 2)))
        k = min(k, len(free))
        if k:
            out = rng.choice(len(pop), size=k, replace=False)
            incoming = rng.choice(free, size=k, replace=False)
            free = np.setdiff1d(free, incoming)
            free = np.union1d(free, pop[out])
            pop[out] = incoming
        pops[c] = np.sort(pop)
    protos = {}
    for a in ACTIONS:
        p = dict(genome.prototypes[a])
        for c, w in list(p.items()):
            p[c] = float(np.clip(w + rng.normal(0, scalar_sigma), 0.05, 1.0))
        if rng.random() < 0.15:  # recruit a new channel into this prototype
            c = CHANNELS[rng.integers(len(CHANNELS))]
            p.setdefault(c, float(rng.uniform(0.2, 0.8)))
        if len(p) > 1 and rng.random() < 0.10:  # drop one
            p.pop(list(p.keys())[rng.integers(len(p))])
        protos[a] = p
    return Genome(
        populations=pops, prototypes=protos,
        amplitude=float(np.clip(genome.amplitude * np.exp(rng.normal(0, scalar_sigma)), 0.6, 3.0)),
        decision_ms=float(np.clip(genome.decision_ms * np.exp(rng.normal(0, scalar_sigma)), 8, 40)),
        stay_floor=float(np.clip(genome.stay_floor * np.exp(rng.normal(0, scalar_sigma)), 0.03, 0.5)),
        lineage=f"mut({genome.lineage})")


def crossover(a: Genome, b: Genome, rng, candidates: np.ndarray | None = None) -> Genome:
    pops, taken = {}, np.array([], dtype=np.uint32)
    for c in CHANNELS:  # per-channel inheritance, repairing any overlap
        pop = (a if rng.random() < 0.5 else b).populations[c]
        clash = np.isin(pop, taken)
        if clash.any():
            kept = pop[~clash]
            donor = np.setdiff1d((b if rng.random() < 0.5 else a).populations[c],
                                 np.concatenate([taken, kept]))
            pop = np.concatenate([kept, donor[:clash.sum()]])
            short = clash.sum() - len(donor)
            if short > 0 and candidates is not None:  # heavy overlap: refill fresh
                fresh = np.setdiff1d(candidates, np.concatenate([taken, pop]))
                pop = np.concatenate([pop, rng.choice(fresh, size=short, replace=False)])
        pops[c] = np.sort(pop)
        taken = np.concatenate([taken, pops[c]])
    protos = {act: dict((a if rng.random() < 0.5 else b).prototypes[act]) for act in ACTIONS}
    pick = lambda x, y: x if rng.random() < 0.5 else y
    return Genome(populations=pops, prototypes=protos,
                  amplitude=pick(a, b).amplitude,
                  decision_ms=pick(a, b).decision_ms,
                  stay_floor=pick(a, b).stay_floor,
                  lineage=f"x({a.lineage},{b.lineage})")


class BattleEvaluator:
    """Fitness = mean battle score of a genome over shared episode seeds."""

    def __init__(self, game, flies, config: MicroConfig | None = None):
        self.game = game
        self.flies = flies
        self.config = config or MicroConfig()
        self.n_enemy = sum(n for _, n in self.config.enemy_army)
        self.n_ours = sum(n for _, n in self.config.our_army)

    def build(self, genome: Genome):
        encoder = SensoryEncoder(genome.populations, amplitude=genome.amplitude)
        import fruitcraft.fly.decoding as dec

        saved = dec.PROTOTYPES
        dec.PROTOTYPES = genome.prototypes
        try:
            decoder = ActionDecoder.calibrate(
                self.flies, encoder, stay_floor=genome.stay_floor)
        finally:
            dec.PROTOTYPES = saved
        return encoder, decoder

    def evaluate(self, genome: Genome, episode_seeds: list[int]) -> float:
        try:
            encoder, decoder = self.build(genome)
        except RuntimeError:
            return -1.0  # calibration found no discriminative readout
        scores = []
        for seed in episode_seeds:
            scores.append(self._episode(genome, encoder, decoder, seed))
        return float(np.mean(scores))

    def _episode(self, genome: Genome, encoder, decoder, seed: int) -> float:
        scenario = MicroScenario(self.game, self.config, seed=seed)
        pool = UnitBrainPool(self.flies, encoder, decoder,
                             decision_ms=genome.decision_ms)
        ctrl = FlyCombatController(self.game, pool,
                                  controlled_types={t for t, _ in self.config.our_army})
        objective = scenario.reset()
        ctrl.set_directive(HivemindDirective(*objective, attack=1.0))
        decision_frames = 12
        while True:
            scenario.script_enemy()
            if self.game.frame % decision_frames == 0:
                ctrl.tick()
            self.game.step(1)
            s = scenario.status()
            if s["done"] or self.game.result is not None:
                break
        pool.sync_units([])
        scenario.cleanup()
        self.game.step(24)
        kills = self.n_enemy - s["enemies_alive"]
        return (2.0 * s["won"]
                + kills / self.n_enemy
                + s["ours_alive"] / self.n_ours)


def evolve(evaluator: BattleEvaluator, candidates: np.ndarray, out_dir: Path,
           generations=15, pop_size=16, episodes=4, elite=4, seed=0,
           seed_genome: Genome | None = None, log=print):
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    population = [random_genome(candidates, rng) for _ in range(pop_size)]
    if seed_genome is not None:
        population[0] = seed_genome
    history = []
    for gen in range(generations):
        t0 = time.perf_counter()
        episode_seeds = [int(rng.integers(1 << 30)) for _ in range(episodes)]
        for g in population:  # common seeds: every genome fights the same battles
            g.fitness = evaluator.evaluate(g, episode_seeds)
        population.sort(key=lambda g: g.fitness, reverse=True)
        best, mean = population[0].fitness, float(np.mean([g.fitness for g in population]))
        history.append({"gen": gen, "best": best, "mean": mean,
                        "best_lineage": population[0].lineage,
                        "wall_s": round(time.perf_counter() - t0, 1)})
        log(f"gen {gen:02d}: best {best:.3f}  mean {mean:.3f}  "
            f"({history[-1]['wall_s']}s)  {population[0].lineage[:60]}")
        population[0].save(out_dir / f"best_gen{gen:02d}.npz")
        (out_dir / "history.json").write_text(json.dumps(history, indent=1))
        # mu+lambda: elites survive; children from tournament parents
        children = []
        while len(children) < pop_size - elite:
            def pick():
                i, j = rng.integers(pop_size), rng.integers(pop_size)
                return population[min(i, j)]  # sorted -> lower index = fitter
            a, b = pick(), pick()
            child = crossover(a, b, rng, candidates) if rng.random() < 0.5 else a
            children.append(mutate(child, candidates, rng))
        population = population[:elite] + children
    population.sort(key=lambda g: g.fitness or -9, reverse=True)
    population[0].save(out_dir / "best_final.npz")
    return population[0], history

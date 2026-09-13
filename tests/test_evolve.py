"""Genome operator invariants — pure numpy, no GPU or game needed."""

import numpy as np
import pytest

from fruitcraft.evolve import Genome, random_genome, mutate, crossover, POP_SIZE
from fruitcraft.fly.encoding import CHANNELS
from fruitcraft.fly.decoding import ACTIONS


@pytest.fixture
def candidates():
    return np.arange(20000, dtype=np.uint32)


def all_ids(genome):
    return np.concatenate([genome.populations[c] for c in CHANNELS])


def assert_valid(genome, candidates):
    ids = all_ids(genome)
    assert len(ids) == len(np.unique(ids)), "channel populations must stay disjoint"
    assert np.isin(ids, candidates).all(), "populations must come from the candidate pool"
    for a in ACTIONS:
        assert genome.prototypes[a], "every action needs a non-empty prototype"
        for c, w in genome.prototypes[a].items():
            assert c in CHANNELS and 0 < w <= 1
    assert 0.6 <= genome.amplitude <= 3.0
    assert 8 <= genome.decision_ms <= 40
    assert 0.03 <= genome.stay_floor <= 0.5


def test_random_genome_valid(candidates):
    g = random_genome(candidates, np.random.default_rng(0))
    assert_valid(g, candidates)
    assert all(len(g.populations[c]) == POP_SIZE for c in CHANNELS)


def test_mutation_chain_stays_valid(candidates):
    rng = np.random.default_rng(1)
    g = random_genome(candidates, rng)
    for _ in range(25):
        g = mutate(g, candidates, rng)
        assert_valid(g, candidates)


def test_mutation_changes_populations(candidates):
    rng = np.random.default_rng(2)
    g = random_genome(candidates, rng)
    m = mutate(g, candidates, rng)
    assert any(not np.array_equal(g.populations[c], m.populations[c]) for c in CHANNELS)


def test_crossover_valid_even_with_identical_parents(candidates):
    rng = np.random.default_rng(3)
    a = random_genome(candidates, rng)
    child = crossover(a, a, rng, candidates)  # maximal overlap case
    assert_valid(child, candidates)
    b = random_genome(candidates, rng)
    child2 = crossover(a, b, rng, candidates)
    assert_valid(child2, candidates)


def test_genome_save_load_roundtrip(tmp_path, candidates):
    g = random_genome(candidates, np.random.default_rng(4))
    g.fitness = 2.5
    path = tmp_path / "g.npz"
    g.save(path)
    loaded = Genome.load(path)
    for c in CHANNELS:
        np.testing.assert_array_equal(g.populations[c], loaded.populations[c])
    assert loaded.prototypes == g.prototypes
    assert loaded.fitness == 2.5
    assert loaded.amplitude == g.amplitude

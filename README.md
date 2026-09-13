# FruitCraft

StarCraft: Brood War, where every combat unit is driven by its own simulated
fly brain. Marries two engines:

- **[FruitLoop-CUDA](https://github.com/vorpus/FruitLoop-CUDA)** — batched CUDA
  simulator of the MaleCNS fly connectome (166,700 neurons shared once on GPU,
  independent state per fly).
- **[OpenSnowstorm](https://github.com/awest813/OpenSnowstorm---Brood-War)**
  (OpenBW lineage) — Linux-native, headless, deterministic Brood War engine,
  pinned as a submodule and driven through its `mini-openbwapi` layer.

```
Brood War state ──► per-unit observation ──► sensory encoder (13 channels)
                                                    │  stimulation
                                     one MaleCNS fly brain per unit (GPU batch)
                                                    │  spikes (20 ms window)
issued unit commands ◄── action decoder (calibrated readouts) ◄──┘
```

The **hivemind** is a per-team directive (objective + aggression) that enters
each fly only as additional sensory input — a suggestion, never an override.
Each fly remains free to do its own thing.

## Status

Fly pipeline validated end-to-end on the real connectome: calibration finds
150 discriminative readout neurons per action and all five action prototypes
decode to their own action; a batched 24-unit decision costs ~19 ms wall.
The engine bridge builds and imports. **Not yet run against a real game** —
see `HANDOFF.md`. You must supply your own legally obtained Brood War data
files (`STARDAT.MPQ`, `BROODAT.MPQ`, `patch_rt.mpq`) plus a 2-player melee map.

## Setup

```bash
git clone --recursive git@github.com:vorpus/FruitCraft.git
python3 -m venv .venv
.venv/bin/pip install numpy pytest cmake ninja pybind11 cupy-cuda13x
.venv/bin/pip install -e ../FruitLoop-CUDA     # fruitloop + its processed MaleCNS data
./scripts/build_engine.sh                      # builds fruitcraft/_engine (pybind11)
.venv/bin/python -m pytest tests/              # no game data needed
.venv/bin/python scripts/run_micro.py --bw-data <dir-with-MPQs> --map "maps/(2)....scm"
```

## Design notes

- **Micro first.** OpenSnowstorm's computer AI does not yet execute aiscript
  build orders, so scenarios spawn armies directly and the opponent is
  scripted (the engine lets us command any unit's owner). Base building via a
  scripted macro layer or upstream aiscript support comes later.
- **Per-decision reset.** Both this connectome (at lif_v1 gain) and dense
  synthetic graphs latch into a saturated all-excited attractor that never
  decays, so by default each fly is reset at the start of its decision tick —
  every decision is a clean 20 ms transient matching the calibration
  condition. Persistent cross-tick state (`reset_each_tick=False`) is the
  long-term research mode and needs a sub-critical gain.
- **Engine facts** (from source): one game per process; MPQs and
  `bwapi-data/bwapi.ini` are read from the process cwd; only Attack_Move,
  Attack_Unit, Move, Build, Train, Right_Click_Unit are wired in
  `issueCommand`; melee defeat triggers when a side's building count is 0.

Layout: `bridge/` C++ pybind11 bridge → `fruitcraft/_engine`; `fruitcraft/`
Python (engine wrapper, micro scenarios, `fly/` encoder–pool–decoder–controller);
`scripts/` build + E2E; `tests/` (no game data required).

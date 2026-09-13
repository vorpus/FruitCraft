# FruitCraft — session handoff (updated 2026-09-12)

## Update 2026-09-12
- All 14 tests pass. Fixed the two initial failures properly:
  - ActionDecoder now normalizes each action's logit by that readout's own
    calibration response (readout excitability differs); STAY is a relative
    floor (default 0.15 of calibration response).
  - Discovered both the synthetic test graph AND the real MaleCNS graph (at
    lif_v1 gain 0.005) latch into a saturated self-sustaining attractor —
    activity never decays after stimulation stops (~1% refractory-limited
    ceiling). UnitBrainPool therefore defaults to `reset_each_tick=True`
    (reset the fly's slots before each decision; reset must precede
    encoder.apply since reset clears i_ext). `reset_each_tick=False` is the
    persistent-state research mode; it needs a sub-critical gain to be useful.
- Real-graph validation: calibration finds full 150-neuron readouts for all 5
  actions; all 5 prototypes decode to their own action; a batched 24-unit
  decision tick costs ~19 ms wall (faster than real time at 1 decision / 0.5
  game-seconds).
- Added README.md + pyproject.toml.
- STILL BLOCKED on Brood War MPQs for anything engine-side (none on machine).

---

# Original handoff (2026-09-11)

StarCraft: Brood War (OpenSnowstorm) driven by FruitLoop fly brains — one
MaleCNS fly brain per combat unit. Sibling repo: `../FruitLoop-CUDA` (complete,
pushed: engine, tests, benchmarks; peak ~115k fly-ms/s on the 4090).

## State: builds and imports; fly pipeline written; NOT yet run against a real game

### Done
- **Engine**: `awest813/OpenSnowstorm---Brood-War` pinned as submodule
  (`third_party/opensnowstorm` @ 57da19c). Headless, deterministic, BWAPI-like.
- **C++ bridge** (`bridge/engine_bindings.cpp` → `fruitcraft/_engine`):
  pybind11 over mini-openbwapi — update/frame/events, fog-filtered unit dumps,
  whitelisted commands, `create_unit`/`kill_unit`, seeds, snapshots.
  **Builds clean and imports** (`./scripts/build_engine.sh`, needs `.venv`).
- **Python layer** (`fruitcraft/`):
  - `engine.py` — `BroodWarGame`: writes `bwapi-data/bwapi.ini`, chdirs to the
    BW data dir (engine loads MPQs from cwd), match lifecycle + restart, fog-
    respecting observation getters, command helpers.
  - `micro.py` — `MicroScenario`: spawns both armies mid-map, scripts the
    opponent with periodic attack-move pulses (commands can be issued as any
    unit's owner), episode status/cleanup.
  - `fly/encoding.py` — 13 sensory channels (enemy direction/proximity, own
    state, hivemind directive) → disjoint MaleCNS sensory populations.
  - `fly/decoding.py` — calibration: probe per-action prototype patterns, pick
    discriminative downstream readout neurons; argmax decode with STAY floor.
  - `fly/brainpool.py` — `UnitBrainPool`: unit-tag ↔ fly-slot, batched
    encode → simulate (20 ms) → decode tick.
  - `fly/controller.py` — `FlyCombatController` + `HivemindDirective`
    (objective + aggression as *suggestion*, encoded as ordinary senses);
    action → engine command mapping (BW y-axis points south; no Hold command
    in engine → STAY = move-to-self).
  - `scripts/run_micro.py` — full E2E: calibrate → spawn 8 marines vs 10
    zerglings → fly swarm vs scripted enemy, 3 episodes.
- **Tests** (`tests/`): encoding/geometry, controller command mapping (fake
  game), fly pipeline calibrate+tick on a synthetic graph (GPU).
  **Written but NOT yet executed** — run them first.

### Key engine facts (learned by reading source)
- One game per process (global singleton). Data files + `bwapi-data/bwapi.ini`
  are read from the process **cwd**. First `Game::update()` starts the match.
- `issueCommand` supports ONLY: Attack_Move, Attack_Unit, Move, Build, Train,
  Right_Click_Unit. Anything else calls the engine's fatal error handler —
  bridge whitelists.
- Computer AI does **no macro** (no aiscript.bin execution yet) → decision:
  micro-first, opponent scripted by us; "native AI base building" parked.
- Melee defeat = your building count hits 0 → don't kill starting bases when
  setting up micro fights.

### Blocked on (user input needed)
- **Brood War data files**: `STARDAT.MPQ`, `BROODAT.MPQ`, `patch_rt.mpq` from a
  legally owned install, in some dir, plus a 2-player melee map (`.scm/.scx`).
  Without them nothing engine-side can actually run (bridge import works).

### Next steps, in order
1. `cd ~/projects/FruitCraft && .venv/bin/python -m pytest tests/ -q`
   (needs GPU; fruitloop installed editable from ../FruitLoop-CUDA — its
   processed MaleCNS data already built there).
2. Engine smoke test with real MPQs: start `BroodWarGame`, step 100 frames,
   dump units, spawn a marine, move it (Phase 0 of the user's BW TDD).
3. `python scripts/run_micro.py --bw-data <dir> --map <map>` — first real
   fly-vs-zergling battle; tune encoder amplitude / decision cadence /
   stay_floor if the swarm is inert or spastic.
4. Missing repo scaffolding: pyproject.toml, README.md, CI-less lint pass.
5. Then: vectorized envs (one game per process, TDD §20), win-rate eval
   vs scripted baseline, macro layer (scripted build orders via Train/Build).

### Repos
- `FruitLoop-CUDA`: pushed (github.com/vorpus/FruitLoop-CUDA), clean.
- `FruitCraft`: this commit. Author identity: Li Zhang <li3zhang@gmail.com>
  (global git config on this machine).

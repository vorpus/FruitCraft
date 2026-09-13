# FruitCraft — session handoff (updated 2026-09-12, night: EVOLUTION + VIEWER)

## Update 2026-09-12 night — evolution works; battle viewer shipped
- `fruitcraft/evolve.py` + `scripts/evolve_micro.py`: GA over the interface
  (channel populations, prototypes, amplitude/decision_ms/stay_floor).
  Run01 (22 gens × pop 20 × 5 shared-seed episodes, ~25 s/gen): best fitness
  3.85/4.0 at gen 13. **Champion wins 18/20 held-out episodes** vs 2/10 for
  the hand-engineered mapping. Champion committed: `genomes/champion_run01.npz`
  (evaluate: `scripts/evolve_micro.py --eval genomes/champion_run01.npz`).
  NOTE: launch long runs with nohup/setsid — a background-shell timeout killed
  run01 at gen 21/30 (fitness had plateaued; harmless this time).
- Introspection plumbing: decoder exposes `logits_for`/`decode_from_logits`,
  pool keeps `last_logits`, controller keeps `last_observations`/`last_commands`
  (and `_execute` returns the issued command).
- `fruitcraft/record.py` + `scripts/record_battle.py`: record battles to
  compact JSON (sampled unit states every 2 frames + per-decision drives,
  logits, action, command, deaths). `battles.json` committed: 3 champion +
  2 pre-evolution episodes.
- `viewer/index.html` — **Swarm Observer** web replay viewer (published as a
  Claude artifact; battles.json ships alongside): canvas battlefield with hp
  arcs, action glyphs, hivemind crosshair, command arrow, death flashes, kill
  feed; click a marine for its brain panel (13 channel drives, 5 normalized
  action logits with stay-floor line, decoded action + issued command,
  decision-history strip); episode chips with WIN/LOSS pills; transport with
  speed control and death-marked scrubber. To refresh data: rerun
  record_battle.py and republish index.html + battles.json.
- `scripts/analyze_run.py` plots run fitness history.



## Update 2026-09-12 evening — the game runs; Milestone 7 reached
- User supplied MPQs -> `bwdata/` (gitignored; engine needs exact names
  StarDat.mpq / BrooDat.mpq / Patch_rt.mpq — wrapper auto-symlinks case
  variants). Maps downloaded: `(2)Benzene.scx` (melee, from sc-docker repo),
  `m5v5_c_far.scm` (TorchCraft micro map, unused so far).
- Phase 0 smoke test PASSES (`scripts/smoke_engine.py`): headless boot, melee
  start units enumerated, spawned marine moves under program control.
- Full battles run at ~500x real time (~0.6 s wall per 8-marine-vs-10-zergling
  episode including all fly-brain decisions).
- Engine combat facts (all verified empirically, tests in git history):
  - attack_unit REQUIRES target coordinates (silently ret=False without).
  - attack_move >> attack_unit for ranged units (engages at range vs chasing).
  - NO auto-acquire: uncommanded units never fight. Every offensive intent
    must be an explicit command.
  - Re-issuing commands resets attack sequences -> DPS ~0 if spammed.
    Controller now only issues when the decoded action changes or the unit
    went idle (order 3 = guard); STAY issues nothing.
- Results (8 fly marines vs 10 scripted zerglings, 10 episodes):
  - scripted attack-move baseline: 10/10 wins (concentrated deathball).
  - fly swarm: 2/10 wins; +centroid-focus attack variant: 3/10.
  - Fly losses come from movement decodes scattering the army mid-fight and
    per-unit targeting splitting focus. Dynamic hivemind directive (live
    enemy centroid) alone didn't help.
- Assessment: the engineered encoder/decoder mapping has hit its ceiling.
  The promising next lever is SELECTION/LEARNING over the interface (evolve
  encoder populations / readouts / prototypes against battle outcome — at
  0.6 s/episode we get ~5k battles/hour), and/or letting the hivemind carry
  cohesion signals the flies are calibrated to obey more strongly.
- Also possible next: SDL rendering of one env to actually watch battles
  (OPENBW_ENABLE_UI build flag), vectorized multi-process envs, win-rate CI.



## Update 2026-09-12 (later)
- Encoder switched to POPULATION CODING: a channel's value sets the fraction
  of its population stimulated, not per-neuron current (per-neuron current
  saturates LIF firing above ~threshold/tau, making analog values binary).
- Command-level behavioral eval on the real graph: 6/6 mixed scenarios produce
  sensible game commands (navigate to objective via decoded moves or the
  attack->attack_move-objective fallback; engage nearby enemies). Note the
  action space is deliberately degenerate with no visible enemy: decoded
  'attack' falls back to attack-move at the hivemind objective, which IS
  navigation — evaluate at the command level, not the action-label level.
- Gain sweep (0.005/0.002/0.001): default 0.005 kept; sub-critical gains not
  clearly better and weaken calibration. Revisit only for persistent-state mode.

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

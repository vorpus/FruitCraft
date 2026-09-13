"""Python wrapper around the OpenSnowstorm bridge (one Brood War game per process).

The engine loads MPQ data files from the process working directory and reads
./bwapi-data/bwapi.ini at match start. This wrapper owns both: point it at a
directory containing your legally obtained Brood War data files
(STARDAT.MPQ, BROODAT.MPQ, patch_rt.mpq) and a map path.
"""

from __future__ import annotations

import os
from pathlib import Path

# exact names the engine opens (case-sensitive on Linux)
REQUIRED_MPQS = ["StarDat.mpq", "BrooDat.mpq", "Patch_rt.mpq"]

MATCH_START, MATCH_END, MATCH_FRAME, UNIT_DESTROY = None, None, None, None  # bound on import


def _find_mpq(data_dir: Path, name: str) -> Path | None:
    for candidate in data_dir.iterdir():
        if candidate.name.lower() == name.lower():
            return candidate
    return None


class BroodWarGame:
    """Frame-stepped headless Brood War match."""

    def __init__(self, data_dir, map_path, my_race="terran", enemy_race="zerg",
                 melee=True, seed: int | None = None):
        global MATCH_START, MATCH_END, MATCH_FRAME, UNIT_DESTROY
        data_dir = Path(data_dir).resolve()
        missing = []
        for name in REQUIRED_MPQS:
            found = _find_mpq(data_dir, name)
            if found is None:
                missing.append(name)
            elif found.name != name:  # engine wants exact case; bridge the gap
                (data_dir / name).symlink_to(found.name)
        if missing:
            raise FileNotFoundError(
                f"Brood War data files missing from {data_dir}: {missing}. "
                f"Copy them from a legally owned StarCraft: Brood War install.")
        os.chdir(data_dir)  # engine reads MPQs and bwapi.ini relative to cwd
        ini_dir = data_dir / "bwapi-data"
        ini_dir.mkdir(exist_ok=True)
        game_type = "MELEE" if melee else "USE_MAP_SETTINGS"
        (ini_dir / "bwapi.ini").write_text(
            f"[auto_menu]\nmap = {map_path}\ngame_type = {game_type}\n"
            f"race = {my_race}\nenemy_race = {enemy_race}\n")

        from fruitcraft import _engine  # noqa: import here so tests run without the .so

        self._e = _engine
        MATCH_START = _engine.EVENTS["MATCH_START"]
        MATCH_END = _engine.EVENTS["MATCH_END"]
        MATCH_FRAME = _engine.EVENTS["MATCH_FRAME"]
        UNIT_DESTROY = _engine.EVENTS["UNIT_DESTROY"]
        self.COMMANDS = _engine.COMMANDS
        self.seed = seed
        self.result: bool | None = None  # None = in progress, True = won
        self._e.set_gui(False)
        self._begin_match()

    def _begin_match(self):
        self.result = None
        for _ in range(10):
            events = self._e.update()
            if any(e["type"] == MATCH_START for e in events):
                break
        else:
            raise RuntimeError("match failed to start")
        if self.seed is not None:
            self._e.set_random_seed(self.seed)
        self.self_id = self._e.self_player()
        self.enemy_id = self._e.enemy_player()

    # ------------------------------------------------------------------ state
    @property
    def frame(self) -> int:
        return self._e.frame()

    def step(self, frames: int = 1) -> list[dict]:
        """Advance N frames; returns all events. Sets .result on match end."""
        events = []
        for _ in range(frames):
            for e in self._e.update():
                events.append(e)
                if e["type"] == MATCH_END:
                    self.result = e["is_winner"]
            if self.result is not None:
                break
        return events

    def restart(self, seed: int | None = None):
        """End the current match and start a fresh one on the same map."""
        if seed is not None:
            self.seed = seed
        self._e.leave_game()
        while self.result is None:
            self.step(1)
        self._begin_match()

    def my_units(self) -> list[dict]:
        return self._e.units(self.self_id)

    def enemy_units_visible(self) -> list[dict]:
        """Enemy units through our fog of war (never full state)."""
        return self._e.units(self.enemy_id, visible_to=self.self_id)

    def enemy_units_all(self) -> list[dict]:
        """Full enemy state — scenario scripting/metrics ONLY, never policy input."""
        return self._e.units(self.enemy_id)

    def player_info(self, player_id=None) -> dict:
        return self._e.player_info(self.self_id if player_id is None else player_id)

    # --------------------------------------------------------------- commands
    def move(self, unit_id: int, x: int, y: int) -> bool:
        return self._e.command(unit_id, self.COMMANDS["MOVE"], -1, int(x), int(y))

    def attack_move(self, unit_id: int, x: int, y: int) -> bool:
        return self._e.command(unit_id, self.COMMANDS["ATTACK_MOVE"], -1, int(x), int(y))

    def attack_unit(self, unit_id: int, target_id: int, x: int, y: int) -> bool:
        # the engine rejects attack-unit commands without target coordinates
        return self._e.command(unit_id, self.COMMANDS["ATTACK_UNIT"], target_id,
                               int(x), int(y))

    # ---------------------------------------------------------- scenario ops
    def spawn(self, player_id: int, unit_type: int, x: int, y: int) -> int:
        return self._e.create_unit(player_id, unit_type, int(x), int(y))

    def kill(self, unit_id: int):
        self._e.kill_unit(unit_id)

    def map_pixel_size(self) -> tuple[int, int]:
        return self._e.map_width() * 32, self._e.map_height() * 32

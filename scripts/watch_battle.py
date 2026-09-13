#!/bin/sh
'''exec' "$(dirname "$0")/../.venv/bin/python" "$0" "$@" #'''
"""Watch fly-brain battles live: the engine's SDL window (real BW graphics)
plus a synchronized Fly Brain telemetry window for one followed marine —
its senses, its 13 input channels, its action readouts, and the command
it issued. Optionally record both windows to an mp4.

Usage:
  scripts/watch_battle.py [--genome genomes/champion_run01.npz]
      [--episodes 3] [--speed 1.0] [--no-brain] [--record battle.mp4]
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # engine mutes anyway
os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "10,40")  # pin for recording

from fruitcraft.engine import BroodWarGame
from fruitcraft.micro import MicroScenario, MicroConfig
from fruitcraft.evolve import BattleEvaluator, Genome
from fruitcraft.fly.brainpool import UnitBrainPool
from fruitcraft.fly.controller import FlyCombatController, HivemindDirective

GAME_POS = (10, 40)
PANEL_POS = (830, 40)
RECORD_OFFSET = (0, 36)      # skip the desktop top bar
RECORD_REGION = "1400x640"   # both windows; NOTE: records that screen region —
                             # anything visible in it (other windows) is captured


def place_game_window(x, y, tries=25):
    """Move+raise the engine's 800x600 window with EWMH messages (GNOME's
    window manager ignores both SDL position hints and XConfigureWindow).
    The window is identified by our own PID + its size (its legacy WM_NAME is
    empty), and a second corrective move cancels the WM's frame offsets."""
    from Xlib import X, display as xdisplay, protocol

    d = xdisplay.Display()
    root = d.screen().root
    net_clients = d.intern_atom("_NET_CLIENT_LIST")
    net_pid = d.intern_atom("_NET_WM_PID")
    net_move = d.intern_atom("_NET_MOVERESIZE_WINDOW")
    net_active = d.intern_atom("_NET_ACTIVE_WINDOW")
    mask = X.SubstructureRedirectMask | X.SubstructureNotifyMask

    def find():
        prop = root.get_full_property(net_clients, X.AnyPropertyType)
        for wid in (prop.value if prop else []):
            w = d.create_resource_object("window", wid)
            try:
                p = w.get_full_property(net_pid, X.AnyPropertyType)
                g = w.get_geometry()
            except Exception:
                continue
            if p and p.value and p.value[0] == os.getpid() \
                    and g.width == 800 and g.height == 600:
                return w
        return None

    def move(w, mx, my):
        flags = 1 | (1 << 8) | (1 << 9) | (2 << 12)  # NW gravity, x+y, pager
        root.send_event(protocol.event.ClientMessage(
            window=w, client_type=net_move,
            data=(32, [flags, int(mx) & 0xFFFFFFFF, int(my) & 0xFFFFFFFF, 0, 0])),
            event_mask=mask)
        d.flush()
        d.sync()

    for _ in range(tries):
        w = find()
        if w is not None:
            move(w, x, y)
            time.sleep(0.4)
            c = root.translate_coords(w, 0, 0)  # achieved client-area origin
            move(w, x - (c.x - x), y - (c.y - y))
            root.send_event(protocol.event.ClientMessage(
                window=w, client_type=net_active,
                data=(32, [2, X.CurrentTime, 0, 0, 0])), event_mask=mask)
            d.flush()
            d.sync()
            return True
        time.sleep(0.2)
    return False


def start_recorder(out_path: str):
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-f", "x11grab",
           "-framerate", "12", "-video_size", RECORD_REGION,
           "-i", os.environ.get("DISPLAY", ":0") + f"+{RECORD_OFFSET[0]},{RECORD_OFFSET[1]}",
           "-pix_fmt", "yuv420p", "-preset", "ultrafast", out_path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_recorder(recorder):
    # ffmpeg must finalize the mp4 trailer: ask politely, escalate if needed
    try:
        recorder.stdin.write(b"q")
        recorder.stdin.flush()
        recorder.wait(timeout=10)
    except Exception:
        import signal

        recorder.send_signal(signal.SIGINT)
        try:
            recorder.wait(timeout=10)
        except Exception:
            recorder.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bw-data", default="bwdata")
    parser.add_argument("--map", default="maps/(2)Benzene.scx")
    parser.add_argument("--genome", default="genomes/champion_run01.npz")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--speed", type=float, default=1.0, help="game speed multiplier")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-brain", action="store_true", help="skip the telemetry window")
    parser.add_argument("--record", help="record game + brain panel to this mp4")
    args = parser.parse_args()

    genome_path = Path(args.genome).resolve()
    record_path = str(Path(args.record).resolve()) if args.record else None
    print("loading fly brains...")
    from fruitloop import ConnectomeGraph
    from fruitloop.simulator import FlyBatch

    graph = ConnectomeGraph.load()
    flies = FlyBatch(16, graph=graph)

    print("starting Brood War with renderer...")
    game = BroodWarGame(Path(args.bw_data).resolve(), args.map, seed=4242, gui=True)
    game.step(1)  # let the window appear
    if not place_game_window(*GAME_POS):
        print("(could not reposition the game window; layout may overlap)")
    config = MicroConfig()
    evaluator = BattleEvaluator(game, flies, config)
    genome = Genome.load(genome_path)
    encoder, decoder = evaluator.build(genome)
    rng = np.random.default_rng(args.seed)

    view = None
    if not args.no_brain:
        from fruitcraft.brainview import BrainView

        view = BrainView(stay_floor=decoder.stay_floor, position=PANEL_POS)

    recorder = start_recorder(record_path) if record_path else None
    if recorder:
        print(f"recording to {record_path}")

    try:
        for episode in range(args.episodes):
            scenario = MicroScenario(game, config, seed=int(rng.integers(1 << 30)))
            pool = UnitBrainPool(flies, encoder, decoder, decision_ms=genome.decision_ms)
            ctrl = FlyCombatController(game, pool,
                                      controlled_types={t for t, _ in config.our_army})
            objective = scenario.reset()
            ctrl.set_directive(HivemindDirective(*objective, attack=1.0))
            print(f"episode {episode}: watching (hero marine ring follows the brain panel)")
            t0 = time.perf_counter()
            frame0 = game.frame
            hero = None
            while True:
                scenario.script_enemy()
                if game.frame % 12 == 0:
                    actions = ctrl.tick()
                    alive = sorted(ctrl.last_observations.keys())
                    if alive and hero not in alive:
                        hero = alive[0]
                    if hero in ctrl.last_observations:
                        # camera follows the hero: the marine whose brain is shown
                        u = next((x for x in game.my_units() if x["id"] == hero), None)
                        if u:
                            game.look_at(u["x"], u["y"])
                        if view is not None and view.alive():
                            view.update(hero, ctrl.last_observations[hero],
                                        pool.last_logits[hero], actions[hero],
                                        ctrl.last_commands.get(hero))
                game.step(1)
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
    finally:
        if recorder:
            stop_recorder(recorder)
            print(f"recording saved: {record_path}")


if __name__ == "__main__":
    main()

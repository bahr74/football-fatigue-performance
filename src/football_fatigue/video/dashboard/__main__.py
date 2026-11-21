"""``python -m football_fatigue.video.dashboard --data-root <read-only data root> [--out-root data/processed/video] [--port 5830]``

Starts the native video player (``football_fatigue.video.player``, port + 1) and the Streamlit review dashboard on
http://127.0.0.1:<port> (this computer only by default). The dashboard embeds the player in its Viewer tab; the player
is also usable on its own at http://127.0.0.1:<port + 1>/."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m football_fatigue.video.dashboard")
    ap.add_argument("--data-root", required=True, help="read-only data root (videos, tracker replays, ground truth)")
    ap.add_argument("--out-root", default="data/processed/video", help="Session B run folder root")
    ap.add_argument("--port", type=int, default=5830)
    ap.add_argument("--player-port", type=int, default=None, help="native player port (default: --port + 1)")
    ap.add_argument("--address", default="127.0.0.1")
    ap.add_argument("--headless", action="store_true", help="do not open a browser")
    a = ap.parse_args(argv)
    data_root, out_root = Path(a.data_root).resolve(), Path(a.out_root).resolve()
    pport = a.player_port or a.port + 1
    player = subprocess.Popen([sys.executable, "-m", "football_fatigue.video.player", "--data-root", str(data_root),
                               "--out-root", str(out_root), "--host", a.address, "--port", str(pport), "--no-browser"],
                              env=os.environ.copy())
    print(f"Native video player: http://{a.address}:{pport}/", flush=True)
    app = Path(__file__).with_name("app.py")
    try:
        return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(a.port),
                                "--server.address", a.address, "--server.headless", str(a.headless).lower(),
                                "--browser.gatherUsageStats", "false", "--", "--data-root", str(data_root),
                                "--out-root", str(out_root), "--player-url", f"http://{a.address}:{pport}"])
    finally:
        player.terminate()


if __name__ == "__main__":
    sys.exit(main())

"""Minimal Chrome DevTools Protocol client (stdlib only) for real-browser QA with headless Microsoft Edge.

Adapted from the M10.8 media-QA client (``data/processed/vision_eval/m108-…/scripts/m108_cdp.py``): real time (no
virtual-time budget, which stalls media), screenshots, JS evaluation, mouse input.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"


class WS:
    def __init__(self, url: str):
        host_port, path = url[len("ws://"):].split("/", 1)
        host, port = host_port.split(":")
        self.s = socket.create_connection((host, int(port)), timeout=120)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET /{path} HTTP/1.1\r\nHost: {host_port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += self.s.recv(4096)
        if b" 101 " not in buf.split(b"\r\n")[0]:
            raise RuntimeError(buf[:200])
        self.rest = buf.split(b"\r\n\r\n", 1)[1]

    def _recv(self, n: int) -> bytes:
        while len(self.rest) < n:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                raise ConnectionError("socket closed")
            self.rest += chunk
        out, self.rest = self.rest[:n], self.rest[n:]
        return out

    def send(self, text: str) -> None:
        data = text.encode()
        hdr = bytearray([0x81])
        mask = os.urandom(4)
        n = len(data)
        hdr += bytes([0x80 | n]) if n < 126 else (bytes([0x80 | 126]) + struct.pack(">H", n) if n < 65536
                                                  else bytes([0x80 | 127]) + struct.pack(">Q", n))
        self.s.sendall(bytes(hdr) + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self) -> str:
        msg = b""
        while True:
            b0, b1 = self._recv(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._recv(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._recv(8))[0]
            payload = self._recv(n)
            if b0 & 0x0F == 0x9:
                continue
            msg += payload
            if b0 & 0x80:
                return msg.decode("utf-8", "replace")


class Browser:
    def __init__(self, port: int = 9344, width: int = 1600, height: int = 1100, exe: str = EDGE):
        self.port, self.w, self.h, self.exe = port, width, height, exe
        self.prof = tempfile.mkdtemp(prefix="sessb_edge_")
        self.console: list[str] = []

    def __enter__(self) -> "Browser":
        self.proc = subprocess.Popen([self.exe, "--headless=new", f"--remote-debugging-port={self.port}",
                                      f"--user-data-dir={self.prof}", "--no-first-run",
                                      "--autoplay-policy=no-user-gesture-required", f"--window-size={self.w},{self.h}",
                                      "--hide-scrollbars", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tabs = []
        for _ in range(80):
            try:
                tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json", timeout=2).read())
                break
            except Exception:                              # noqa: BLE001
                time.sleep(0.5)
        page = next(t for t in tabs if t.get("type") == "page")
        self.ws = WS(page["webSocketDebuggerUrl"])
        self.i = 0
        self.cmd("Page.enable")
        self.cmd("Runtime.enable")
        self.cmd("Emulation.setDeviceMetricsOverride", width=self.w, height=self.h, deviceScaleFactor=1, mobile=False)
        return self

    def __exit__(self, *a) -> None:
        try:
            self.cmd("Browser.close")
        except Exception:                                  # noqa: BLE001
            pass
        try:
            self.proc.wait(10)
        except Exception:                                  # noqa: BLE001
            self.proc.kill()
        shutil.rmtree(self.prof, ignore_errors=True)

    def cmd(self, method: str, **params):
        self.i += 1
        mid = self.i
        self.ws.send(json.dumps({"id": mid, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("method") == "Runtime.consoleAPICalled" and m["params"].get("type") in ("error", "warning"):
                self.console.append(m["params"]["type"] + ": " + " ".join(
                    str(a.get("value", a.get("description", ""))) for a in m["params"]["args"])[:400])
            elif m.get("method") == "Runtime.exceptionThrown":
                d = m["params"]["exceptionDetails"]
                self.console.append("EXCEPTION " + str(d.get("exception", {}).get("description", d.get("text")))[:400])
            if m.get("id") == mid:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def goto(self, url: str, settle: float = 1.0) -> None:
        self.cmd("Page.navigate", url=url)
        self.wait("document.readyState === 'complete'", 60)
        time.sleep(settle)

    def eval(self, expr: str, await_promise: bool = True):
        r = self.cmd("Runtime.evaluate", expression=expr, awaitPromise=await_promise, returnByValue=True)
        if "exceptionDetails" in r:
            d = r["exceptionDetails"]
            raise RuntimeError(d.get("exception", {}).get("description", d.get("text")))
        return r["result"].get("value")

    def wait(self, expr: str, timeout: float = 30, poll: float = 0.1) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if self.eval(expr):
                    return True
            except RuntimeError:
                pass
            time.sleep(poll)
        raise TimeoutError(expr)

    def mouse(self, kind: str, x: float, y: float, **kw) -> None:
        self.cmd("Input.dispatchMouseEvent", type=kind, x=x, y=y, **kw)

    def click(self, x: float, y: float) -> None:
        self.mouse("mousePressed", x, y, button="left", clickCount=1)
        self.mouse("mouseReleased", x, y, button="left", clickCount=1)

    def screenshot(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(base64.b64decode(self.cmd("Page.captureScreenshot", format="png")["data"]))

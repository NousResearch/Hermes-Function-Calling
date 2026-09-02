#!/usr/bin/env python3
"""Film a tmux+ttyd terminal of the Hermes toolkit via headless Chrome.

Adapted from proving-it-works examples/film-terminal.py: real characters in
a real shell, screenshotted from the canvas ttyd draws. Software GL is
required or headless Chrome paints that canvas black.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
FPS = 2.0
DEBUG_PORT = 7222
SESSION = "hermes-demo"
CHROME = "/usr/bin/google-chrome"


def pane_command() -> str:
    r = subprocess.run(
        ["tmux", "display-message", "-p", "-t", SESSION, "#{pane_current_command}"],
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def wait_for_shell(timeout: float = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pane_command() in ("bash", "sh", "zsh"):
            return
        time.sleep(0.4)
    raise SystemExit("shell never came back; refusing to type into a running program")


def send(keys: str, enter: bool = True) -> None:
    cmd = ["tmux", "send-keys", "-t", SESSION, keys]
    if enter:
        cmd.append("Enter")
    subprocess.run(cmd, check=True, capture_output=True)


def send_and_wait(keys: str, timeout: float = 120) -> None:
    """Type a command, wait for it to start, then wait for the shell to return."""
    send(keys)
    deadline = time.time() + 8
    while time.time() < deadline and pane_command() in ("bash", "sh", "zsh"):
        time.sleep(0.1)
    wait_for_shell(timeout=timeout)


async def cdp(ws, mid, method, params=None):
    await ws.send(json.dumps({"id": mid, "method": method, "params": params or {}}))
    while True:
        msg = json.loads(await ws.recv())
        if msg.get("id") == mid:
            return msg.get("result", {})


async def shoot(ws, outdir: Path, stop: asyncio.Event, counter: list[int]) -> None:
    import websockets  # noqa: F401 — imported by caller

    mid = 1000
    while not stop.is_set():
        mid += 1
        try:
            res = await asyncio.wait_for(
                cdp(ws, mid, "Page.captureScreenshot", {"format": "png"}), timeout=3
            )
            data = res.get("data")
            if data:
                (outdir / f"f{counter[0]:05d}.png").write_bytes(base64.b64decode(data))
                counter[0] += 1
        except asyncio.TimeoutError:
            pass
        await asyncio.sleep(1 / FPS)


async def preflight(ws) -> None:
    wait_for_shell()
    send("echo PREFLIGHT_OK")
    await asyncio.sleep(1.5)
    res = await cdp(ws, 900, "Page.captureScreenshot", {"format": "png"})
    png = base64.b64decode(res["data"])
    from io import BytesIO

    from PIL import Image

    im = Image.open(BytesIO(png)).convert("L")
    px = list(im.getdata())
    lit = sum(1 for v in px if v > 90) / len(px)
    if lit < 0.002:
        raise SystemExit(
            f"preflight failed: terminal renders blank ({lit:.4%} lit pixels). "
            "Check software GL flags before filming."
        )
    print(f"preflight ok: {lit:.2%} of pixels lit")
    send("clear")
    await asyncio.sleep(0.8)


async def beats_tools() -> None:
    await asyncio.sleep(1.2)
    send_and_wait("python proof/show.py tools")
    await asyncio.sleep(5)


async def beats_parse() -> None:
    await asyncio.sleep(1.0)
    send("clear")
    await asyncio.sleep(0.6)
    send_and_wait("python proof/show.py parse")
    await asyncio.sleep(5)


async def beats_validate() -> None:
    await asyncio.sleep(1.0)
    send("clear")
    await asyncio.sleep(0.6)
    send_and_wait("python proof/show.py validate")
    await asyncio.sleep(6)


async def beats_execute() -> None:
    await asyncio.sleep(1.0)
    send("clear")
    await asyncio.sleep(0.6)
    send_and_wait("python proof/show.py execute")
    await asyncio.sleep(6)


BEATS = {
    "tools": beats_tools,
    "parse": beats_parse,
    "validate": beats_validate,
    "execute": beats_execute,
}


async def main(segment: str, url: str) -> None:
    import websockets

    if segment not in BEATS:
        raise SystemExit(f"unknown segment {segment!r}; expected {sorted(BEATS)}")

    outdir = HERE / "frames" / segment
    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.glob("*.png"):
        old.unlink()

    # Drop a leftover debug Chrome so this take owns DEBUG_PORT.
    subprocess.run(
        ["pkill", "-f", f"--remote-debugging-port={DEBUG_PORT}"],
        capture_output=True,
    )
    time.sleep(0.4)

    profile = HERE / "chrome-profile"
    profile.mkdir(exist_ok=True)
    chrome = subprocess.Popen(
        [
            CHROME,
            f"--remote-debugging-port={DEBUG_PORT}",
            "--headless=new",
            "--user-data-dir=" + str(profile),
            "--window-size=1280,800",
            "--hide-scrollbars",
            "--no-sandbox",
            "--force-device-scale-factor=2",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            "--enable-unsafe-swiftshader",
            "about:blank",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(3)
        tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{DEBUG_PORT}/json").read())
        ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=40 * 1024 * 1024) as ws:
            await cdp(ws, 1, "Page.enable")
            await cdp(ws, 3, "Page.navigate", {"url": url})
            await asyncio.sleep(4)
            subprocess.run(
                ["tmux", "refresh-client", "-t", SESSION], capture_output=True
            )
            await asyncio.sleep(1)
            await preflight(ws)

            stop = asyncio.Event()
            counter = [0]
            task = asyncio.create_task(shoot(ws, outdir, stop, counter))
            await BEATS[segment]()
            stop.set()
            await task
            print(f"{segment}: {counter[0]} frames -> {outdir}")
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=5)
        except Exception:
            chrome.kill()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: film.py <tools|parse|validate|execute> <ttyd-url>")
    asyncio.run(main(sys.argv[1], sys.argv[2]))

"""Background live capture for the console: packets -> flows -> a sliding window of recent traffic.

The console polls :meth:`LiveSession.collect` every ``window_s`` seconds and
runs the detector on what it returns. Capturing needs scapy and raw-socket
rights: root (``sudo``) on Linux/macOS, an Administrator terminal plus Npcap
on Windows. :func:`capture_status` reports what is missing in plain words.
"""
from __future__ import annotations

import platform
import threading
import time
import uuid

import numpy as np
import pandas as pd

from .flowmeter import FlowMeter, Packet, scapy_to_packet


def capture_status() -> dict:
    """Can this machine capture packets? Returns {available, reason, interfaces, default}."""
    try:
        from scapy.all import conf  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - depends on the machine
        return {"available": False, "reason": f"scapy is not installed ({exc}). Run: pip install scapy",
                "interfaces": [], "default": None}
    interfaces = []
    try:
        for i in conf.ifaces.values():
            name = str(getattr(i, "name", "") or getattr(i, "network_name", ""))
            if not name:
                continue
            interfaces.append({"name": name, "description": str(getattr(i, "description", "") or name),
                               "ip": str(getattr(i, "ip", "") or "")})
    except Exception:  # pragma: no cover
        pass
    default = str(getattr(conf.iface, "name", conf.iface)) if conf.iface else None
    try:
        sock = conf.L2listen(iface=conf.iface)
        sock.close()
    except PermissionError:
        hint = ("run the terminal as Administrator" if platform.system() == "Windows"
                else "start the app with sudo, e.g. sudo .venv/bin/streamlit run app/streamlit_app.py")
        return {"available": False, "reason": f"Capturing packets needs administrator rights: {hint}.",
                "interfaces": interfaces, "default": default}
    except Exception as exc:  # pragma: no cover - e.g. missing Npcap on Windows
        hint = " Install Npcap from npcap.com (tick 'WinPcap API-compatible mode')." if platform.system() == "Windows" else ""
        return {"available": False, "reason": f"Packet capture is not available here: {exc}.{hint}",
                "interfaces": interfaces, "default": default}
    return {"available": True, "reason": "", "interfaces": interfaces, "default": default}


class LiveSession:
    """Thread-safe flow assembly over a sliding window of ``keep_windows`` x ``window_s`` seconds."""

    def __init__(self, iface: str | None = None, window_s: float = 20.0, keep_windows: int = 15):
        self.id = uuid.uuid4().hex[:10]
        self.iface = iface
        self.window_s = window_s
        self.keep_windows = keep_windows
        self.t0 = time.time()
        self.meter = FlowMeter(idle_timeout=min(5.0, window_s / 2), active_timeout=window_s)
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.packets = 0
        self.sniffer = None
        self.polls = 0

    # -------------------------------------------------------------- packets in
    def feed(self, p: Packet) -> None:
        """Add one packet (timestamp in epoch seconds)."""
        p.ts = p.ts - self.t0
        with self.lock:
            self.meter.add(p)
            self.packets += 1

    def _on_packet(self, pkt) -> None:
        p = scapy_to_packet(pkt)
        if p is not None:
            self.feed(p)

    def start(self) -> None:
        from scapy.all import AsyncSniffer  # noqa: WPS433

        self.sniffer = AsyncSniffer(iface=self.iface or None, prn=self._on_packet, store=False)
        self.sniffer.start()
        time.sleep(0.4)
        thread = getattr(self.sniffer, "thread", None)
        if thread is not None and not thread.is_alive():
            raise RuntimeError("the capture thread stopped immediately - check permissions and the interface name")

    def stop(self) -> None:
        if self.sniffer is not None:
            try:
                self.sniffer.stop()
            except Exception:  # already stopped
                pass
            self.sniffer = None

    @property
    def running(self) -> bool:
        return self.sniffer is not None

    # -------------------------------------------------------------- flows out
    def elapsed(self, now: float | None = None) -> float:
        return (now or time.time()) - self.t0

    def origin(self, now: float | None = None) -> float:
        """Start of the oldest window still kept (aligned to the window grid)."""
        current = np.floor(self.elapsed(now) / self.window_s) * self.window_s
        return float(max(0.0, current - (self.keep_windows - 1) * self.window_s))

    def collect(self, now: float | None = None) -> pd.DataFrame:
        """Export finished flows, drop ones older than the kept windows, return the recent traffic."""
        now = now or time.time()
        with self.lock:
            self.rows.extend(self.meter.sweep(now - self.t0))
            cutoff = self.origin(now)
            self.rows = [r for r in self.rows if r["ts"] >= cutoff]
            rows = list(self.rows)
            self.polls += 1
        df = pd.DataFrame(rows)
        if not df.empty:
            df["segment"] = "live"
            df = df.sort_values("ts", kind="stable").reset_index(drop=True)
        return df

    def info(self, now: float | None = None) -> dict:
        return {"id": self.id, "iface": self.iface or "default interface", "elapsed": round(self.elapsed(now), 1),
                "packets": self.packets, "flows": len(self.rows), "window_s": self.window_s,
                "keep_s": self.window_s * self.keep_windows}

"""Minimal bidirectional flow meter (packets -> canonical flow records).

Works on any packet source: live sniffing with scapy, pcap files, or plain
Python tuples in tests. Mirrors the CICFlowMeter conventions used by the
training data: the first packet defines the forward direction, flag counts
are the number of packets carrying the flag, flows end on FIN/RST, idle
timeout or active timeout.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

FIN, SYN, RST, PSH, ACK = 0x01, 0x02, 0x04, 0x08, 0x10
SERVICE_PORTS = {20, 21, 22, 23, 25, 53, 67, 68, 80, 110, 123, 137, 138, 139, 143, 161, 389, 443, 445, 465, 587, 993, 995,
                 1883, 1900, 3306, 3389, 5353, 5355, 5432, 8080, 8443}


def _is_service(port: int) -> bool:
    return port in SERVICE_PORTS or port < 1024


@dataclass
class Packet:
    ts: float
    src: str
    dst: str
    sport: int
    dport: int
    proto: int
    length: int
    flags: int = 0   # TCP flag bitmask


@dataclass
class _Flow:
    key: tuple
    first: float
    last: float
    fwd_pkts: int = 0
    bwd_pkts: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    syn: int = 0
    ack: int = 0
    fin: int = 0
    rst: int = 0
    psh: int = 0
    gap_sum: float = 0.0
    closed: bool = False
    _fin_seen: set = field(default_factory=set)

    def row(self) -> dict:
        src, dst, sport, dport, proto = self.key
        n = self.fwd_pkts + self.bwd_pkts
        return {"ts": self.first, "src_ip": src, "dst_ip": dst, "src_port": sport, "dst_port": dport,
                "protocol": proto, "duration": self.last - self.first, "fwd_pkts": self.fwd_pkts,
                "bwd_pkts": self.bwd_pkts, "fwd_bytes": self.fwd_bytes, "bwd_bytes": self.bwd_bytes,
                "syn": self.syn, "ack": self.ack, "fin": self.fin, "rst": self.rst, "psh": self.psh,
                "iat_mean": self.gap_sum / max(n - 1, 1), "label": "UNKNOWN"}


class FlowMeter:
    def __init__(self, idle_timeout: float = 5.0, active_timeout: float = 60.0):
        self.idle_timeout = idle_timeout
        self.active_timeout = active_timeout
        self.flows: dict[tuple, _Flow] = {}
        self.done: list[dict] = []

    def add(self, p: Packet) -> None:
        key = (p.src, p.dst, p.sport, p.dport, p.proto)
        rkey = (p.dst, p.src, p.dport, p.sport, p.proto)
        if key in self.flows:
            f, fwd = self.flows[key], True
        elif rkey in self.flows:
            f, fwd = self.flows[rkey], False
        else:
            f, fwd = None, True
        if f is not None and (p.ts - f.last > self.idle_timeout or p.ts - f.first > self.active_timeout or f.closed):
            self._export(f)
            f = None
            fwd = True
        if f is None:
            # A capture can start mid-connection: if this first packet comes *from* a service port to an
            # ephemeral port, it is the server's reply, so orient the flow client -> server.
            if _is_service(p.sport) and not _is_service(p.dport) and not (p.flags & SYN and not p.flags & ACK):
                key, fwd = rkey, False
            f = _Flow(key=key, first=p.ts, last=p.ts)
            self.flows[key] = f
        f.gap_sum += max(p.ts - f.last, 0.0)
        f.last = p.ts
        if fwd:
            f.fwd_pkts += 1
            f.fwd_bytes += p.length
        else:
            f.bwd_pkts += 1
            f.bwd_bytes += p.length
        if p.proto == 6:
            f.syn += bool(p.flags & SYN)
            f.ack += bool(p.flags & ACK)
            f.fin += bool(p.flags & FIN)
            f.rst += bool(p.flags & RST)
            f.psh += bool(p.flags & PSH)
            if p.flags & FIN:
                f._fin_seen.add(fwd)
            if p.flags & RST or len(f._fin_seen) == 2:
                f.closed = True

    def _export(self, f: _Flow) -> None:
        self.done.append(f.row())
        self.flows.pop(f.key, None)

    def sweep(self, now: float) -> list[dict]:
        """Export finished / timed-out flows and return everything exported since the last call."""
        for f in list(self.flows.values()):
            if f.closed or now - f.last > self.idle_timeout or now - f.first > self.active_timeout:
                self._export(f)
        out, self.done = self.done, []
        return out

    def flush(self) -> list[dict]:
        for f in list(self.flows.values()):
            self._export(f)
        out, self.done = self.done, []
        return out


def scapy_to_packet(pkt) -> Packet | None:
    """Convert a scapy packet to :class:`Packet` (IPv4 TCP/UDP/ICMP only)."""
    from scapy.layers.inet import ICMP, IP, TCP, UDP  # noqa: WPS433

    if IP not in pkt:
        return None
    ip = pkt[IP]
    if TCP in pkt:
        t = pkt[TCP]
        return Packet(float(pkt.time), ip.src, ip.dst, int(t.sport), int(t.dport), 6, int(ip.len), int(t.flags))
    if UDP in pkt:
        u = pkt[UDP]
        return Packet(float(pkt.time), ip.src, ip.dst, int(u.sport), int(u.dport), 17, int(ip.len))
    if ICMP in pkt:
        return Packet(float(pkt.time), ip.src, ip.dst, 0, 0, 1, int(ip.len))
    return None


def pcap_to_flows(path: str, idle_timeout: float = 5.0, active_timeout: float = 60.0,
                  max_packets: int | None = None) -> pd.DataFrame:
    """Read a pcap/pcapng file and return canonical flow records (needs scapy)."""
    from scapy.utils import PcapReader  # noqa: WPS433

    meter = FlowMeter(idle_timeout, active_timeout)
    t0 = None
    with PcapReader(str(path)) as reader:
        for i, pkt in enumerate(reader):
            if max_packets and i >= max_packets:
                break
            p = scapy_to_packet(pkt)
            if p is None:
                continue
            t0 = p.ts if t0 is None else t0
            meter.add(p)
    df = pd.DataFrame(meter.flush())
    if df.empty:
        return df
    df["ts"] = df["ts"] - (t0 or 0.0)
    df["segment"] = "pcap"
    return df.sort_values("ts").reset_index(drop=True)

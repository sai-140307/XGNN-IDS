"""Realistic synthetic network-flow generator with graph-structured attacks.

Why synthetic data at all? Public benchmarks (CIC-IDS2017, CICDDoS2019 ...)
are several GB and cannot be shipped with a demo or a CI pipeline, so the
project ships a generator that reproduces the *relational* signatures the
GNN is meant to learn, in the same canonical schema the real-dataset loaders
produce. Real datasets plug into exactly the same pipeline
(see `xgnn_ids.data.loaders`).

Design goals
------------
* Individual attack flows overlap with benign flows (an HTTP-flood request
  looks like a normal page visit; a brute-force login looks like an SSH
  session), so the *structure* of the window matters: many sources converging
  on one host (DDoS), one source fanning out over ports (scan) or hosts
  (lateral movement), periodic beacons from several hosts to one rare
  external address (botnet).
* Hard negatives: a benign "flash crowd" (many real visitors at once) and, in
  the IoT profile, periodic benign telemetry that resembles beaconing.
* Every attack family has episodes in the train, validation and test parts of
  the timeline, so chronological splitting is possible.
* Two profiles (enterprise, iot) with different addressing, services, benign
  mix and attack variants, used for cross-domain generalisation tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..config import ATTACK_CLASSES

# ---------------------------------------------------------------- helpers

_PUBLIC_FIRST_OCTETS = [3, 8, 13, 18, 23, 31, 34, 35, 45, 52, 54, 64, 66, 69, 72, 77,
                        81, 89, 91, 95, 98, 104, 109, 117, 121, 142, 151, 157, 162,
                        176, 178, 185, 188, 193, 195, 199, 203, 212, 217]


def _public_ips(rng: np.random.Generator, n: int) -> list[str]:
    out: set[str] = set()
    while len(out) < n:
        o1 = int(rng.choice(_PUBLIC_FIRST_OCTETS))
        out.add(f"{o1}.{rng.integers(0, 256)}.{rng.integers(0, 256)}.{rng.integers(1, 255)}")
    return sorted(out)


def _ln(rng: np.random.Generator, median: float, sigma: float, size: int) -> np.ndarray:
    """Log-normal samples parameterised by their median."""
    return median * np.exp(sigma * rng.standard_normal(size))


def _geom(rng: np.random.Generator, p: float, size: int) -> np.ndarray:
    return rng.geometric(p, size) - 1


def _zipf_choice(rng: np.random.Generator, items: list[str], size: int, a: float = 1.2) -> np.ndarray:
    ranks = np.arange(1, len(items) + 1)
    w = 1.0 / ranks**a
    return rng.choice(np.asarray(items, dtype=object), size=size, p=w / w.sum())


def _flows(
    rng: np.random.Generator,
    ts: np.ndarray,
    src,
    dst,
    dport,
    proto,
    duration: np.ndarray,
    fwd_pkts: np.ndarray,
    bwd_pkts: np.ndarray,
    fwd_len: np.ndarray,
    bwd_len: np.ndarray,
    flags: str,
    label: str,
    detail: str,
    psh_rate: float = 0.2,
    sport=None,
) -> pd.DataFrame:
    """Assemble flow records and derive TCP flag counts consistently."""
    n = len(ts)
    if n == 0:
        return pd.DataFrame()

    def _arr(v, dtype=object):
        if np.isscalar(v) or isinstance(v, str):
            return np.full(n, v, dtype=dtype)
        return np.asarray(v, dtype=dtype)

    def _vec(v) -> np.ndarray:
        return np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()

    duration, fwd_len, bwd_len = _vec(duration), _vec(fwd_len), _vec(bwd_len)
    fp = np.maximum(np.round(_vec(fwd_pkts)), 1).astype(int)
    bp = np.maximum(np.round(_vec(bwd_pkts)), 0).astype(int)
    proto_a = _arr(proto, int)
    total = fp + bp
    syn = np.zeros(n, int)
    ack = np.zeros(n, int)
    fin = np.zeros(n, int)
    rst = np.zeros(n, int)
    psh = np.zeros(n, int)
    tcp = proto_a == 6
    if flags == "normal":
        syn[:] = np.where(bp > 0, 2, 1)
        ack[:] = np.maximum(total - 1, 0)
        fin[:] = np.where(rng.random(n) < 0.9, np.minimum(2, total), 0)
        rst[:] = (rng.random(n) < 0.04).astype(int)
        psh[:] = rng.binomial(total, psh_rate)
    elif flags == "syn_only":       # half-open: SYN (+ optional SYN-ACK)
        syn[:] = fp + bp
        ack[:] = bp
        rst[:] = (rng.random(n) < 0.15).astype(int)
    elif flags == "syn_rst":        # closed port: SYN -> RST/ACK
        syn[:] = 1
        rst[:] = bp
        ack[:] = bp
    elif flags == "slow":           # slow HTTP: no FIN, many PSH
        syn[:] = np.where(bp > 0, 2, 1)
        ack[:] = np.maximum(total - 1, 0)
        psh[:] = rng.binomial(total, 0.8)
    for a in (syn, ack, fin, rst, psh):
        a[~tcp] = 0

    fb = fp * np.clip(fwd_len, 28, 1500)
    bb = bp * np.clip(bwd_len, 28, 1500)
    dur = np.maximum(duration, 0)
    iat = dur / np.maximum(total - 1, 1) * rng.uniform(0.7, 1.3, n)
    if sport is None:
        sport = rng.integers(1024, 65535, n)
    return pd.DataFrame(
        {
            "ts": ts.astype(float),
            "src_ip": _arr(src),
            "dst_ip": _arr(dst),
            "src_port": _arr(sport, int),
            "dst_port": _arr(dport, int),
            "protocol": proto_a,
            "duration": dur,
            "fwd_pkts": fp,
            "bwd_pkts": bp,
            "fwd_bytes": np.round(fb),
            "bwd_bytes": np.round(bb),
            "syn": syn,
            "ack": ack,
            "fin": fin,
            "rst": rst,
            "psh": psh,
            "iat_mean": iat,
            "label": label,
            "attack_detail": detail,
        }
    )


def _poisson_times(rng, rate: float, t0: float, t1: float) -> np.ndarray:
    n = rng.poisson(max(rate, 0) * max(t1 - t0, 0))
    return np.sort(rng.uniform(t0, t1, n))


# ---------------------------------------------------------------- profiles


@dataclass
class Profile:
    name: str
    servers: dict[str, str]
    clients: list[str]
    admins: list[str]
    external: list[str]           # benign internet destinations
    visitors: list[str]           # benign external clients of our services
    attackers: list[str]
    web_target: tuple[str, int]
    auth_targets: list[tuple[str, int]]
    scan_targets: list[str]
    c2_ports: list[int]
    lateral_ports: list[int]
    ddos_udp_frac: float
    benign: list[tuple[Callable, float]] = field(default_factory=list)
    flash_crowds: int = 0


def _enterprise(rng) -> Profile:
    net = "192.168.10."
    servers = {"web": net + "50", "ssh": net + "51", "ftp": net + "52", "dns": net + "53",
               "db": net + "54", "file": net + "55", "mail": net + "56"}
    clients = [net + str(i) for i in range(100, 180)]
    p = Profile(
        name="enterprise",
        servers=servers,
        clients=clients,
        admins=clients[:5],
        external=_public_ips(rng, 400),
        visitors=_public_ips(rng, 250),
        attackers=_public_ips(rng, 4),
        web_target=(servers["web"], 80),
        auth_targets=[(servers["ssh"], 22), (servers["ftp"], 21)],
        scan_targets=list(servers.values()) + clients[:20],
        c2_ports=[8080, 6667, 443, 4444],
        lateral_ports=[445, 139, 3389, 22, 135],
        ddos_udp_frac=0.1,
        flash_crowds=2,
    )
    p.benign = [
        (_b_web_out, 5.0), (_b_dns, 3.0), (_b_visitors, 2.0), (_b_intranet, 1.0),
        (_b_db, 1.0), (_b_smb, 0.4), (_b_ssh, 0.05), (_b_ntp, 0.2), (_b_mail, 0.3),
    ]
    return p


def _iot(rng) -> Profile:
    servers = {"web": "10.20.0.10", "broker": "10.20.0.1", "dns": "10.20.0.2",
               "ssh": "10.20.0.5", "ftp": "10.20.0.6"}
    cameras = [f"10.20.0.{i}" for i in range(10, 20)]
    devices = [f"10.20.1.{i}" for i in range(1, 151)]
    p = Profile(
        name="iot",
        servers={**servers, **{f"cam{i}": c for i, c in enumerate(cameras)}},
        clients=devices,
        admins=["10.20.0.100", "10.20.0.101"],
        external=_public_ips(rng, 50),
        visitors=_public_ips(rng, 80),
        attackers=_public_ips(rng, 4),
        web_target=(cameras[0], 80),
        auth_targets=[(servers["ssh"], 23), (servers["ssh"], 22)],
        scan_targets=cameras + devices[:30],
        c2_ports=[23, 48101, 6667],
        lateral_ports=[23, 22, 80, 1883],
        ddos_udp_frac=0.5,
        flash_crowds=0,
    )
    p.benign = [
        (_b_iot_telemetry, 0.0), (_b_iot_cloud, 2.0), (_b_dns, 0.8), (_b_ntp, 0.4),
        (_b_iot_app, 1.0), (_b_iot_rtsp, 0.1),
    ]
    return p


def _home(rng) -> Profile:
    """A home / small-office Wi-Fi network as a laptop would see it: one router, a few laptops and phones,
    a smart TV, a NAS, and very bursty browsing to hundreds of CDN addresses."""
    net = "192.168.1."
    router, nas = net + "1", net + "20"
    laptops = [net + str(i) for i in range(10, 14)]
    phones = [net + str(i) for i in range(30, 36)]
    others = [net + "40", net + "50", net + "51", net + "52"]          # TV, smart plugs, speaker
    p = Profile(
        name="home",
        servers={"web": nas, "dns": router, "router": router, "nas": nas, "ssh": nas, "file": nas},
        clients=laptops + phones + others,
        admins=laptops[:1],
        external=_public_ips(rng, 700),
        visitors=_public_ips(rng, 8),
        attackers=_public_ips(rng, 4),
        web_target=(nas, 80),
        auth_targets=[(nas, 22), (router, 23)],
        scan_targets=laptops + [nas, router],
        c2_ports=[443, 8443, 6667],
        lateral_ports=[445, 22, 3389, 139],
        ddos_udp_frac=0.3,
    )
    p.laptops, p.phones, p.tv = laptops, phones, others[0]
    p.benign = [
        (_b_home_browsing, 0.0), (_b_home_quic, 1.2), (_b_home_dns, 4.0), (_b_home_discovery, 0.0),
        (_b_home_streaming, 0.03), (_b_home_background, 0.0), (_b_home_failed, 0.4), (_b_ntp, 0.1),
        (_b_home_nas, 0.15),
    ]
    return p


PROFILES = {"enterprise": _enterprise, "iot": _iot, "home": _home}

# ---------------------------------------------------------------- benign traffic


def _b_web_out(rng, ts, P: Profile):
    n = len(ts)
    fp = 2 + _geom(rng, 0.25, n)
    return _flows(rng, ts, rng.choice(P.clients, n), _zipf_choice(rng, P.external, n),
                  np.where(rng.random(n) < 0.85, 443, 80), 6,
                  np.minimum(_ln(rng, 2.0, 1.2, n), 120), fp, fp * rng.uniform(0.8, 2.5, n),
                  _ln(rng, 180, 0.5, n), _ln(rng, 900, 0.5, n), "normal", "BENIGN", "web-browsing", 0.3)


def _b_dns(rng, ts, P: Profile):
    n = len(ts)
    fp = np.where(rng.random(n) < 0.9, 1, 2)
    to_ext = rng.random(n) < 0.25
    src = np.where(to_ext, P.servers["dns"], rng.choice(P.clients, n))
    dst = np.where(to_ext, rng.choice(["8.8.8.8", "1.1.1.1", "9.9.9.9"], n), P.servers["dns"])
    return _flows(rng, ts, src, dst, 53, 17, _ln(rng, 0.03, 0.8, n), fp, fp,
                  rng.uniform(60, 90, n), rng.uniform(90, 320, n), "udp", "BENIGN", "dns")


def _b_visitors(rng, ts, P: Profile):
    n = len(ts)
    fp = 3 + _geom(rng, 0.3, n)
    return _flows(rng, ts, _zipf_choice(rng, P.visitors, n, 0.8), P.web_target[0],
                  np.where(rng.random(n) < 0.7, 443, 80), 6, _ln(rng, 1.5, 1.0, n),
                  fp, fp * rng.uniform(1.0, 3.0, n), _ln(rng, 250, 0.4, n), _ln(rng, 1100, 0.4, n),
                  "normal", "BENIGN", "web-visitors", 0.3)


def _b_intranet(rng, ts, P: Profile):
    n = len(ts)
    fp = 3 + _geom(rng, 0.3, n)
    return _flows(rng, ts, rng.choice(P.clients, n), P.web_target[0], np.where(rng.random(n) < 0.5, 443, 80),
                  6, _ln(rng, 1.0, 1.0, n), fp, fp * rng.uniform(1.0, 3.0, n),
                  _ln(rng, 230, 0.4, n), _ln(rng, 1000, 0.5, n), "normal", "BENIGN", "intranet")


def _b_db(rng, ts, P: Profile):
    n = len(ts)
    fp = 4 + _geom(rng, 0.3, n)
    return _flows(rng, ts, P.servers["web"], P.servers["db"], 3306, 6, _ln(rng, 0.2, 0.8, n),
                  fp, fp * rng.uniform(0.8, 1.5, n), _ln(rng, 200, 0.6, n), _ln(rng, 600, 0.8, n),
                  "normal", "BENIGN", "database")


def _b_smb(rng, ts, P: Profile):
    n = len(ts)
    fp = np.round(_ln(rng, 40, 1.0, n))
    return _flows(rng, ts, rng.choice(P.clients, n), P.servers["file"], 445, 6, _ln(rng, 5.0, 1.2, n),
                  fp, fp * rng.uniform(1.0, 2.0, n), _ln(rng, 300, 0.6, n), _ln(rng, 1200, 0.3, n),
                  "normal", "BENIGN", "file-share")


def _b_ssh(rng, ts, P: Profile):
    n = len(ts)
    fp = np.round(_ln(rng, 150, 0.8, n))
    return _flows(rng, ts, rng.choice(P.admins, n), P.auth_targets[0][0], P.auth_targets[0][1], 6,
                  np.minimum(_ln(rng, 60, 1.0, n), 900), fp, fp * rng.uniform(0.8, 1.3, n),
                  _ln(rng, 90, 0.3, n), _ln(rng, 150, 0.5, n), "normal", "BENIGN", "admin-ssh", 0.6)


def _b_ntp(rng, ts, P: Profile):
    n = len(ts)
    return _flows(rng, ts, rng.choice(P.clients, n), rng.choice(["129.6.15.28", "162.159.200.1"], n), 123, 17,
                  _ln(rng, 0.05, 0.5, n), 1, 1, 76, 76, "udp", "BENIGN", "ntp")


def _b_mail(rng, ts, P: Profile):
    n = len(ts)
    fp = 6 + _geom(rng, 0.2, n)
    return _flows(rng, ts, rng.choice(P.clients, n), P.servers.get("mail", P.servers["web"]),
                  rng.choice([587, 993], n), 6, _ln(rng, 3.0, 1.0, n), fp, fp * rng.uniform(0.8, 2.0, n),
                  _ln(rng, 400, 0.6, n), _ln(rng, 700, 0.6, n), "normal", "BENIGN", "mail")


def _b_iot_telemetry(rng, ts, P: Profile, t0: float = 0.0, t1: float = 0.0):
    """Periodic MQTT telemetry: every device publishes every 10-30 s (beacon-like)."""
    parts = []
    for dev in P.clients:
        period = rng.uniform(10, 30)
        times = np.arange(t0 + rng.uniform(0, period), t1, period)
        times = times + rng.normal(0, 0.2, len(times))
        n = len(times)
        if n == 0:
            continue
        fp = 2 + _geom(rng, 0.6, n)
        parts.append(_flows(rng, times, dev, P.servers["broker"], 1883, 6, _ln(rng, 0.08, 0.5, n), fp,
                            fp, _ln(rng, 110, 0.2, n), _ln(rng, 60, 0.2, n), "normal", "BENIGN",
                            "mqtt-telemetry"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _b_iot_cloud(rng, ts, P: Profile):
    n = len(ts)
    fp = 3 + _geom(rng, 0.4, n)
    return _flows(rng, ts, rng.choice(P.clients, n), _zipf_choice(rng, P.external, n), 443, 6,
                  _ln(rng, 0.8, 0.8, n), fp, fp * rng.uniform(0.8, 1.6, n), _ln(rng, 200, 0.5, n),
                  _ln(rng, 400, 0.6, n), "normal", "BENIGN", "iot-cloud")


def _b_iot_app(rng, ts, P: Profile):
    n = len(ts)
    cams = [v for k, v in P.servers.items() if k.startswith("cam")]
    fp = 4 + _geom(rng, 0.2, n)
    return _flows(rng, ts, rng.choice(P.visitors, n), rng.choice(cams, n), np.where(rng.random(n) < 0.8, 443, 80),
                  6, _ln(rng, 4.0, 1.0, n), fp, fp * rng.uniform(2, 6, n), _ln(rng, 150, 0.4, n),
                  _ln(rng, 1300, 0.2, n), "normal", "BENIGN", "camera-app")


def _b_iot_rtsp(rng, ts, P: Profile):
    n = len(ts)
    cams = [v for k, v in P.servers.items() if k.startswith("cam")]
    fp = np.round(_ln(rng, 300, 0.5, n))
    return _flows(rng, ts, rng.choice(P.admins, n), rng.choice(cams, n), 554, 6, _ln(rng, 60, 0.6, n), fp,
                  fp * rng.uniform(5, 10, n), _ln(rng, 80, 0.2, n), _ln(rng, 1350, 0.1, n), "normal",
                  "BENIGN", "rtsp-stream")


def _b_flash_crowd(rng, t0, length, P: Profile):
    """Benign hard negative: a burst of real visitors (e.g. after a marketing email)."""
    ts = _poisson_times(rng, rng.uniform(25, 45), t0, t0 + length)
    n = len(ts)
    crowd = _public_ips(rng, int(rng.integers(120, 250)))
    fp = 3 + _geom(rng, 0.3, n)
    return _flows(rng, ts, rng.choice(crowd, n), P.web_target[0], np.where(rng.random(n) < 0.7, 443, 80), 6,
                  _ln(rng, 1.5, 1.0, n), fp, fp * rng.uniform(1.0, 3.0, n), _ln(rng, 250, 0.4, n),
                  _ln(rng, 1100, 0.4, n), "normal", "BENIGN", "flash-crowd", 0.3)


# ---------------------------------------------------------------- home / laptop traffic

def _b_home_browsing(rng, ts, P: Profile, t0: float = 0.0, t1: float = 0.0):
    """Page loads: one device opens 5-60 HTTPS connections within ~2 s to 3-25 CDN / tracker addresses."""
    parts = []
    loads = _poisson_times(rng, 0.35, t0, t1)
    devices = P.laptops * 3 + P.phones
    for t in loads:
        dev = rng.choice(devices)
        k = int(rng.integers(5, 60))
        dests = _zipf_choice(rng, P.external, int(rng.integers(3, 26)), 1.05)
        times = np.sort(t + rng.exponential(0.5, k))
        fp = 3 + _geom(rng, 0.12, k)
        parts.append(_flows(rng, times, dev, rng.choice(dests, k), np.where(rng.random(k) < 0.9, 443, 80), 6,
                            np.minimum(_ln(rng, 3.0, 1.4, k), 300), fp, fp * rng.uniform(1.0, 4.0, k),
                            _ln(rng, 160, 0.5, k), _ln(rng, 1100, 0.4, k), "normal", "BENIGN", "home-browsing", 0.35))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _b_home_quic(rng, ts, P: Profile):
    n = len(ts)
    fp = np.round(_ln(rng, 25, 1.2, n)) + 2
    return _flows(rng, ts, rng.choice(P.laptops + P.phones, n), _zipf_choice(rng, P.external, n, 1.2), 443, 17,
                  np.minimum(_ln(rng, 6.0, 1.3, n), 300), fp, fp * rng.uniform(1.0, 6.0, n), _ln(rng, 300, 0.6, n),
                  _ln(rng, 1150, 0.2, n), "udp", "BENIGN", "home-quic")


def _b_home_dns(rng, ts, P: Profile):
    n = len(ts)
    fp = np.where(rng.random(n) < 0.85, 1, 2)
    return _flows(rng, ts, rng.choice(P.clients, n), P.servers["router"], 53, 17, _ln(rng, 0.02, 0.8, n), fp, fp,
                  rng.uniform(55, 95, n), rng.uniform(90, 400, n), "udp", "BENIGN", "home-dns")


def _b_home_discovery(rng, ts, P: Profile, t0: float = 0.0, t1: float = 0.0):
    """mDNS / SSDP / LLMNR multicast announcements: periodic, never answered by the multicast address."""
    parts = []
    targets = [("224.0.0.251", 5353), ("239.255.255.250", 1900), ("224.0.0.252", 5355)]
    for dev in P.clients:
        for addr, port in targets:
            if rng.random() < 0.4:
                continue
            period = rng.uniform(20, 120)
            times = np.arange(t0 + rng.uniform(0, period), t1, period)
            n = len(times)
            if n:
                parts.append(_flows(rng, times, dev, addr, port, 17, _ln(rng, 0.5, 1.0, n), rng.integers(1, 4, n), 0,
                                    rng.uniform(80, 400, n), 0, "udp", "BENIGN", "home-discovery"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _b_home_streaming(rng, ts, P: Profile):
    n = len(ts)
    fp = np.round(_ln(rng, 800, 0.6, n))
    return _flows(rng, ts, P.tv, _zipf_choice(rng, P.external[:40], n), 443, 6, _ln(rng, 120, 0.5, n), fp,
                  fp * rng.uniform(3, 8, n), _ln(rng, 80, 0.2, n), _ln(rng, 1380, 0.05, n), "normal", "BENIGN",
                  "home-streaming", 0.2)


def _b_home_background(rng, ts, P: Profile, t0: float = 0.0, t1: float = 0.0):
    """Periodic app / OS telemetry: every device checks in with a few services every 30-300 s (beacon-like)."""
    parts = []
    for dev in P.clients:
        for dst in _zipf_choice(rng, P.external[:60], int(rng.integers(1, 5))):
            period = rng.uniform(30, 300)
            times = np.arange(t0 + rng.uniform(0, period), t1, period) + rng.normal(0, 1.0)
            n = len(times)
            if n:
                fp = 3 + _geom(rng, 0.5, n)
                parts.append(_flows(rng, times, dev, dst, 443, 6, _ln(rng, 0.6, 0.6, n), fp, fp * rng.uniform(1, 2, n),
                                    _ln(rng, 250, 0.4, n), _ln(rng, 400, 0.5, n), "normal", "BENIGN", "home-telemetry"))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _b_home_failed(rng, ts, P: Profile):
    """Connections that never complete (blocked trackers, offline hosts, IPv4 fallbacks): SYN with no reply or a RST."""
    n = len(ts)
    rst = rng.random(n) < 0.4
    return _flows(rng, ts, rng.choice(P.laptops + P.phones, n), rng.choice(P.external, n),
                  np.where(rng.random(n) < 0.85, 443, 80), 6, _ln(rng, 2.0, 0.8, n), rng.integers(1, 4, n),
                  rst.astype(int), 60, 40, "syn_only", "BENIGN", "home-failed-connection")


def _b_home_nas(rng, ts, P: Profile):
    n = len(ts)
    fp = np.round(_ln(rng, 60, 1.0, n))
    return _flows(rng, ts, rng.choice(P.laptops, n), P.servers["nas"], rng.choice([445, 80, 443], n), 6,
                  _ln(rng, 4.0, 1.0, n), fp, fp * rng.uniform(1, 3, n), _ln(rng, 300, 0.6, n), _ln(rng, 1200, 0.3, n),
                  "normal", "BENIGN", "home-nas")


# ---------------------------------------------------------------- attacks


def _a_ddos(rng, t0, length, P: Profile, intensity: float):
    rate = rng.uniform(50, 120) * intensity
    ts = _poisson_times(rng, rate, t0, t0 + length)
    n = len(ts)
    bots = _public_ips(rng, int(rng.integers(150, 400)))
    src = rng.choice(bots, n)
    target, port = P.web_target
    kind = rng.random(n)
    syn_m = kind < (1 - P.ddos_udp_frac) * 0.5
    udp_m = kind >= 1 - P.ddos_udp_frac
    http_m = ~(syn_m | udp_m)
    parts = []
    k = syn_m.sum()
    if k:
        fp = rng.integers(1, 4, k)
        bp = (rng.random(k) < 0.7).astype(int)
        parts.append(_flows(rng, ts[syn_m], src[syn_m], target, port, 6, _ln(rng, 0.01, 1.5, k), fp, bp,
                            rng.uniform(40, 60, k), rng.uniform(44, 60, k), "syn_only", "DDoS", "DDoS-SYN-flood"))
    k = http_m.sum()
    if k:
        fp = 3 + _geom(rng, 0.5, k)
        parts.append(_flows(rng, ts[http_m], src[http_m], target, np.where(rng.random(k) < 0.8, port, 443), 6,
                            _ln(rng, 0.4, 0.8, k), fp, 2 + _geom(rng, 0.4, k), _ln(rng, 220, 0.3, k),
                            _ln(rng, 600, 0.5, k), "normal", "DDoS", "DDoS-HTTP-flood", 0.3))
    k = udp_m.sum()
    if k:
        parts.append(_flows(rng, ts[udp_m], src[udp_m], target, rng.integers(1, 65535, k), 17,
                            _ln(rng, 0.5, 1.0, k), rng.integers(1, 20, k), 0, rng.uniform(500, 1400, k), 0,
                            "udp", "DDoS", "DDoS-UDP-flood"))
    return pd.concat(parts, ignore_index=True)


def _a_dos(rng, t0, length, P: Profile, intensity: float):
    attacker = rng.choice(P.attackers)
    target, port = P.web_target
    kind = rng.choice(["hulk", "slowloris", "goldeneye"])
    if kind == "slowloris":
        ts = _poisson_times(rng, rng.uniform(4, 10) * intensity, t0, t0 + length)
        n = len(ts)
        return _flows(rng, ts, attacker, target, port, 6, _ln(rng, 40, 0.5, n), np.round(_ln(rng, 8, 0.4, n)),
                      rng.integers(0, 3, n), rng.uniform(60, 120, n), 60, "slow", "DoS", "DoS-Slowloris")
    ts = _poisson_times(rng, rng.uniform(25, 50) * intensity, t0, t0 + length)
    n = len(ts)
    fp = 4 + _geom(rng, 0.4, n)
    sport = (int(rng.integers(20000, 40000)) + np.arange(n)) % 64511 + 1024  # sequential source ports
    return _flows(rng, ts, attacker, target, port, 6, _ln(rng, 0.5, 0.8, n), fp, 3 + _geom(rng, 0.4, n),
                  _ln(rng, 300, 0.3, n), _ln(rng, 400, 0.6, n), "normal", "DoS", f"DoS-{kind.capitalize()}",
                  0.4, sport=sport)


def _a_portscan(rng, t0, length, P: Profile, intensity: float):
    attacker = rng.choice(P.attackers)
    target = rng.choice(P.scan_targets)
    n_ports = int(rng.integers(400, 1200) * intensity)
    ports = np.concatenate([rng.permutation(np.arange(1, 1025)), rng.integers(1025, 10000, n_ports)])[:n_ports]
    scan_len = min(length, rng.uniform(5, 25))
    ts = np.sort(rng.uniform(t0, t0 + scan_len, n_ports))
    open_ports = {22, 23, 53, 80, 443, 445, 1883, 3306, 3389, 554}
    is_open = np.isin(ports, list(open_ports))
    filtered = (~is_open) & (rng.random(n_ports) < 0.2)
    closed = ~(is_open | filtered)
    udp_scan = rng.random() < 0.15
    parts = []
    if udp_scan:
        return _flows(rng, ts, attacker, target, ports, 17, _ln(rng, 0.0005, 0.5, n_ports), 1,
                      (rng.random(n_ports) < 0.3).astype(int), rng.uniform(28, 40, n_ports), 56, "udp",
                      "PortScan", "PortScan-UDP")
    for mask, flags, bp, detail in ((closed, "syn_rst", 1, "closed"), (filtered, "syn_only", 0, "filtered"),
                                    (is_open, "normal", 1, "open")):
        k = mask.sum()
        if not k:
            continue
        parts.append(_flows(rng, ts[mask], attacker, target, ports[mask], 6, _ln(rng, 0.0004, 0.6, k),
                            rng.integers(1, 3, k) if flags != "syn_rst" else 1, bp, rng.uniform(40, 60, k),
                            rng.uniform(40, 60, k), flags, "PortScan", f"PortScan-{detail}"))
    return pd.concat(parts, ignore_index=True)


def _a_bruteforce(rng, t0, length, P: Profile, intensity: float):
    attacker = rng.choice(P.attackers)
    target, port = P.auth_targets[int(rng.integers(0, len(P.auth_targets)))]
    ts = _poisson_times(rng, rng.uniform(2, 6) * intensity, t0, t0 + length)
    n = len(ts)
    return _flows(rng, ts, attacker, target, port, 6, _ln(rng, 2.0, 0.25, n), np.round(_ln(rng, 18, 0.15, n)),
                  np.round(_ln(rng, 16, 0.15, n)), _ln(rng, 95, 0.1, n), _ln(rng, 110, 0.1, n), "normal",
                  "BruteForce", f"BruteForce-port{port}", 0.5)


def _a_botnet(rng, t0, length, P: Profile, intensity: float):
    bots = rng.choice(P.clients, int(rng.integers(3, 9)), replace=False)
    c2 = _public_ips(rng, 1)[0]
    c2_port = int(rng.choice(P.c2_ports))
    parts = []
    for bot in bots:
        period = rng.uniform(5, 12) / intensity
        times = np.arange(t0 + rng.uniform(0, period), t0 + length, period)
        times = times + rng.normal(0, 0.3, len(times))
        n = len(times)
        if n == 0:
            continue
        fp = 3 + _geom(rng, 0.6, n)
        parts.append(_flows(rng, times, bot, c2, c2_port, 6, _ln(rng, 0.3, 0.5, n), fp, 2 + _geom(rng, 0.6, n),
                            _ln(rng, 150, 0.2, n), _ln(rng, 200, 0.3, n), "normal", "Botnet", "Botnet-C2-beacon"))
    return pd.concat(parts, ignore_index=True)


def _a_webattack(rng, t0, length, P: Profile, intensity: float):
    attacker = rng.choice(P.attackers)
    target, port = P.web_target
    kind = rng.choice(["SQLi", "XSS", "WebBruteForce"])
    ts = _poisson_times(rng, rng.uniform(1, 4) * intensity, t0, t0 + length)
    n = len(ts)
    fwd_len = _ln(rng, 450, 0.15, n) if kind == "WebBruteForce" else _ln(rng, 700, 0.4, n)
    return _flows(rng, ts, attacker, target, port, 6, _ln(rng, 0.8, 0.6, n), 3 + _geom(rng, 0.5, n),
                  2 + _geom(rng, 0.5, n), fwd_len, _ln(rng, 500, 0.5, n), "normal", "WebAttack",
                  f"WebAttack-{kind}", 0.8)


def _a_infiltration(rng, t0, length, P: Profile, intensity: float):
    victim = rng.choice(P.clients)
    internal = [h for h in P.clients + list(P.servers.values()) if h != victim]
    ts = _poisson_times(rng, rng.uniform(8, 25) * intensity, t0, t0 + length)
    n = len(ts)
    ok = rng.random(n) < 0.25
    parts = [
        _flows(rng, ts[~ok], victim, rng.choice(internal, (~ok).sum()), rng.choice(P.lateral_ports, (~ok).sum()), 6,
               _ln(rng, 0.001, 0.8, (~ok).sum()), 1, (rng.random((~ok).sum()) < 0.5).astype(int), 60, 40, "syn_rst",
               "Infiltration", "Infiltration-lateral-probe"),
        _flows(rng, ts[ok], victim, rng.choice(internal, ok.sum()), rng.choice(P.lateral_ports, ok.sum()), 6,
               _ln(rng, 0.5, 0.8, ok.sum()), 4 + _geom(rng, 0.4, ok.sum()), 3 + _geom(rng, 0.4, ok.sum()),
               _ln(rng, 200, 0.4, ok.sum()), _ln(rng, 300, 0.4, ok.sum()), "normal", "Infiltration",
               "Infiltration-lateral-session"),
    ]
    k = int(rng.integers(2, 6))
    ex_ts = np.sort(rng.uniform(t0, t0 + length, k))
    parts.append(_flows(rng, ex_ts, victim, _public_ips(rng, 1)[0], 443, 6, _ln(rng, 20, 0.5, k),
                        np.round(_ln(rng, 800, 0.3, k)), np.round(_ln(rng, 200, 0.3, k)), 1300, 60, "normal",
                        "Infiltration", "Infiltration-exfiltration", 0.6))
    return pd.concat(parts, ignore_index=True)


ATTACKS: dict[str, tuple[Callable, tuple[float, float]]] = {
    # name: (generator, episode length range in seconds)
    "DDoS": (_a_ddos, (30, 60)),
    "DoS": (_a_dos, (30, 70)),
    "PortScan": (_a_portscan, (10, 30)),
    "BruteForce": (_a_bruteforce, (40, 90)),
    "Botnet": (_a_botnet, (150, 300)),
    "WebAttack": (_a_webattack, (30, 80)),
    "Infiltration": (_a_infiltration, (30, 60)),
}


SELF_TIMED = {_b_iot_telemetry, _b_home_browsing, _b_home_discovery, _b_home_background}


def _episode_starts(rng, n: int, duration: float, ep_len: float, split) -> list[float]:
    """Spread episodes so that each chronological split region receives some."""
    bounds = np.cumsum([0.0, *split]) * duration
    counts = np.maximum(1, np.floor(np.asarray(split) * n).astype(int))
    while counts.sum() < n:
        counts[0] += 1
    starts = []
    for (a, b), c in zip(zip(bounds[:-1], bounds[1:]), counts):
        hi = max(a, b - ep_len - 1)
        starts.extend(rng.uniform(a, hi, c).tolist())
    return starts


def generate_traffic(
    profile: str = "enterprise",
    duration_s: float = 1800.0,
    seed: int = 0,
    episodes_per_attack: int = 5,
    attacks: list[str] | None = None,
    intensity: float = 1.0,
    benign_scale: float = 1.0,
    split: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> pd.DataFrame:
    """Generate a labelled flow table in the canonical schema.

    Returns flows sorted by ``ts`` with columns from
    :data:`xgnn_ids.data.features.CANONICAL_COLUMNS` plus ``attack_detail``.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    rng = np.random.default_rng(seed)
    P = PROFILES[profile](rng)
    parts = []
    for fn, rate in P.benign:
        if fn in SELF_TIMED:
            parts.append(fn(rng, None, P, 0.0, duration_s))
        else:
            parts.append(fn(rng, _poisson_times(rng, rate * benign_scale, 0, duration_s), P))
    for i in range(P.flash_crowds):
        lo, hi = (0.1, 0.5) if i % 2 == 0 else (0.82, 0.95)
        parts.append(_b_flash_crowd(rng, rng.uniform(lo, hi) * duration_s, rng.uniform(20, 40), P))

    for name in (ATTACK_CLASSES if attacks is None else attacks):
        fn, (lo, hi) = ATTACKS[name]
        for t0 in _episode_starts(rng, episodes_per_attack, duration_s, hi, split):
            length = rng.uniform(lo, hi)
            parts.append(fn(rng, t0, min(length, duration_s - t0), P, intensity))

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df = df[(df["ts"] >= 0) & (df["ts"] < duration_s)]
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["segment"] = f"synthetic-{profile}-seed{seed}"
    return df


def make_profile(profile: str = "enterprise", seed: int = 0) -> Profile:
    """Re-create the network profile used by ``generate_traffic(profile, seed=seed)``."""
    return PROFILES[profile](np.random.default_rng(seed))


def inject_attack(df: pd.DataFrame, attack: str, t0: float, length: float, profile: str = "enterprise",
                  seed: int = 0, intensity: float = 1.0, attack_seed: int | None = None) -> pd.DataFrame:
    """Add one attack episode to an existing synthetic capture (used by the dashboard simulator).

    The episode targets the same hosts as the base traffic because the profile is
    re-created from the same seed.
    """
    P = make_profile(profile, seed)
    rng = np.random.default_rng(attack_seed)
    fn, _ = ATTACKS[attack]
    new = fn(rng, t0, length, P, intensity)
    new = new[(new["ts"] >= t0) & (new["ts"] < t0 + length)].copy()
    new["segment"] = df["segment"].iloc[0] if len(df) else f"synthetic-{profile}-seed{seed}"
    out = pd.concat([df, new], ignore_index=True)
    return out.sort_values("ts", kind="stable").reset_index(drop=True)

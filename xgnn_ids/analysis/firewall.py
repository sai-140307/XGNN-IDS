"""Generate *suggested* iptables / nftables rules for an incident.

The rules are recommendations for an analyst to review - an IDS should not
push blocking rules automatically (false positives would become outages).
Internal (private) sources are quarantined rather than blacklisted.
"""
from __future__ import annotations

import hashlib

from ..data.features import is_private_ip


def firewall_rules(inc: dict, max_ips: int = 20) -> dict[str, list[str]]:
    cls = inc["class"]
    ent = inc["entity"]
    port = inc["top_ports"][0] if inc["top_ports"] else 80
    ipt: list[str] = [f"# {inc.get('id', 'incident')}: {inc['title']} -- review before applying"]
    nft: list[str] = [f"# {inc.get('id', 'incident')}: {inc['title']} -- review before applying"]

    if cls == "DDoS":
        victim = ent["dst_ip"]
        name = "xddos_" + hashlib.md5(victim.encode()).hexdigest()[:8]  # iptables names: <= 15 chars
        external = [ip for ip in inc["top_sources"] if not is_private_ip(ip)][:max_ips]
        ipt += [
            "sysctl -w net.ipv4.tcp_syncookies=1",
            f"iptables -A INPUT -p tcp -d {victim} --dport {port} --syn -m hashlimit --hashlimit-name {name} "
            "--hashlimit-mode srcip --hashlimit-above 20/second --hashlimit-burst 40 -j DROP",
            f"ipset create {name} hash:ip timeout 3600 -exist",
            *[f"ipset add {name} {ip} -exist" for ip in external],
            f"iptables -I INPUT -m set --match-set {name} src -j DROP",
        ]
        nft += [
            f"nft add set inet filter {name} '{{ type ipv4_addr; flags timeout; timeout 1h; }}'",
            f"nft add element inet filter {name} '{{ {', '.join(external)} }}'" if external else "",
            f"nft add rule inet filter input ip saddr @{name} drop",
            f"nft add rule inet filter input ip daddr {victim} tcp dport {port} tcp flags syn "
            "meter ddos_meter '{ ip saddr limit rate over 20/second }' drop",
        ]
    elif cls in {"DoS", "PortScan", "BruteForce", "WebAttack"}:
        src = ent["src_ip"]
        if is_private_ip(src):
            ipt.append(f"iptables -I FORWARD -s {src} -j DROP   # internal source: quarantine and investigate")
            nft.append(f"nft add rule inet filter forward ip saddr {src} drop")
        else:
            ipt.append(f"iptables -I INPUT -s {src} -j DROP")
            nft.append(f"nft add rule inet filter input ip saddr {src} drop")
        if cls == "BruteForce":
            ipt.append(f"iptables -A INPUT -p tcp --dport {port} -m state --state NEW -m recent --set --name auth")
            ipt.append(f"iptables -A INPUT -p tcp --dport {port} -m state --state NEW -m recent --update "
                       "--seconds 60 --hitcount 5 --name auth -j DROP")
    elif cls == "Botnet":
        c2 = ent["dst_ip"]
        ipt.append(f"iptables -I FORWARD -d {c2} -j DROP")
        ipt.append(f"iptables -I OUTPUT -d {c2} -j DROP")
        nft.append(f"nft add rule inet filter forward ip daddr {c2} drop")
        for bot in inc["top_sources"][:max_ips]:
            ipt.append(f"iptables -I FORWARD -s {bot} -j DROP   # quarantine infected host")
            nft.append(f"nft add rule inet filter forward ip saddr {bot} drop")
    elif cls == "Infiltration":
        host = ent["src_ip"]
        ipt.append(f"iptables -I FORWARD -s {host} -j DROP   # quarantine compromised host")
        nft.append(f"nft add rule inet filter forward ip saddr {host} drop")
    return {"iptables": [r for r in ipt if r], "nftables": [r for r in nft if r]}

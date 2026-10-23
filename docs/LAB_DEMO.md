# Live lab demonstration

Two ways to demo the system in front of an evaluator.

## A. No network setup (recommended for the viva)

1. Open the dashboard (deployed link, or `streamlit run app/streamlit_app.py`).
2. Sidebar → *Attack simulator*. Pick e.g. DDoS + PortScan + Botnet + BruteForce.
3. **Monitor** tab: show the alert timeline, then move the window slider to the DDoS window - the graph
   shows hundreds of sources converging on one server.
4. Open *Replay this capture as a live feed* and press *Start replay* to show detection window by window.
5. **Investigate an alert**: select a DDoS row, then a Botnet row. Walk through the narrative, the network
   evidence, the feature attribution and the GNNExplainer subgraph. Toggle SHAP to compare with IG.
6. Switch the network to *IoT network (never seen in training)* - the drift KPI turns orange/red: the
   system tells you when it is outside its comfort zone.
7. **Detection scorecard**: precision/recall per attack and time-to-first-detection.
8. **Model evaluation**: GNN vs Random Forest, zero-day and adversarial results.

## B. Real packets on an isolated lab network

> Only generate attack traffic against machines you own, on an isolated network (VirtualBox/VMware
> host-only network or a lab switch). Never against college or public infrastructure.

Topology: two VMs on a host-only network, e.g. *victim* `192.168.56.10` (Ubuntu, running
`python3 -m http.server 80` and `sshd`) and *attacker* `192.168.56.20` (Kali/Ubuntu).

On the victim (or a third VM that sees the traffic):

```bash
sudo python -m xgnn_ids live --iface enp0s8 --window 20
```

On the attacker, one at a time:

```bash
nmap -sS -p 1-2000 192.168.56.10                      # port scan
hydra -l test -P /usr/share/wordlists/rockyou.txt ssh://192.168.56.10 -t 4   # brute force
sudo hping3 -S -p 80 --flood --rand-source 192.168.56.10  # SYN flood (spoofed sources ≈ DDoS)
```

Stop each attack with Ctrl+C after ~30 s. The sensor prints incidents such as

```
[14:02:41]  2113 flows,  2009 alerts, 1 incident(s)
   !! [CRITICAL] DDoS against 192.168.56.10 from 1874 sources  (2009 flows, conf 0.97)  MITRE T1498 ...
```

and appends them to `live_alerts.jsonl`. You can also record a pcap with
`sudo tcpdump -i enp0s8 -w demo.pcap` and upload it in the dashboard (*Upload packet capture*).

**Expectation management:** the shipped model is trained on the synthetic enterprise network. On a new
network the absolute numbers will differ (the drift monitor will show it); scans, floods and brute force
have strong structural signatures and are normally still detected. For a production-grade result, retrain
on traffic from the target network (`python -m xgnn_ids train --data your_flows.csv`).

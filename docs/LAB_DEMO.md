# Live lab demonstration

Two ways to demo the system in front of an evaluator.

## A. No network setup (recommended for the viva)

1. Open the console (deployed link, or `streamlit run app/streamlit_app.py`). It opens on the busiest attack
   window of a 10-minute enterprise capture with a DDoS, a port scan, a botnet and SSH brute force.
2. Point at the map: internal hosts inside the dashed perimeter, the internet on the outer ring, attacks in
   colour. The headline names what is happening in this window.
3. Press play (or the space bar). The capture replays window by window; the coloured tracks above the
   timeline are the incidents.
4. Click the **Botnet** incident: the map dims everything except the internal machines calling one
   address outside. Show the response text and the firewall rules.
5. Under "Flows to investigate", open a flow. Walk through the case file: network evidence, what the model
   looked at (flow features and host statistics), the neighbouring flows highlighted on the map, and what
   would change the decision. Add SHAP to compare with Integrated Gradients.
6. **Change scenario** → *IoT network* → run. The sensor block shows the drift indicator rising: the system
   tells you it is outside its training conditions.
7. **How well it works**: graph vs no graph, adversarial robustness, zero-day, the unseen network and the
   explanation-quality benchmark (including where GNNExplainer fell short).

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
`sudo tcpdump -i enp0s8 -w demo.pcap` and load it in the console (*Change scenario* → *Choose a file*).

**Expectation management:** the shipped model is trained on the synthetic enterprise network. On a new
network the absolute numbers will differ (the drift monitor will show it); scans, floods and brute force
have strong structural signatures and are normally still detected. For a production-grade result, retrain
on traffic from the target network (`python -m xgnn_ids train --data your_flows.csv`).

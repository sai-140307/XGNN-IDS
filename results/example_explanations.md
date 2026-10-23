# Example explanations (auto-generated)

## DDoS

**DDoS** detected with 90% confidence on flow `117.212.205.155:30423 -> 192.168.10.50:80/TCP`.

**Why:**
- **Network context:** 192.168.10.50 received 366 flows from 251 distinct sources in this 20s window (typical host: 1 source).
- **Network context:** 46% of flows to 192.168.10.50 got no proper response (window average 26%).
- **Flow properties pushing towards this class:** psh flag count = 0.00.
- **Structural explanation (GNNExplainer):** 0 of 518 neighbouring flows are important; 71% of the explanation weight lies on flows towards the same destination.

**What would change the decision:**
- The flow would be classified BENIGN if: No response from destination no -> no; ACK flag count 6.00 -> 7.82; Mean inter-arrival time 24 ms -> 34 ms; Mean forward packet size 189 B -> 224 B.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1498 Network Denial of Service / T1499 Endpoint DoS
**Recommended response:** Enable upstream rate-limiting / scrubbing, apply SYN cookies, block or geo-filter top source prefixes, scale the targeted service.

---

## DoS

**DoS** detected with 98% confidence on flow `95.35.235.21:28637 -> 192.168.10.50:80/TCP`.

**Why:**
- **Network context:** 95.35.235.21 opened 970 connections to 192.168.10.50:80 in this window (typical: 1).
- **Network context:** 192.168.10.50 received 1,029 flows from 47 distinct sources in this 20s window (typical host: 1 source).
- **Flow properties pushing towards this class:** forward bytes = 3,212 B, forward packets = 8.00 packets, destination port is web (80/443/8080) = yes.
- **Structural explanation (GNNExplainer):** 0 of 1180 neighbouring flows are important; 87% of the explanation weight lies on flows towards the same destination.

**What would change the decision:**
- The flow would be classified BENIGN if: Forward packets 8.00 packets -> -0.81 packets; Forward bytes 3,212 B -> -0.65 B; Byte rate 8,732 B/s -> 1.30 B/s; Backward (response) bytes 1,479 B -> 95,358,976,000 B.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1499 Endpoint Denial of Service
**Recommended response:** Block the source IP at the edge, tune web-server connection timeouts and per-client connection limits.

---

## PortScan

**PortScan** detected with 99% confidence on flow `109.117.151.79:59370 -> 192.168.10.117:636/TCP`.

**Why:**
- **Network context:** 109.117.151.79 used 1,119 different destination ports towards 192.168.10.117 (typical pair: 1).
- **Flow properties pushing towards this class:** other well-known port (<1024) = yes, fin flag count = 0.00, ack flag count = 1.00.
- **Structural explanation (GNNExplainer):** 0 of 1153 neighbouring flows are important; 97% of the explanation weight lies on flows towards the same destination.

**What would change the decision:**
- The flow would be classified BENIGN if: Backward (response) bytes 50 B -> 11,695,262 B; Packet rate 2,000 packets/s -> 0.04 packets/s; Mean forward packet size 47 B -> 1,448 B; Mean inter-arrival time 0.71 ms -> 1,904 ms.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1046 Network Service Discovery
**Recommended response:** Block the scanner, verify which services responded as open, watch the source for follow-up exploitation.

---

## BruteForce

**BruteForce** detected with 100% confidence on flow `109.117.151.79:51070 -> 192.168.10.52:21/TCP`.

**Why:**
- **Network context:** 109.117.151.79 opened 42 connections to 192.168.10.52:21 in this window (typical: 1).
- **Flow properties pushing towards this class:** destination port is remote-login (21/22/23/3389) = yes, psh flag count = 10, ack flag count = 31.
- **Structural explanation (GNNExplainer):** 0 of 41 neighbouring flows are important; 100% of the explanation weight lies on flows towards the same destination.

**What would change the decision:**
- The flow would be classified BENIGN if: Mean inter-arrival time 61 ms -> 16,243 ms; Backward (response) bytes 1,605 B -> 20,759,104 B; Flow duration 1,723 ms -> 4,417,798 ms; Response/request byte ratio 1.01x -> 2054.02x.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1110 Brute Force
**Recommended response:** Lock out / fail2ban the source, enforce key-based auth or MFA, audit the targeted accounts for successful logins.

---

## Botnet

**Botnet** detected with 99% confidence on flow `192.168.10.159:54747 -> 66.137.26.58:6667/TCP`.

**Why:**
- **Network context:** 66.137.26.58 received 12 flows from 8 distinct sources in this 20s window (typical host: 1 source).
- **Flow properties pushing towards this class:** registered port (1024-49151) = yes, destination port is web (80/443/8080) = no, destination port is dns (53) = no.
- **Structural explanation (GNNExplainer):** 0 of 299 neighbouring flows are important; 4% of the explanation weight lies on flows towards the same destination.

**What would change the decision:**
- The flow would be classified BENIGN if: Mean forward packet size 139 B -> 515 B; Packet rate 21 packets/s -> 0.22 packets/s; ACK flag count 6.00 -> 40; Flow duration 339 ms -> 16,240 ms.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1071 Application Layer Protocol (C2)
**Recommended response:** Isolate the internal hosts, block the C2 address/domain, image the hosts for forensic analysis.

---

## WebAttack

**WebAttack** detected with 95% confidence on flow `89.194.59.89:22872 -> 192.168.10.50:80/TCP`.

**Why:**
- **Network context:** 192.168.10.50 received 114 flows from 56 distinct sources in this 20s window (typical host: 1 source).
- **Network context:** 89.194.59.89 opened 45 connections to 192.168.10.50:80 in this window (typical: 1).
- **Network context:** Mean request packet of this flow is 854 B vs 377 B for other port-80 flows.
- **Flow properties pushing towards this class:** mean forward packet size = 854 B, mean inter-arrival time = 591 ms, packet rate = 2.27 packets/s.
- **Structural explanation (GNNExplainer):** 113 of 307 neighbouring flows are important; 95% of the explanation weight lies on flows towards the same destination.

**What would change the decision:**
- The flow would be classified BENIGN if: ACK flag count 5.00 -> 69; Backward (response) packets 3.00 packets -> 25 packets; Backward (response) bytes 1,755 B -> 849,857 B; Flow duration 2,645 ms -> 473,060 ms.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1190 Exploit Public-Facing Application
**Recommended response:** Inspect web-server logs for the request payloads, enable WAF rules, patch / sanitise the vulnerable endpoint.

---

## Infiltration

**Infiltration** detected with 100% confidence on flow `192.168.10.130:1815 -> 192.168.10.144:3389/TCP`.

**Why:**
- **Network context:** 192.168.10.130 contacted 87 distinct hosts in this window (typical host: 1).
- **Network context:** 192.168.10.130 used 4 different destination ports towards 192.168.10.144 (typical pair: 1).
- **Network context:** 88% of flows to 192.168.10.144 got no proper response (window average 43%).
- **Flow properties pushing towards this class:** destination port is remote-login (21/22/23/3389) = yes, rst flag count = 1.00, destination port is web (80/443/8080) = no.
- **Structural explanation (GNNExplainer):** 0 of 767 neighbouring flows are important; 58% of the explanation weight lies on other flows from the same source.

**What would change the decision:**
- The flow would be classified BENIGN if: Backward (response) bytes 40 B -> 123,644,048 B; Response/request byte ratio 0.67x -> 90365.62x; Mean forward packet size 60 B -> 3,493 B; Forward bytes 60 B -> 78,702 B.
- Removing neighbouring flows alone does not change the decision: the flow's own features are sufficient evidence.

**MITRE ATT&CK:** T1021 Remote Services / T1570 Lateral Tool Transfer
**Recommended response:** Quarantine the internal source host, reset its credentials, review east-west traffic for other compromised hosts.
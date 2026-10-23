# Example explanations (auto-generated)

## DDoS

**DDoS** detected with 100% confidence on flow `121.160.102.234:38608 -> 192.168.10.50:35802/UDP`.

**Why:**
- **Network context:** 192.168.10.50 received 366 flows from 251 distinct sources in this 20s window (typical host: 1 source).
- **Network context:** 46% of flows to 192.168.10.50 got no proper response (window average 26%).
- **Graph context the model relied on:** destination host: distinct hosts contacting it = 251; destination host: distinct own ports targeted = 41.
- **Neighbouring flows:** 1 of 518 neighbouring flows carry 80% of the structural evidence; 93% of it comes from flows towards the same destination.

**What would change the decision:**
- No realistic change to this flow's own properties makes it benign - the decision rests on its context.

**MITRE ATT&CK:** T1498 Network Denial of Service / T1499 Endpoint DoS
**Recommended response:** Enable upstream rate-limiting / scrubbing, apply SYN cookies, block or geo-filter top source prefixes, scale the targeted service.

---

## DoS

**DoS** detected with 100% confidence on flow `95.35.235.21:28637 -> 192.168.10.50:80/TCP`.

**Why:**
- **Network context:** 95.35.235.21 opened 970 connections to 192.168.10.50:80 in this window (typical: 1).
- **Network context:** 192.168.10.50 received 1,029 flows from 47 distinct sources in this 20s window (typical host: 1 source).
- **Graph context the model relied on:** source host: flows sent by host = 970; destination host: flows sent by host = 24.
- **Neighbouring flows:** 724 of 1180 neighbouring flows carry 80% of the structural evidence; 100% of it comes from flows towards the same destination.

**What would change the decision:**
- No realistic change to this flow's own properties makes it benign - the decision rests on its context.

**MITRE ATT&CK:** T1499 Endpoint Denial of Service
**Recommended response:** Block the source IP at the edge, tune web-server connection timeouts and per-client connection limits.

---

## PortScan

**PortScan** detected with 100% confidence on flow `109.117.151.79:59370 -> 192.168.10.117:636/TCP`.

**Why:**
- **Network context:** 109.117.151.79 used 1,119 different destination ports towards 192.168.10.117 (typical pair: 1).
- **Graph context the model relied on:** source host: distinct destination ports used = 1,119; source host: flows sent by host = 1,119.
- **Properties of the flow itself:** other well-known port (<1024) = yes.
- **Neighbouring flows:** 879 of 1153 neighbouring flows carry 80% of the structural evidence; 100% of it comes from flows towards the same destination.

**What would change the decision:**
- No realistic change to this flow's own properties makes it benign - the decision rests on its context.

**MITRE ATT&CK:** T1046 Network Service Discovery
**Recommended response:** Block the scanner, verify which services responded as open, watch the source for follow-up exploitation.

---

## BruteForce

**BruteForce** detected with 100% confidence on flow `109.117.151.79:51070 -> 192.168.10.52:21/TCP`.

**Why:**
- **Network context:** 109.117.151.79 opened 42 connections to 192.168.10.52:21 in this window (typical: 1).
- **Graph context the model relied on:** source host: flows sent by host = 42; source host: private (internal) address = no.
- **Properties of the flow itself:** destination port is remote-login (21/22/23/3389) = yes; forward packets = 16; psh flag count = 10.
- **Neighbouring flows:** 33 of 41 neighbouring flows carry 80% of the structural evidence; 100% of it comes from flows towards the same destination.

**What would change the decision:**
- No realistic change to this flow's own properties makes it benign - the decision rests on its context.

**MITRE ATT&CK:** T1110 Brute Force
**Recommended response:** Lock out / fail2ban the source, enforce key-based auth or MFA, audit the targeted accounts for successful logins.

---

## Botnet

**Botnet** detected with 100% confidence on flow `192.168.10.159:54747 -> 66.137.26.58:6667/TCP`.

**Why:**
- **Network context:** 66.137.26.58 received 12 flows from 8 distinct sources in this 20s window (typical host: 1 source).
- **Graph context the model relied on:** destination host: distinct hosts contacting it = 8; destination host: flows received by host = 12.
- **Properties of the flow itself:** registered port (1024-49151) = yes; destination port is web (80/443/8080) = no; protocol = udp = no.
- **Neighbouring flows:** 9 of 299 neighbouring flows carry 80% of the structural evidence; 100% of it comes from flows towards the same destination.

**What would change the decision:**
- It would be classified BENIGN if its ack flag count were 128 instead of 6; backward (response) packets were 48 instead of 3; forward packets were 25 instead of 4; byte rate were 98,192 B/s instead of 4,547 B/s.

**MITRE ATT&CK:** T1071 Application Layer Protocol (C2)
**Recommended response:** Isolate the internal hosts, block the C2 address/domain, image the hosts for forensic analysis.

---

## WebAttack

**WebAttack** detected with 99% confidence on flow `89.194.59.89:22872 -> 192.168.10.50:80/TCP`.

**Why:**
- **Network context:** 192.168.10.50 received 114 flows from 56 distinct sources in this 20s window (typical host: 1 source).
- **Network context:** 89.194.59.89 opened 45 connections to 192.168.10.50:80 in this window (typical: 1).
- **Network context:** Mean request packet of this flow is 854 B vs 377 B for other port-80 flows.
- **Graph context the model relied on:** source host: flows sent by host = 45; destination host: distinct hosts contacting it = 56.
- **Neighbouring flows:** 87 of 307 neighbouring flows carry 80% of the structural evidence; 80% of it comes from flows towards the same destination.

**What would change the decision:**
- No realistic change to this flow's own properties makes it benign - the decision rests on its context.

**MITRE ATT&CK:** T1190 Exploit Public-Facing Application
**Recommended response:** Inspect web-server logs for the request payloads, enable WAF rules, patch / sanitise the vulnerable endpoint.

---

## Infiltration

**Infiltration** detected with 100% confidence on flow `192.168.10.130:1815 -> 192.168.10.144:3389/TCP`.

**Why:**
- **Network context:** 192.168.10.130 contacted 87 distinct hosts in this window (typical host: 1).
- **Network context:** 192.168.10.130 used 4 different destination ports towards 192.168.10.144 (typical pair: 1).
- **Network context:** 88% of flows to 192.168.10.144 got no proper response (window average 43%).
- **Graph context the model relied on:** destination host: share of inbound syn-only flows = 88%; source host: flows sent by host = 446.
- **Neighbouring flows:** 228 of 767 neighbouring flows carry 80% of the structural evidence; 69% of it comes from other flows from the same source.

**What would change the decision:**
- No realistic change to this flow's own properties makes it benign - the decision rests on its context.

**MITRE ATT&CK:** T1021 Remote Services / T1570 Lateral Tool Transfer
**Recommended response:** Quarantine the internal source host, reset its credentials, review east-west traffic for other compromised hosts.
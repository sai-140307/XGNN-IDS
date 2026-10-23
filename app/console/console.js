/* XGNN-IDS console: a perimeter map of the network, replayed window by window.
   Talks to the Streamlit host through the component postMessage protocol. */
"use strict";

const CLASSES = ["BENIGN", "DDoS", "DoS", "PortScan", "BruteForce", "Botnet", "WebAttack", "Infiltration"];
const ATTACKS = CLASSES.slice(1);
const COLOR = { BENIGN: "#B4BEC6", DDoS: "#E0352B", DoS: "#F07F1F", PortScan: "#C99A00", BruteForce: "#7A4FE0",
  Botnet: "#D02F8C", WebAttack: "#0E9C86", Infiltration: "#2563EB" };
const NICE = { BENIGN: "Benign", DDoS: "DDoS", DoS: "DoS", PortScan: "Port scan", BruteForce: "Brute force",
  Botnet: "Botnet", WebAttack: "Web attack", Infiltration: "Infiltration" };
const SEV = { critical: 3, high: 2, medium: 1, none: 0 };
const GRAPH_MODELS = new Set(["gcn", "gat", "egraphsage", "egraphsage_nostats"]);

const $ = (s) => document.querySelector(s);
const fmt = d3.format(",");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const IP_RE = /\b(\d{1,3}(?:\.\d{1,3}){3})(:\d+)?\b/g;
const withAddr = (s) => esc(s).replace(IP_RE, '<span class="addr">$1$2</span>');
const mmss = (t) => { t = Math.max(0, Math.round(t || 0)); return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, "0")}`; };
const pct = (v, d = 1) => (v == null ? "n/a" : `${(v * 100).toFixed(d)}%`);
const num = (v, d = 3) => (v == null ? "n/a" : (+v).toFixed(d));

/* ------------------------------------------------------------------ Streamlit bridge */
const ST = {
  send(type, data = {}) { window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*"); },
  ready() { this.send("streamlit:componentReady", { apiVersion: 1 }); },
  value(v) { this.send("streamlit:setComponentValue", { value: v, dataType: "json" }); },
  height(h) { this.send("streamlit:setFrameHeight", { height: h }); },
};
let pendingNonce = null;
function request(action, extra = {}) {
  const nonce = `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
  pendingNonce = nonce;
  ST.value({ action, nonce, ...extra });
  return nonce;
}

const S = {
  data: null, key: null, win: 0, playing: false, speed: 1, timer: null, view: "map",
  panel: { mode: "list" }, highlight: null, rulesFmt: "iptables", attrMethod: null,
  scenario: { profile: "enterprise", minutes: 10, attacks: ["DDoS", "PortScan", "Botnet", "BruteForce"], intensity: 1, seed: 42 },
  evalRendered: null, methodRendered: false,
};

window.addEventListener("message", (e) => {
  if (e.data && e.data.type === "streamlit:render") onRender(e.data.args || {});
});

function onRender(args) {
  const p = args.payload;
  if (!p) return;
  const first = !S.data;
  const changed = first || (p.meta && p.meta.key) !== S.key;
  S.data = p;
  if (p.nonce && p.nonce === pendingNonce) { pendingNonce = null; busy(false); }
  showError(p.error);
  if (changed) { S.key = p.meta && p.meta.key; onNewData(); } else { onUpdate(); }
}

/* ------------------------------------------------------------------ busy / errors */
function busy(on, text) {
  $("#busy").hidden = !on;
  if (text) $("#busyText").textContent = text;
}
function showError(msg) {
  const box = $("#errorBox");
  box.hidden = !msg;
  if (msg) box.textContent = msg;
}

/* ------------------------------------------------------------------ new data */
function onNewData() {
  setPlaying(false);
  const d = S.data;
  if (d.meta.scenario) S.scenario = { ...S.scenario, ...d.meta.scenario };
  const w = d.windows;
  S.win = w.length ? w.reduce((b, x) => (x.alerts > w[b].alerts ? x.i : b), 0) : 0;
  S.panel = { mode: "list" };
  S.highlight = null;
  renderTop();
  M.reset();
  drawWave();
  gotoWindow(S.win, true);
  renderSide();
  S.evalRendered = null;
  if (S.view === "eval") renderEval();
}

function onUpdate() {
  renderTop();
  const ex = S.data.explanation;
  if (S.panel.mode === "explain" && ex && ex.row === S.panel.row) {
    S.highlight = explanationHighlight(ex);
    if (ex.flow && ex.flow.window !== S.win && S.data.graphs[ex.flow.window]) gotoWindow(ex.flow.window);
    else M.applyHighlight();
  }
  renderSide();
}

/* ------------------------------------------------------------------ top bar */
function renderTop() {
  const m = S.data.meta;
  const sc = m.scenario;
  const line = sc
    ? `<b>${sc.profile === "iot" ? "IoT network" : "Enterprise LAN"}</b>, ${sc.minutes} min, ${sc.attacks.length ? sc.attacks.map((a) => NICE[a]).join(", ") : "no attacks"}`
    : `<b>${esc(m.source || "Capture")}</b>, ${fmt(m.flows)} flows`;
  $("#scenarioLine").innerHTML = line;
  const sel = $("#modelSelect");
  sel.innerHTML = (m.models || []).map((x) => `<option value="${esc(x.name)}" ${x.name === m.model_name ? "selected" : ""}>${esc(x.display)}</option>`).join("");
}

$("#modelSelect").addEventListener("change", (e) => {
  busy(true, `Switching to ${e.target.selectedOptions[0].textContent}`);
  request("model", { name: e.target.value });
});

document.querySelectorAll(".views button").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
function setView(v) {
  S.view = v;
  document.querySelectorAll(".views button").forEach((b) => b.classList.toggle("on", b.dataset.view === v));
  $("#view-map").hidden = v !== "map";
  $("#view-eval").hidden = v !== "eval";
  $("#view-method").hidden = v !== "method";
  if (v !== "map") setPlaying(false);
  if (v === "eval") renderEval();
  if (v === "method") renderMethod();
  if (v === "map") { M.layout(); drawWave(); }
}

/* ------------------------------------------------------------------ map */
function hash(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0) / 4294967295;
}

const M = {
  init() {
    this.svg = d3.select("#map");
    this.root = this.svg.append("g");
    this.zone = this.root.append("g");
    this.gLinks = this.root.append("g");
    this.gHits = this.root.append("g");
    this.gNodes = this.root.append("g");
    this.gLabels = this.root.append("g");
    this.gZoneLabels = this.root.append("g");
    this.nodes = new Map();
    this.nodeArr = [];
    this.links = [];
    this.zoom = d3.zoom().scaleExtent([0.5, 8]).on("zoom", (e) => this.root.attr("transform", e.transform));
    this.svg.call(this.zoom).on("dblclick.zoom", null);
    this.sim = d3.forceSimulation()
      .alphaDecay(0.035).velocityDecay(0.38)
      .force("link", d3.forceLink().id((d) => d.id).distance(36).strength((l) => (l.attack ? 0.035 : 0.006)))
      .force("charge", d3.forceManyBody().strength((d) => (d.internal ? -16 : -3)).distanceMax(110))
      .force("collide", d3.forceCollide((d) => d.r + 1.4).iterations(2))
      .on("tick", () => this.tick());
    new ResizeObserver(() => this.layout()).observe(document.getElementById("mapwrap"));
    $("#resetZoom").addEventListener("click", () => this.svg.transition().duration(400).call(this.zoom.transform, d3.zoomIdentity));
  },
  layout() {
    const r = document.getElementById("mapwrap").getBoundingClientRect();
    if (!r.width) return;
    this.w = r.width; this.h = r.height;
    const wide = this.w > 760;
    this.cx = wide ? this.w * 0.57 : this.w * 0.5;
    this.cy = this.h * 0.55;
    this.R = Math.max(110, Math.min(wide ? this.w * 0.4 : this.w * 0.46, this.h * 0.42));
    this.rin = this.R * 0.48;
    this.zone.selectAll("*").remove();
    this.zone.append("circle").attr("class", "outer-ring").attr("cx", this.cx).attr("cy", this.cy).attr("r", this.R * 1.04);
    this.zone.append("circle").attr("class", "perimeter").attr("cx", this.cx).attr("cy", this.cy).attr("r", this.rin * 1.08);
    this.gZoneLabels.selectAll("*").remove();
    this.gZoneLabels.append("text").attr("class", "zone-label").attr("x", this.cx).attr("y", this.cy + this.rin * 1.08 + 16)
      .attr("text-anchor", "middle").text("Your network");
    this.gZoneLabels.append("text").attr("class", "zone-label").attr("x", this.cx).attr("y", this.cy - this.R * 1.04 - 8)
      .attr("text-anchor", "middle").text("Internet");
    this.sim.force("radial", d3.forceRadial((d) => (d.internal ? d.rt * this.rin : this.R * 0.94), this.cx, this.cy)
      .strength((d) => (d.internal ? 0.09 : 0.45)));
    if (this.nodeArr.length) this.sim.alpha(0.3).restart();
  },
  reset() {
    this.nodes.clear();
    this.layout();
    this.svg.call(this.zoom.transform, d3.zoomIdentity);
  },
  placeNew(n) {
    const a = 2 * Math.PI * hash(n.id);
    if (n.internal) {
      n.rt = 0.25 + 0.7 * Math.sqrt(hash(n.id + "r"));
      n.x = this.cx + Math.cos(a) * n.rt * this.rin; n.y = this.cy + Math.sin(a) * n.rt * this.rin;
    } else {
      n.x = this.cx + Math.cos(a) * this.R * 0.94; n.y = this.cy + Math.sin(a) * this.R * 0.94;
    }
  },
  show(gi, first) {
    const g = S.data.graphs[gi];
    if (!g) return;
    const nodes = g.nodes.map(([ip, internal, deg, alert]) => {
      let n = this.nodes.get(ip);
      if (!n) { n = { id: ip, internal: !!internal }; this.placeNew(n); this.nodes.set(ip, n); }
      n.deg = deg; n.alert = !!alert; n.acls = null;
      n.r = n.internal ? 3.2 + Math.min(9, Math.sqrt(deg) * 0.75) : 2.3 + Math.min(4.5, Math.sqrt(deg) * 0.45);
      return n;
    });
    const present = new Set(nodes.map((n) => n.id));
    for (const k of [...this.nodes.keys()]) if (!present.has(k)) this.nodes.delete(k);
    const links = g.links.map(([s, t, c, n, row, conf]) => ({
      source: nodes[s].id, target: nodes[t].id, sn: nodes[s], tn: nodes[t], cls: CLASSES[c], attack: c > 0, n, row, conf,
      key: `${nodes[s].id}>${nodes[t].id}`,
    })).sort((a, b) => a.attack - b.attack);
    for (const l of links) if (l.attack) { l.sn.acls = l.sn.acls || l.cls; l.tn.acls = l.tn.acls || l.cls; }
    this.nodeArr = nodes; this.links = links;
    const nAttack = links.filter((l) => l.attack).length;
    this.attackOpacity = nAttack > 200 ? 0.38 : nAttack > 80 ? 0.6 : 0.9;
    this.attackWidth = nAttack > 200 ? 0.7 : 1;

    this.linkSel = this.gLinks.selectAll("line").data(links, (d) => d.key).join("line")
      .attr("class", (d) => `link ${d.attack ? "attack" : "benign"}`)
      .attr("stroke", (d) => (d.attack ? COLOR[d.cls] : null))
      .style("stroke-opacity", (d) => (d.attack ? this.attackOpacity : null))
      .attr("stroke-width", (d) => this.baseWidth(d));
    this.hitSel = this.gHits.selectAll("line").data(links, (d) => d.key).join("line").attr("class", "hit")
      .on("mousemove", (e, d) => tip(e, `<span class="mono">${esc(d.sn.id)}</span> to <span class="mono">${esc(d.tn.id)}</span><small>${fmt(d.n)} flow${d.n > 1 ? "s" : ""}, ${d.attack ? `${NICE[d.cls]} (${pct(d.conf, 0)})` : "benign"}. Click to explain.</small>`))
      .on("mouseleave", hideTip)
      .on("click", (e, d) => openExplain(d.row, { from: S.panel.mode === "incident" ? S.panel : { mode: "list" } }));
    this.nodeSel = this.gNodes.selectAll("circle").data(nodes, (d) => d.id).join("circle")
      .attr("class", (d) => `node ${d.internal ? "internal" : "external"}`)
      .attr("r", (d) => d.r)
      .attr("stroke", (d) => (d.acls ? COLOR[d.acls] : null))
      .attr("stroke-width", (d) => (d.acls ? 2 : null))
      .on("mousemove", (e, d) => tip(e, `<span class="mono">${esc(d.id)}</span><small>${d.internal ? "Internal host" : "External host"}, ${fmt(d.deg)} flows in this window${d.acls ? `, involved in ${NICE[d.acls]}` : ""}</small>`))
      .on("mouseleave", hideTip)
      .call(d3.drag()
        .on("start", (e, d) => { if (!e.active) this.sim.alphaTarget(0.2).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
        .on("end", (e, d) => { if (!e.active) this.sim.alphaTarget(0); d.fx = null; d.fy = null; }));
    this.updateLabels();
    this.sim.nodes(nodes);
    this.sim.force("link").links(links);
    this.sim.alpha(first ? 1 : 0.4).restart();
    this.applyHighlight();
  },
  baseWidth(d) {
    return d.attack ? (1.1 + Math.log2(d.n + 1) * 0.45) * this.attackWidth : 0.55 + Math.log2(d.n + 1) * 0.25;
  },
  updateLabels() {
    const keyNodes = new Set();
    if (S.highlight && S.highlight.win === S.win) S.highlight.nodes.forEach((n) => keyNodes.add(n));
    const victims = this.nodeArr.filter((n) => n.acls).sort((a, b) => b.deg - a.deg).slice(0, 3).map((n) => n.id);
    victims.forEach((v) => keyNodes.add(v));
    this.labelNodes = this.nodeArr.filter((n) => keyNodes.has(n.id)).slice(0, 10);
    this.labelSel = this.gLabels.selectAll("text").data(this.labelNodes, (d) => d.id).join("text").attr("class", "node-label").text((d) => d.id);
  },
  applyHighlight() {
    const h = S.highlight;
    const on = !!(h && (h.win === S.win || h.win == null));
    this.root.classed("dimmed", on);
    if (!this.linkSel) return;
    this.linkSel.classed("key", (d) => on && h.links.has(d.key))
      .classed("explained", (d) => on && h.target === d.key)
      .attr("stroke", (d) => (on && h.links.has(d.key) && !d.attack ? "#0F1720" : d.attack ? COLOR[d.cls] : null))
      .attr("stroke-width", (d) => {
        if (on && h.target === d.key) return 4.5;
        if (on && h.links.has(d.key)) return 1.2 + 3 * (h.links.get(d.key) || 0.5);
        return this.baseWidth(d);
      });
    this.nodeSel.classed("key", (d) => on && h.nodes.has(d.id));
    this.updateLabels();
    this.tick();
  },
  tick() {
    const lim = this.rin * 1.02;
    for (const n of this.nodeArr) {
      if (!n.internal) continue;
      const dx = n.x - this.cx, dy = n.y - this.cy, dist = Math.hypot(dx, dy) || 1, max = lim - n.r;
      if (dist > max) { n.x = this.cx + (dx / dist) * max; n.y = this.cy + (dy / dist) * max; }
    }
    if (!this.linkSel) return;
    const pos = (sel) => sel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y).attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    pos(this.linkSel); pos(this.hitSel);
    this.nodeSel.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    // place labels right of their node, hiding any that would collide with one already placed
    const placed = [];
    this.labelSel.attr("x", (d) => d.x + d.r + 4).attr("y", (d) => d.y + 3.5).attr("visibility", (d) => {
      const box = [d.x + d.r + 4, d.y - 6, d.x + d.r + 4 + d.id.length * 6.6, d.y + 6];
      const hit = placed.some((b) => box[0] < b[2] && b[0] < box[2] && box[1] < b[3] && b[1] < box[3]);
      if (!hit) placed.push(box);
      return hit ? "hidden" : "visible";
    });
  },
};

function tip(e, html) {
  const t = $("#tooltip");
  const r = document.getElementById("mapwrap").getBoundingClientRect();
  t.innerHTML = html; t.hidden = false;
  const x = Math.min(e.clientX - r.left + 14, r.width - 330);
  t.style.left = `${x}px`; t.style.top = `${e.clientY - r.top + 14}px`;
}
function hideTip() { $("#tooltip").hidden = true; }

/* ------------------------------------------------------------------ window navigation */
function gotoWindow(i, first = false) {
  const n = S.data.windows.length;
  if (!n) return;
  S.win = Math.max(0, Math.min(n - 1, i));
  M.show(S.win, first);
  renderHeadline();
  renderLegend();
  updateWave();
  renderClock();
  if (S.panel.mode === "list") renderSide();
}

function activeIncidents(w) {
  return S.data.incidents.filter((i) => i.w0 <= w && w <= i.w1)
    .sort((a, b) => SEV[b.severity] - SEV[a.severity] || b.flows - a.flows);
}

function renderHeadline() {
  const d = S.data, W = d.windows[S.win];
  if (!W) { $("#headline").innerHTML = ""; return; }
  const act = activeIncidents(S.win);
  const when = `<div class="when">${mmss(W.t0)} to ${mmss(W.t1)} of ${mmss(d.meta.duration)}</div>`;
  let h1, p;
  if (act.length) {
    const top = act[0];
    h1 = `<span class="bar" style="background:${COLOR[top.cls]}"></span>${withAddr(top.title)}`;
    const more = act.length > 1 ? ` ${act.length - 1} other incident${act.length > 2 ? "s are" : " is"} active at the same time.` : "";
    p = `${fmt(W.alerts)} of the ${fmt(W.flows)} flows in this window are malicious.${more}`;
  } else if (W.alerts > 0) {
    h1 = "Scattered alerts";
    p = `${fmt(W.alerts)} flows look malicious but do not form an incident yet.`;
  } else {
    h1 = "Quiet window";
    p = `${fmt(W.flows)} flows between ${fmt(W.hosts)} hosts. Nothing crossed the alert threshold.`;
  }
  $("#headline").innerHTML = `${when}<h1>${h1}</h1><p>${p}</p>`;
}

function renderLegend() {
  const W = S.data.windows[S.win];
  const g = S.data.graphs[S.win];
  if (!W) return;
  const items = [];
  ATTACKS.forEach((c, k) => { if (W.counts[k] > 0) items.push(`<span><i style="background:${COLOR[c]}"></i>${NICE[c]} ${fmt(W.counts[k])}</span>`); });
  items.push(`<span><i style="background:${COLOR.BENIGN}"></i>Benign ${fmt(W.flows - W.alerts)}</span>`);
  items.push(`<span><b class="node-in"></b>Internal host</span><span><b class="node-out"></b>External host</span>`);
  if (g && g.hidden) items.push(`<span>${fmt(g.hidden)} quieter hosts not drawn</span>`);
  $("#legend").innerHTML = items.join("");
}

function renderClock() {
  const d = S.data, W = d.windows[S.win];
  if (!W) return;
  $("#clock").innerHTML = `${mmss(W.t1)} / ${mmss(d.meta.duration)}<small>Window ${S.win + 1} of ${d.windows.length}</small>`;
}

/* ------------------------------------------------------------------ transport */
const WV = {};
function drawWave() {
  const wrap = document.getElementById("waveWrap");
  const r = wrap.getBoundingClientRect();
  if (!r.width || !S.data) return;
  const svg = d3.select("#wave");
  svg.selectAll("*").remove();
  const ws = S.data.windows, n = ws.length;
  const W = r.width, H = r.height;
  // incident tracks (lanes) above the bars
  const lanes = [];
  const incs = [...S.data.incidents].sort((a, b) => a.w0 - b.w0);
  const placed = incs.map((inc) => {
    let lane = lanes.findIndex((end) => end < inc.w0);
    if (lane === -1) { lane = lanes.length; lanes.push(-1); }
    lanes[lane] = inc.w1;
    return { inc, lane };
  }).filter((p) => p.lane < 4);
  const nl = Math.min(lanes.length, 4);
  const trackH = nl ? nl * 7 + 4 : 0;
  const x = d3.scaleBand().domain(d3.range(n)).range([0, W]).paddingInner(n > 60 ? 0.1 : 0.2);
  const top = trackH + 4, bottom = H - 16;
  const maxFlows = d3.max(ws, (w) => w.flows) || 1;
  const y = d3.scaleSqrt().domain([0, maxFlows]).range([0, bottom - top]);
  Object.assign(WV, { x, n });
  WV.cur = svg.append("rect").attr("class", "cur").attr("y", 0).attr("height", bottom).attr("width", x.step());
  const gb = svg.append("g");
  ws.forEach((w, i) => {
    const total = y(w.flows);
    const scale = w.flows ? total / w.flows : 0;
    let yy = bottom;
    const benign = w.flows - w.alerts;
    gb.append("rect").attr("x", x(i)).attr("width", x.bandwidth()).attr("y", yy - benign * scale).attr("height", benign * scale).attr("fill", "#C9D1D7");
    yy -= benign * scale;
    ATTACKS.forEach((c, k) => {
      const h = w.counts[k] * scale;
      if (h <= 0) return;
      gb.append("rect").attr("x", x(i)).attr("width", x.bandwidth()).attr("y", yy - h).attr("height", h).attr("fill", COLOR[c]);
      yy -= h;
    });
  });
  const gt = svg.append("g");
  placed.forEach(({ inc, lane }) => {
    gt.append("rect").attr("x", x(inc.w0)).attr("width", Math.max(3, x(inc.w1) + x.bandwidth() - x(inc.w0)))
      .attr("y", lane * 7).attr("height", 4).attr("rx", 2).attr("fill", COLOR[inc.cls])
      .append("title").text(`${inc.id}: ${inc.title}`);
  });
  const ticks = d3.ticks(0, S.data.meta.duration || 1, Math.max(2, Math.floor(W / 110)));
  const tx = (t) => {
    const i = ws.findIndex((w) => w.t1 > t);
    return i < 0 ? W : x(i) + x.bandwidth() * Math.min(1, Math.max(0, (t - ws[i].t0) / ((ws[i].t1 - ws[i].t0) || 1)));
  };
  const ax = svg.append("g").attr("class", "axis");
  ticks.forEach((t) => ax.append("text").attr("x", tx(t)).attr("y", H - 2).attr("text-anchor", "middle").text(mmss(t)));
  WV.head = svg.append("line").attr("class", "playhead").attr("y1", 0).attr("y2", bottom + 2);
  const pick = (e) => {
    const [mx] = d3.pointer(e, svg.node());
    const i = Math.max(0, Math.min(n - 1, Math.floor(mx / x.step())));
    if (i !== S.win) gotoWindow(i);
  };
  svg.on("mousedown", (e) => { pick(e); svg.on("mousemove.drag", pick); })
    .on("mouseup mouseleave", () => svg.on("mousemove.drag", null));
  updateWave();
}
function updateWave() {
  if (!WV.x) return;
  const xx = WV.x(S.win);
  if (xx == null) return;
  WV.cur.attr("x", xx - (WV.x.step() - WV.x.bandwidth()) / 2);
  WV.head.attr("x1", xx + WV.x.bandwidth() / 2).attr("x2", xx + WV.x.bandwidth() / 2);
}
new ResizeObserver(() => drawWave()).observe(document.getElementById("waveWrap"));

function setPlaying(on) {
  S.playing = on;
  $("#play").classList.toggle("on", on);
  $("#play").setAttribute("aria-label", on ? "Pause" : "Play");
  clearInterval(S.timer);
  if (!on || !S.data) return;
  if (S.win >= S.data.windows.length - 1) gotoWindow(0);
  S.timer = setInterval(() => {
    if (S.win >= S.data.windows.length - 1) { setPlaying(false); return; }
    gotoWindow(S.win + 1);
  }, 1200 / S.speed);
}
$("#play").addEventListener("click", () => setPlaying(!S.playing));
document.querySelectorAll("#speed button").forEach((b) => b.addEventListener("click", () => {
  S.speed = +b.dataset.speed;
  document.querySelectorAll("#speed button").forEach((x) => x.classList.toggle("on", x === b));
  if (S.playing) setPlaying(true);
}));
document.addEventListener("keydown", (e) => {
  if (S.view !== "map" || !S.data || e.target.closest("input, select, textarea")) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!S.playing); }
  if (e.code === "ArrowRight") gotoWindow(S.win + 1);
  if (e.code === "ArrowLeft") gotoWindow(S.win - 1);
});

/* ------------------------------------------------------------------ side panel */
function renderSide() {
  const side = $("#side");
  if (!S.data) { side.innerHTML = ""; return; }
  if (S.panel.mode === "incident") return renderIncident(side);
  if (S.panel.mode === "explain") return renderExplain(side);
  return renderList(side);
}

function renderList(side) {
  const incs = S.data.incidents;
  const live = new Set(activeIncidents(S.win).map((i) => i.id));
  let html = `<div class="side-head"><h2>Incidents</h2><span class="count">${incs.length} from ${fmt(S.data.meta.alerts)} malicious flows</span></div>`;
  if (!incs.length) {
    html += `<div class="empty"><b>No incidents</b>No group of alerts was large enough to form an incident. Inject an attack with Change scenario, or lower the threshold.</div>`;
  }
  html += incs.map((i) => `
    <button class="inc ${live.has(i.id) ? "active" : ""}" data-inc="${esc(i.id)}">
      <span class="rail" style="background:${COLOR[i.cls]}"></span>
      <span><span class="t">${withAddr(i.title)}</span>${live.has(i.id) ? '<span class="live">in this window</span>' : ""}
      <span class="m" style="display:block">${esc(i.id)}, ${esc(i.severity)}, ${fmt(i.flows)} flows, ${mmss(i.start)} to ${mmss(i.end)}</span></span>
    </button>`).join("");
  html += truthBlock() + statusBlock();
  side.innerHTML = html;
  side.querySelectorAll("[data-inc]").forEach((b) => b.addEventListener("click", () => openIncident(b.dataset.inc)));
}

function truthBlock() {
  const t = S.data.truth;
  if (!t) return "";
  const rows = t.rows.filter((r) => r.cls !== "BENIGN").map((r) => `<tr><td><span class="dot" style="background:${COLOR[r.cls]}"></span>${NICE[r.cls]}</td>
    <td>${pct(r.recall, 1)}</td><td>${pct(r.precision, 1)}</td><td>${r.delay == null ? "missed" : `${r.delay.toFixed(0)} s`}</td></tr>`).join("");
  return `<div class="block truthcard"><h3>Checked against the simulator's ground truth</h3>
    <p>Caught <b>${pct(t.detection_rate)}</b> of attack flows with <b>${pct(t.fpr, 2)}</b> of benign flows falsely flagged.</p>
    <table><tr><th>Attack</th><th>Caught</th><th>Precision</th><th>First alert</th></tr>${rows}</table></div>`;
}

function statusBlock() {
  const m = S.data.meta;
  const d = m.drift;
  const drift = d == null ? "not enough benign traffic to measure" : d < 0.1 ? `stable (PSI ${d.toFixed(2)})` : d < 0.25 ? `moderate (PSI ${d.toFixed(2)})` : `high (PSI ${d.toFixed(2)}), the model is outside its training conditions`;
  return `<div class="block"><h3>Sensor</h3><dl class="facts">
    <dt>Model</dt><dd>${esc(m.model)}</dd>
    <dt>Speed</dt><dd>${m.latency == null ? "n/a" : `${m.latency.toFixed(1)} ms`} per ${m.window_s}-second window</dd>
    <dt>Traffic drift</dt><dd>${drift}</dd>
    <dt>Capture</dt><dd>${fmt(m.flows)} flows, ${fmt(m.hosts)} hosts</dd></dl></div>`;
}

function openIncident(id) {
  const inc = S.data.incidents.find((i) => i.id === id);
  if (!inc) return;
  setPlaying(false);
  S.panel = { mode: "incident", id };
  const hosts = new Set(inc.hosts);
  const links = new Map();
  (S.data.graphs[inc.w0] || { links: [], nodes: [] }).links.forEach(([s, t, c]) => {
    const g = S.data.graphs[inc.w0];
    const a = g.nodes[s][0], b = g.nodes[t][0];
    if (CLASSES[c] === inc.cls && (hosts.has(a) || hosts.has(b))) links.set(`${a}>${b}`, 0.6);
  });
  S.highlight = { win: inc.w0, target: null, links, nodes: hosts };
  gotoWindow(inc.w0);
  renderSide();
}

function renderIncident(side) {
  const inc = S.data.incidents.find((i) => i.id === S.panel.id);
  if (!inc) { S.panel = { mode: "list" }; return renderSide(); }
  const rules = S.rulesFmt === "iptables" ? inc.rules : inc.nft;
  side.innerHTML = `
    <div class="side-head"><button class="back" id="back">All incidents</button><span class="count">${esc(inc.id)}</span></div>
    <div class="block">
      <div class="verdict"><span class="swatch" style="background:${COLOR[inc.cls]}"></span><span class="cls">${NICE[inc.cls]}</span>
        <span class="pct">${pct(inc.conf, 0)}<small>mean confidence</small></span></div>
      <p style="margin-top:10px;font-size:15px;font-weight:600">${withAddr(inc.title).replace(/class="addr"/g, 'class="mono"')}</p>
      <dl class="facts">
        <dt>Severity</dt><dd>${esc(inc.severity)}</dd>
        <dt>When</dt><dd>${mmss(inc.start)} to ${mmss(inc.end)}</dd>
        <dt>Flows</dt><dd>${fmt(inc.flows)}</dd>
        <dt>Sources</dt><dd>${fmt(inc.sources)}</dd>
        <dt>Targets</dt><dd>${fmt(inc.targets)}</dd>
        <dt>Ports</dt><dd>${inc.ports.join(", ")}</dd>
        <dt>MITRE ATT&amp;CK</dt><dd>${esc(inc.mitre)}</dd>
      </dl>
    </div>
    <div class="block"><h3>Recommended response</h3><p>${esc(inc.response)}</p></div>
    <div class="block"><h3>Suggested firewall rules, review before applying</h3>
      <div class="seg" id="rulesSeg"><button data-f="iptables" class="${S.rulesFmt === "iptables" ? "on" : ""}">iptables</button><button data-f="nftables" class="${S.rulesFmt === "nftables" ? "on" : ""}">nftables</button></div>
      <pre class="rules">${esc(rules.join("\n"))}</pre>
      <div class="row-actions"><button class="btn ghost" id="dlRules">Download as script</button></div>
    </div>
    <div class="block samples"><h3>Flows to investigate</h3>
      ${inc.samples.map((s) => `<button data-row="${s.row}"><span>${esc(s.src)} to ${esc(s.dst)}</span><span>${pct(s.conf, 0)}</span></button>`).join("")}
    </div>`;
  $("#back").addEventListener("click", () => { S.panel = { mode: "list" }; S.highlight = null; M.applyHighlight(); renderSide(); });
  side.querySelectorAll("#rulesSeg button").forEach((b) => b.addEventListener("click", () => { S.rulesFmt = b.dataset.f; renderSide(); }));
  $("#dlRules").addEventListener("click", () => download(`${inc.id}_${S.rulesFmt}.sh`, `#!/bin/sh\n${rules.join("\n")}\n`));
  side.querySelectorAll("[data-row]").forEach((b) => b.addEventListener("click", () => openExplain(+b.dataset.row, { from: { mode: "incident", id: inc.id } })));
}

function openExplain(row, { from } = {}) {
  setPlaying(false);
  S.panel = { mode: "explain", row, from: from || { mode: "list" } };
  S.attrMethod = null;
  const ex = S.data.explanation;
  if (ex && ex.row === row) {
    S.highlight = explanationHighlight(ex);
    M.applyHighlight();
  } else {
    request("explain", { row, shap: false });
  }
  renderSide();
}

function explanationHighlight(ex) {
  const links = new Map();
  const nodes = new Set([ex.flow.src_ip, ex.flow.dst_ip]);
  const st = ex.structure;
  if (st && st.top_edges) st.top_edges.forEach((e) => { links.set(`${e.src}>${e.dst}`, e.importance); nodes.add(e.src); nodes.add(e.dst); });
  const target = `${ex.flow.src_ip}>${ex.flow.dst_ip}`;
  links.set(target, 1);
  return { win: ex.flow.window, target, links, nodes };
}

function renderExplain(side) {
  const ex = S.data.explanation;
  const back = S.panel.from && S.panel.from.mode === "incident" ? `Back to ${esc(S.panel.from.id)}` : "All incidents";
  let html = `<div class="side-head"><button class="back" id="back">${back}</button><span class="count">Case file</span></div>`;
  if (!ex || ex.row !== S.panel.row) {
    html += `<div class="block"><p style="color:var(--ink-2)">Explaining this flow: attributions, neighbouring flows and counterfactuals.</p>
      <div class="skeleton" style="width:70%"></div><div class="skeleton"></div><div class="skeleton" style="width:85%"></div><div class="skeleton" style="width:60%"></div></div>`;
    side.innerHTML = html;
    bindBack();
    return;
  }
  const p = ex.prediction, f = ex.flow, cls = p.label;
  const truth = S.data.truth && f.true_label && f.true_label !== "UNKNOWN"
    ? `<div class="truth ${f.true_label === cls ? "ok" : "bad"}">Ground truth: ${NICE[f.true_label] || esc(f.true_label)}${f.true_label === cls ? ", correct" : ", the model is wrong here"}</div>` : "";
  html += `<div class="block">
      <div class="verdict"><span class="swatch" style="background:${COLOR[cls]}"></span><span class="cls">${NICE[cls]}</span>
        <span class="pct">${pct(p.confidence, 0)}<small>confidence</small></span></div>
      <div class="tuple">${esc(f.src_ip)}:${f.src_port} to ${esc(f.dst_ip)}:${f.dst_port} ${esc(f.protocol)}</div>${truth}
    </div>`;
  if (cls === "BENIGN") {
    html += `<div class="block"><p>The model considers this flow normal traffic.</p></div>`;
  }
  if (ex.evidence && ex.evidence.length) {
    html += `<div class="block"><h3>Why it was flagged</h3>${ex.evidence.map((e) => `<div class="ev"><b>${esc(e.title)}</b><span>${withAddr(e.text).replace(/class="addr"/g, 'class="mono"')}</span></div>`).join("")}</div>`;
  }
  const methods = Object.keys(ex.features || {});
  const method = S.attrMethod && methods.includes(S.attrMethod) ? S.attrMethod : methods[0];
  if (method) {
    const waitingShap = S.panel.shapPending && !methods.includes("SHAP");
    html += `<div class="block"><h3>What the model looked at</h3>
      <div class="seg" id="attrSeg">${methods.map((m) => `<button data-m="${esc(m)}" class="${m === method ? "on" : ""}">${esc(m)}</button>`).join("")}
      ${methods.includes("SHAP") ? "" : `<button id="shapBtn">${waitingShap ? "Computing SHAP" : "Add SHAP"}</button>`}</div>
      <div id="attrChart"></div>
      <div class="caption">Bars in the attack colour push towards ${NICE[cls]}, grey bars push away. ${ex.timings_ms && ex.timings_ms[method] ? `Computed in ${Math.round(ex.timings_ms[method])} ms.` : ""}</div></div>`;
  }
  const st = ex.structure;
  if (st && st.neighbour_flows) {
    const share = Math.max(st.share_same_destination, st.share_same_source);
    const where = st.share_same_destination >= st.share_same_source ? "flows towards the same destination" : "other flows from the same source";
    html += `<div class="block"><h3>Neighbouring flows that mattered</h3>
      <p>${st.important_flows ? `${fmt(st.important_flows)} of ${fmt(st.neighbour_flows)} neighbouring flows carry 80% of the structural evidence; ${pct(share, 0)} of it comes from ${where}. They are highlighted on the map.`
        : `None of the ${fmt(st.neighbour_flows)} neighbouring flows adds evidence; the flow's own properties and host statistics decide.`}</p>
      ${(st.top_edges || []).slice(0, 5).map((e) => `<div class="tuple" style="display:flex;justify-content:space-between;margin-top:4px"><span>${esc(e.src)} to ${esc(e.dst)}:${e.dst_port}</span><span style="display:inline-block;width:${Math.round(40 * e.importance)}px;height:6px;background:var(--ink);border-radius:2px;margin-top:5px"></span></div>`).join("")}
      <div class="caption">${esc(st.method)}</div></div>`;
  }
  const cf = ex.counterfactual_features, scf = ex.counterfactual_structure;
  if (cf || scf) {
    let items = "";
    if (cf && cf.found && cf.text) items += `<div class="cf">It would be classified benign if its ${esc(cf.text)}.</div>`;
    else if (cf) items += `<div class="cf">No realistic change to this flow's own properties makes it benign; the decision rests on its context.</div>`;
    if (scf && scf.found) items += `<div class="cf">Removing the ${fmt(scf.removed)} most important of ${fmt(scf.total_neighbours)} neighbouring flows would make it look benign.</div>`;
    html += `<div class="block"><h3>What would change the decision</h3>${items}</div>`;
  }
  if (cls !== "BENIGN") {
    html += `<div class="block"><h3>Response</h3><p>${esc(ex.response)}</p><p style="color:var(--ink-2);font-size:13px">MITRE ATT&amp;CK ${esc(ex.mitre)}</p>
      <div class="row-actions"><button class="btn ghost" id="dlReport">Download case file</button></div></div>`;
  }
  side.innerHTML = html;
  bindBack();
  if (method) attrChart(ex.features[method], cls);
  side.querySelectorAll("#attrSeg [data-m]").forEach((b) => b.addEventListener("click", () => { S.attrMethod = b.dataset.m; renderSide(); }));
  const shapBtn = $("#shapBtn");
  if (shapBtn) shapBtn.addEventListener("click", () => {
    if (S.panel.shapPending) return;
    S.panel.shapPending = true;
    S.attrMethod = "SHAP";
    request("explain", { row: S.panel.row, shap: true });
    renderSide();
  });
  const dl = $("#dlReport");
  if (dl) dl.addEventListener("click", () => download(`case_${f.src_ip}_${f.dst_ip}.md`, caseFile(ex)));
}

function bindBack() {
  $("#back").addEventListener("click", () => {
    const from = S.panel.from || { mode: "list" };
    S.highlight = null;
    if (from.mode === "incident") openIncident(from.id);
    else { S.panel = { mode: "list" }; M.applyHighlight(); renderSide(); }
  });
}

function attrChart(feats, cls) {
  const el = document.getElementById("attrChart");
  if (!el || !feats || !feats.length) return;
  const W = el.getBoundingClientRect().width || 356;
  const rowH = 36, H = feats.length * rowH;
  const max = d3.max(feats, (d) => Math.abs(d.attribution)) || 1;
  const mid = W * 0.5;
  const sx = d3.scaleLinear().domain([0, max]).range([0, W * 0.5 - 2]);
  const svg = d3.select(el).append("svg").attr("class", "attr").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);
  svg.append("line").attr("x1", mid).attr("x2", mid).attr("y1", 0).attr("y2", H).attr("stroke", "#D3D9D5");
  feats.forEach((d, i) => {
    const y = i * rowH;
    const [scope, label] = d.label.includes(": ") ? d.label.split(": ") : ["", d.label];
    const t = svg.append("text").attr("x", 0).attr("y", y + 12);
    t.append("tspan").attr("class", "scope").text(scope ? `${scope}: ` : "");
    t.append("tspan").text(label.length > 46 ? `${label.slice(0, 45)}…` : label);
    t.append("tspan").attr("class", "val").text(` = ${d.value}`);
    const w = sx(Math.abs(d.attribution));
    svg.append("rect").attr("x", d.attribution >= 0 ? mid : mid - w).attr("y", y + 18).attr("width", Math.max(1, w)).attr("height", 9)
      .attr("rx", 2).attr("fill", d.attribution >= 0 ? COLOR[cls] : "#B4BEC6");
  });
}

function caseFile(ex) {
  const f = ex.flow, p = ex.prediction;
  const feats = Object.entries(ex.features || {}).map(([m, fs]) => `### ${m}\n${fs.slice(0, 6).map((x) => `- ${x.label} = ${x.value} (${x.attribution >= 0 ? "+" : ""}${x.attribution.toFixed(4)})`).join("\n")}`).join("\n\n");
  return `# XGNN-IDS case file\n\nModel: ${S.data.meta.model}\n\nFlow: ${f.src_ip}:${f.src_port} -> ${f.dst_ip}:${f.dst_port} ${f.protocol} at t=${mmss(f.ts)}\n\n` +
    `Prediction: ${p.label} (${pct(p.confidence, 1)})\n\n${ex.narrative || ""}\n\n## Feature attribution\n\n${feats}\n`;
}

function download(name, text) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type: "text/plain" }));
  a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
}

/* ------------------------------------------------------------------ scenario sheet */
function openSheet(on) {
  $("#sheet").hidden = !on; $("#sheetBackdrop").hidden = !on;
  if (on) syncSheet();
}
function syncSheet() {
  const sc = S.scenario;
  document.querySelectorAll('input[name=profile]').forEach((r) => { r.checked = r.value === sc.profile; });
  $("#attackChips").innerHTML = ATTACKS.map((a) => `<button class="chip ${sc.attacks.includes(a) ? "on" : ""}" data-a="${a}"><i style="background:${COLOR[a]}"></i>${NICE[a]}</button>`).join("");
  document.querySelectorAll("#attackChips .chip").forEach((c) => c.addEventListener("click", () => {
    const a = c.dataset.a;
    sc.attacks = sc.attacks.includes(a) ? sc.attacks.filter((x) => x !== a) : [...sc.attacks, a];
    c.classList.toggle("on");
  }));
  document.querySelectorAll("#lengthSeg button").forEach((b) => b.classList.toggle("on", +b.dataset.min === sc.minutes));
  $("#intensity").value = sc.intensity;
  $("#intensityVal").textContent = `${(+sc.intensity).toFixed(1)}x`;
}
$("#openScenario").addEventListener("click", () => openSheet(true));
$("#closeScenario").addEventListener("click", () => openSheet(false));
$("#sheetBackdrop").addEventListener("click", () => openSheet(false));
document.querySelectorAll('input[name=profile]').forEach((r) => r.addEventListener("change", () => { S.scenario.profile = r.value; }));
document.querySelectorAll("#lengthSeg button").forEach((b) => b.addEventListener("click", () => {
  S.scenario.minutes = +b.dataset.min;
  document.querySelectorAll("#lengthSeg button").forEach((x) => x.classList.toggle("on", x === b));
}));
$("#intensity").addEventListener("input", (e) => { S.scenario.intensity = +e.target.value; $("#intensityVal").textContent = `${(+e.target.value).toFixed(1)}x`; });
function runScenario() {
  openSheet(false);
  busy(true, "Simulating traffic, building graphs and classifying flows");
  request("simulate", { ...S.scenario });
}
$("#runScenario").addEventListener("click", runScenario);
$("#randomSeed").addEventListener("click", () => { S.scenario.seed = Math.floor(Math.random() * 9000) + 1; runScenario(); });
$("#fileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (file.size > 25 * 1024 * 1024) { showError("That file is larger than 25 MB. Trim it, or run python -m xgnn_ids detect on it locally."); return; }
  openSheet(false);
  busy(true, `Reading ${file.name}`);
  const reader = new FileReader();
  reader.onload = () => request("upload", { name: file.name, data: String(reader.result).split(",")[1] || "" });
  reader.readAsDataURL(file);
  e.target.value = "";
});

/* ------------------------------------------------------------------ evaluation page */
function renderEval() {
  const E = S.data && S.data.eval;
  const el = $("#view-eval");
  if (S.evalRendered === E && el.innerHTML) return;
  S.evalRendered = E;
  if (!E) {
    el.innerHTML = `<div class="inner"><h1>How well it works</h1><p class="lede">No evaluation results were found. Run <span class="mono">python -m xgnn_ids train</span> to produce them.</p></div>`;
    return;
  }
  const byKey = Object.fromEntries(E.models.map((m) => [m.key, m]));
  const graphRows = E.models.filter((m) => GRAPH_MODELS.has(m.key));
  const flatRows = E.models.filter((m) => ["mlp", "rf", "hgb"].includes(m.key));
  const best = byKey[E.default];
  const bestFlat = flatRows.reduce((a, b) => (b.macro_f1 > (a ? a.macro_f1 : -1) ? b : a), null);
  const graphKind = (k) => (GRAPH_MODELS.has(k) ? "learned" : k === "rf_graphstats" ? "hand-made" : "none");
  const maxLat = d3.max(E.models, (m) => m.latency) || 1;
  const modelTable = `<table><thead><tr><th>Model</th><th>Graph</th><th>Macro-F1</th><th>DDoS F1</th><th>False alarms</th><th>Time per window</th></tr></thead><tbody>
    ${[...E.models].sort((a, b) => b.macro_f1 - a.macro_f1).map((m) => `<tr class="${m.key === E.default ? "hl" : ""}">
      <td>${esc(m.name)}${m.key === E.default ? '<span class="tag">in use</span>' : ""}</td><td>${graphKind(m.key)}</td>
      <td><span class="barcell">${num(m.macro_f1)}<i class="${GRAPH_MODELS.has(m.key) ? "" : "light"}" style="width:${Math.max(2, (m.macro_f1 - 0.6) / 0.4 * 90)}px"></i></span></td>
      <td>${num(m.ddos_f1)}</td><td>${pct(m.fpr, 2)}</td>
      <td><span class="barcell">${m.latency.toFixed(1)} ms<i class="light" style="width:${Math.max(2, m.latency / maxLat * 60)}px"></i></span></td></tr>`).join("")}
    </tbody></table>`;

  const heatModels = ["mlp", "rf", "rf_graphstats", "gcn", "gat", "egraphsage"].filter((k) => byKey[k]);
  // cubic ramp: separates 0.85 from 0.95 from 1.00, where the interesting differences are
  const heatT = (v) => Math.pow(Math.max(0, Math.min(1, (v - 0.7) / 0.3)), 3);
  const heatColor = (v) => d3.interpolateRgb("#F1F4F1", "#0F1720")(heatT(v));
  const heat = `<div class="heat" style="grid-template-columns: 200px repeat(${CLASSES.length}, 1fr)">
      <div class="h"></div>${CLASSES.map((c) => `<div class="h">${NICE[c]}</div>`).join("")}
      ${heatModels.map((k) => `<div class="m">${esc(byKey[k].name)}</div>${CLASSES.map((c) => {
        const v = byKey[k].per_class[c];
        return `<div style="background:${heatColor(v ?? 0)};color:${heatT(v ?? 0) > 0.45 ? "#fff" : "#0F1720"}">${v == null ? "" : v.toFixed(2)}</div>`;
      }).join("")}`).join("")}</div>`;

  let advTake = "";
  if (E.adversarial) {
    const at = (k) => { const m = E.adversarial.find((x) => x.key === k); const p = m && m.points.find((q) => q[0] === 1); return p ? p[1] : null; };
    advTake = `At a budget of one standard deviation the ${esc(byKey[E.default] ? byKey[E.default].name : "graph model")} still catches ${pct(at(E.default) ?? at("gcn"), 0)} of attacks; the flow-only network catches ${pct(at("mlp"), 0)}.`;
  }
  const zd = E.zero_day || [];
  const zdCaught = zd.filter((r) => r.detected >= 0.9).length;

  const fs = E.few_shot;
  const fsMain = fs && (fs.models.find((m) => m.key === E.default) || fs.models[0]);

  const xf = E.xai ? E.xai.features : [], xs = E.xai ? E.xai.structure : [];
  const xTable = (rows, cols) => `<table><thead><tr><th>Method</th>${cols.map((c) => `<th>${c[1]}</th>`).join("")}</tr></thead><tbody>
    ${rows.map((r) => `<tr class="${r.method === "Random" ? "" : ""}"><td>${esc(r.method)}</td>${cols.map(([k, , f]) => `<td>${r[k] == null ? "n/a" : f(r[k])}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  const xcols = [["fid_plus", "Fidelity+", (v) => v.toFixed(2)], ["fid_minus", "Fidelity-", (v) => v.toFixed(2)],
    ["stability", "Stability", (v) => v.toFixed(2)], ["runtime", "Time", (v) => `${Math.round(v)} ms`]];
  const ig = xf.find((r) => r.method === "Integrated Gradients"), rnd = xf.find((r) => r.method === "Random");
  const eig = xs.find((r) => r.method === "Edge Integrated Gradients"), gx = xs.find((r) => r.method === "GNNExplainer");

  el.innerHTML = `<div class="inner">
    <h1>How well it works</h1>
    <p class="lede">Every number on this page was measured on traffic recorded after the model's training period, so the model never saw any of it while learning.</p>
    <p class="src">Data: ${esc(E.source)}, ${fmt(E.flows)} flows in ${E.windows} graph windows. Reproduce with <span class="mono">python -m xgnn_ids train</span>.</p>

    <section><header><h2>Detection on future traffic</h2>
      <p>Macro-F1 weighs all eight classes equally, so a rare web attack counts as much as a DDoS.</p>
      <p class="take">Reading the graph lifts macro-F1 from ${num(bestFlat && bestFlat.macro_f1)} to ${num(best && best.macro_f1)} and cuts false alarms from ${pct(bestFlat && bestFlat.fpr, 1)} to ${pct(best && best.fpr, 2)}.</p></header>
      <div>${modelTable}</div></section>

    <section><header><h2>Per attack type</h2><p>F1 per class. Without the graph, DDoS, DoS and botnet traffic are the hardest: single flows look normal, the pattern only shows across hosts.</p></header>
      <div>${heat}</div></section>

    ${E.adversarial ? `<section><header><h2>When the attacker adapts</h2><p>A white-box attacker tunes the timing, volume and packet sizes of their own flows to look benign. The budget is in standard deviations of each feature.</p><p class="take">${advTake}</p></header>
      <div class="chart" id="advChart"></div></section>` : ""}

    ${zd.length ? `<section><header><h2>Attacks it was never taught</h2><p>Each family is removed from the training labels in turn. A flow counts as caught if it is flagged as any attack or the model is unsure.</p>
      <p class="take">${zdCaught} of ${zd.length} unseen families are still caught. Low-volume botnet beaconing is the exception.</p></header>
      <div><table><thead><tr><th>Unseen family</th><th>Caught</th><th>As attack</th><th>Unsure</th><th>Usually labelled</th><th>False alarms</th></tr></thead><tbody>
      ${zd.map((r) => `<tr><td><span class="dot" style="background:${COLOR[r.cls]}"></span>${NICE[r.cls]}</td><td><span class="barcell">${pct(r.detected, 0)}<i style="width:${Math.max(2, r.detected * 70)}px;background:${COLOR[r.cls]}"></i></span></td><td>${pct(r.attack, 0)}</td><td>${pct(r.uncertain, 0)}</td><td>${NICE[r.label] || esc(r.label)}</td><td>${pct(r.false_alarm, 1)}</td></tr>`).join("")}
      </tbody></table></div></section>` : ""}

    ${E.cross ? `<section><header><h2>A network it has never seen</h2><p>Trained on an office LAN, tested on an IoT network: different addresses, services and attack variants. Periodic sensor telemetry looks like botnet beaconing to an office-trained model.</p>
      <p class="take">The drift monitor flags the change (PSI ${num(E.cross.psi_out, 2)} vs ${num(E.cross.psi_in, 3)} at home).${fsMain ? ` About ${fs.minutes} minutes of labelled local traffic cut false alarms from ${pct(fsMain.fpr_before, 0)} to ${pct(fsMain.fpr_after, 2)}.` : ""}</p></header>
      <div>${fs ? `<div class="chart" id="fewChart"></div>` : ""}
      <table style="margin-top:18px"><thead><tr><th>Model, no retraining</th><th>Home macro-F1</th><th>IoT macro-F1</th><th>IoT false alarms</th></tr></thead><tbody>
      ${E.cross.models.map((m) => `<tr><td>${esc(m.name)}</td><td>${num(m.in)}</td><td>${num(m.out)}</td><td>${pct(m.fpr, 1)}</td></tr>`).join("")}</tbody></table></div></section>` : ""}

    ${E.xai ? `<section><header><h2>Explanations, measured</h2><p>Fidelity+ is how much the prediction (in log-odds) drops when the highlighted items are removed; higher means they really mattered. Fidelity- is the drop when only they are kept; lower is better. Stability is the overlap of the top items under small input noise.</p>
      <p class="take">Integrated Gradients is the most faithful (${num(ig && ig.fid_plus, 1)} vs ${num(rnd && rnd.fid_plus, 1)} for random). For neighbouring flows, edge Integrated Gradients scores ${num(eig && eig.fid_plus, 1)}; GNNExplainer scored ${num(gx && gx.fid_plus, 1)}, no better than random, so it is kept only for comparison.</p></header>
      <div><h3 style="margin:0 0 8px;font-size:14px">Which properties of the flow and its hosts mattered</h3>${xTable(xf, xcols)}
      <h3 style="margin:22px 0 8px;font-size:14px">Which neighbouring flows mattered</h3>${xTable(xs, xcols)}
      <p class="caption">${E.xai.n} correctly detected attack flows, top 5 items (features) or top 20% (neighbouring flows).</p></div></section>` : ""}

    <div class="note">The traffic comes from the project's built-in network simulator, so these numbers are not comparable with papers that report on CIC-IDS2017 directly. The same pipeline trains and evaluates on CIC-IDS2017, CICDDoS2019 and NetFlow datasets.</div>
  </div>`;
  if (E.adversarial) advChart(E.adversarial);
  if (fs) fewChart(fs);
}

function advChart(series) {
  const el = document.getElementById("advChart");
  const W = el.getBoundingClientRect().width || 600, H = 300, m = { l: 40, r: 170, t: 10, b: 36 };
  const x = d3.scaleLinear().domain([0, d3.max(series, (s) => d3.max(s.points, (p) => p[0]))]).range([m.l, W - m.r]);
  const y = d3.scaleLinear().domain([0.4, 1]).range([H - m.b, m.t]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);
  svg.append("g").attr("class", "grid").selectAll("line").data(y.ticks(6)).join("line")
    .attr("x1", m.l).attr("x2", W - m.r).attr("y1", y).attr("y2", y);
  svg.append("g").attr("transform", `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(5).tickSize(0).tickPadding(8)).call((g) => g.select(".domain").remove());
  svg.append("g").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(6).tickFormat(d3.format(".0%")).tickSize(0).tickPadding(6)).call((g) => g.select(".domain").remove());
  svg.append("text").attr("x", (m.l + W - m.r) / 2).attr("y", H - 2).attr("text-anchor", "middle").text("attacker budget (standard deviations)");
  const style = { mlp: ["#8A96A1", "5 4"], gcn: ["#0F1720", null], gat: ["#4B5865", null], egraphsage: ["#4B5865", "2 3"] };
  const ends = [];
  series.forEach((s) => {
    const [c, dash] = style[s.key] || ["#4B5865", null];
    svg.append("path").attr("d", d3.line().x((p) => x(p[0])).y((p) => y(p[1]))(s.points)).attr("fill", "none")
      .attr("stroke", c).attr("stroke-width", s.key === "gcn" ? 2.6 : 1.6).attr("stroke-dasharray", dash);
    svg.selectAll(null).data(s.points).join("circle").attr("cx", (p) => x(p[0])).attr("cy", (p) => y(p[1])).attr("r", 2.6).attr("fill", c);
    const last = s.points[s.points.length - 1];
    ends.push({ y: y(last[1]), text: `${s.name}  ${pct(last[1], 0)}`, c });
  });
  ends.sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 14) ends[i].y = ends[i - 1].y + 14;
  ends.forEach((e) => svg.append("text").attr("x", W - m.r + 8).attr("y", e.y + 4).attr("fill", e.c).style("fill", e.c).text(e.text));
}

function fewChart(fs) {
  const el = document.getElementById("fewChart");
  const W = el.getBoundingClientRect().width || 600, rowH = 46, m = { l: 210, r: 60, t: 24, b: 10 };
  const H = m.t + fs.models.length * rowH + m.b;
  const x = d3.scaleLinear().domain([0, 1]).range([m.l, W - m.r]);
  const svg = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`).attr("height", H);
  svg.append("text").attr("x", m.l).attr("y", 12).text("IoT false-alarm rate, before and after adapting");
  fs.models.forEach((r, i) => {
    const y = m.t + i * rowH + 18;
    svg.append("text").attr("x", 0).attr("y", y + 4).style("fill", "#0F1720").text(r.name);
    svg.append("line").attr("x1", x(r.fpr_after)).attr("x2", x(r.fpr_before)).attr("y1", y).attr("y2", y).attr("stroke", "#8A96A1").attr("stroke-width", 2);
    svg.append("circle").attr("cx", x(r.fpr_before)).attr("cy", y).attr("r", 6).attr("fill", "#F8FAF8").attr("stroke", "#0F1720").attr("stroke-width", 1.6);
    svg.append("circle").attr("cx", x(r.fpr_after)).attr("cy", y).attr("r", 6).attr("fill", "#0F1720");
    svg.append("text").attr("x", x(r.fpr_before) + 10).attr("y", y + 4).text(pct(r.fpr_before, 0));
    svg.append("text").attr("x", x(r.fpr_after) - 10).attr("y", y - 10).attr("text-anchor", "start").style("fill", "#0F1720").text(pct(r.fpr_after, 2));
  });
}

/* ------------------------------------------------------------------ method page */
function renderMethod() {
  if (S.methodRendered) return;
  S.methodRendered = true;
  $("#view-method").innerHTML = `<div class="inner">
    <h1>How it works</h1>
    <p class="lede">Attacks are patterns between machines, not properties of a single connection. So the traffic is turned into a graph and read by a graph neural network that sees each flow together with its neighbourhood.</p>
    <p class="src">BCSE308P Computer Networks project, Vellore Institute of Technology. Guided by Dr. Aaseegha M D (SCORE). Sai Chandra Reddy (24BYB0011), N Rishi Pranav (24BYB0082).</p>

    <section class="wide"><header><h2>The pipeline</h2><p>The same five steps run on a live network interface, a packet capture, or a public dataset.</p></header>
      <ol class="steps">
        <li><b>Collect flows</b><span>Packets are grouped into flows (one connection each), or flow records are read from CICFlowMeter and NetFlow files.</span></li>
        <li><b>Build a graph</b><span>Every 20 seconds: hosts become nodes, flows become directed edges carrying 27 features; each host gets 11 statistics.</span></li>
        <li><b>Classify every flow</b><span>A graph neural network (GCN, GAT or E-GraphSAGE) labels each edge as benign or one of seven attack families.</span></li>
        <li><b>Explain</b><span>Integrated Gradients, SHAP and counterfactuals say why; edge attributions say which neighbouring flows mattered.</span></li>
        <li><b>Respond</b><span>Alerts are grouped into incidents with MITRE ATT&amp;CK techniques and suggested firewall rules.</span></li>
      </ol></section>

    <section><header><h2>Why a graph</h2><p>Each of these looks harmless one flow at a time. The shape gives them away.</p></header>
      <div class="motifs">
        <div class="motif"><svg id="mot-ddos" viewBox="0 0 300 150"></svg><b>DDoS</b><span>Hundreds of sources converge on one server within seconds.</span></div>
        <div class="motif"><svg id="mot-scan" viewBox="0 0 300 150"></svg><b>Port scan</b><span>One source touches hundreds of ports on one host, most go unanswered.</span></div>
        <div class="motif"><svg id="mot-bot" viewBox="0 0 300 150"></svg><b>Botnet</b><span>Several internal machines call the same rare address on a steady beat.</span></div>
      </div></section>

    <section><header><h2>Gaps from the literature review, and what this project does</h2></header>
      <div><table><thead><tr><th>Gap in the reviewed studies</th><th>What this project does</th></tr></thead><tbody>
        ${[["Single-dataset evaluation", "Cross-network test with drift monitoring and few-shot adaptation; loaders for CIC-IDS2017, CICDDoS2019 and NetFlow"],
          ["Explanation quality not measured", "Fidelity, stability and runtime for every explainer against a random baseline"],
          ["Real-time cost unreported", "Latency per window and throughput for every model; live capture mode"],
          ["Adversarial robustness", "White-box evasion test on attacker-controlled features"],
          ["Unknown attacks", "Leave-one-attack-out test with uncertainty flagging"],
          ["Temporal leakage", "Chronological train, validation and test split"],
          ["Alerts are hard to act on", "Incidents, MITRE ATT&CK mapping, firewall rules, counterfactual explanations"]]
          .map(([a, b]) => `<tr><td style="text-align:left">${a}</td><td style="text-align:left">${b}</td></tr>`).join("")}
      </tbody></table></div></section>
  </div>`;
  drawMotifs();
}

function drawMotifs() {
  const ink = "#0F1720";
  const ddos = d3.select("#mot-ddos");
  const cx = 150, cy = 75;
  d3.range(46).forEach((i) => {
    const a = (i / 46) * Math.PI * 2, r = 62 + (i % 3) * 4;
    const x = cx + Math.cos(a) * r * 1.7, y = cy + Math.sin(a) * r * 0.95;
    ddos.append("line").attr("x1", x).attr("y1", y).attr("x2", cx).attr("y2", cy).attr("stroke", COLOR.DDoS).attr("stroke-width", 0.8).attr("opacity", 0.75);
    ddos.append("circle").attr("cx", x).attr("cy", y).attr("r", 2.3).attr("fill", "#F8FAF8").attr("stroke", ink);
  });
  ddos.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 9).attr("fill", ink).attr("stroke", COLOR.DDoS).attr("stroke-width", 2.5);

  const scan = d3.select("#mot-scan");
  scan.append("circle").attr("cx", 40).attr("cy", 75).attr("r", 6).attr("fill", "#F8FAF8").attr("stroke", COLOR.PortScan).attr("stroke-width", 2.5);
  scan.append("circle").attr("cx", 210).attr("cy", 75).attr("r", 9).attr("fill", ink);
  d3.range(28).forEach((i) => {
    const a = -Math.PI / 2 + (i / 27) * Math.PI, x = 210 + Math.cos(a) * 48, y = 75 + Math.sin(a) * 58;
    scan.append("path").attr("d", `M46,75 Q130,${75 + (y - 75) * 0.5} ${x},${y}`).attr("fill", "none").attr("stroke", COLOR.PortScan).attr("stroke-width", 0.8);
    scan.append("rect").attr("x", x - 2).attr("y", y - 2).attr("width", 4).attr("height", 4).attr("fill", i % 5 === 0 ? ink : "#D3D9D5");
  });

  const bot = d3.select("#mot-bot");
  bot.append("ellipse").attr("cx", 95).attr("cy", 75).attr("rx", 80).attr("ry", 62).attr("fill", "#F8FAF8").attr("stroke", "#8A96A1").attr("stroke-dasharray", "4 4");
  const c2 = [262, 75];
  [[60, 40], [95, 105], [120, 50], [55, 95], [130, 92]].forEach(([x, y], i) => {
    bot.append("line").attr("x1", x).attr("y1", y).attr("x2", c2[0]).attr("y2", c2[1]).attr("stroke", COLOR.Botnet).attr("stroke-width", 1.2).attr("stroke-dasharray", "2 6");
    bot.append("circle").attr("cx", x).attr("cy", y).attr("r", 5).attr("fill", ink).attr("stroke", COLOR.Botnet).attr("stroke-width", 2);
  });
  bot.append("circle").attr("cx", c2[0]).attr("cy", c2[1]).attr("r", 7).attr("fill", "#F8FAF8").attr("stroke", ink).attr("stroke-width", 1.5);
}

/* ------------------------------------------------------------------ boot */
M.init();
ST.ready();
ST.height(window.innerHeight || 900);

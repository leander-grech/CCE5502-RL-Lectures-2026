/* CCE5502 interactive slide deck — engine + shared widget utilities.
 * No build step, no network needed (KaTeX is vendored in assets/katex).
 *
 * Keys:  → / Space / PgDn  next      ← / PgUp  previous
 *        Home / End  first / last     N  speaker notes      P  presenter window
 *        O  overview   F  fullscreen   B  blackout   G  go to slide   T  reset timer   ?  help
 */
(function () {
  "use strict";

  // ------------------------------------------------------------------ utilities
  const U = {};
  U.$ = (sel, root = document) => root.querySelector(sel);
  U.$$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  U.el = function (tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === "style" && typeof v === "object") Object.assign(e.style, v);
      else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
      else if (k === "html") e.innerHTML = v;
      else e.setAttribute(k, v);
    }
    for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
    return e;
  };
  U.svg = function (tag, attrs, ...kids) {
    const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v);
    for (const k of kids.flat()) if (k != null) e.append(k.nodeType ? k : document.createTextNode(k));
    return e;
  };
  U.clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  // Colour semantics used in every equation (and matching CSS classes for text):
  //   \est{..} estimate being updated · \rew{..} real experience (R, G) · \nxt{..} estimate of the next state
  //   \tgt{..} TD target · \err{..} error / surprise δ
  U.KATEX = {
    trust: (ctx) => ctx.command === "\\htmlClass",
    macros: { "\\est": "\\htmlClass{c-est}{#1}", "\\rew": "\\htmlClass{c-rew}{#1}", "\\nxt": "\\htmlClass{c-next}{#1}", "\\tgt": "\\htmlClass{c-tgt}{#1}", "\\err": "\\htmlClass{c-err}{#1}" },
  };
  U.lerp = (a, b, t) => a + (b - a) * t;
  U.fmt = (x, d = 2) => (Math.abs(x) < 1e-9 ? 0 : x).toFixed(d);
  U.sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  U.css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  U.C = {
    ink: "#1d2433", muted: "#5b6475", faint: "#e6e3dc", panel: "#f3f1ec",
    blue: "#1f5fae", orange: "#d9480f", green: "#2b8a3e", red: "#c92a2a", purple: "#6741d9", gold: "#e8a100",
    teal: "#0b7285", pink: "#c2255c",
  };

  // seeded RNG (mulberry32)
  U.rng = function (seed = 1) {
    let a = seed >>> 0;
    const r = function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    r.int = (n) => Math.floor(r() * n);
    r.choice = (arr) => arr[Math.floor(r() * arr.length)];
    r.normal = () => { let u = 0, v = 0; while (u === 0) u = r(); while (v === 0) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
    r.categorical = (ps) => { let x = r(), c = 0; for (let i = 0; i < ps.length; i++) { c += ps[i]; if (x < c) return i; } return ps.length - 1; };
    return r;
  };

  // solve A x = b (Gaussian elimination with partial pivoting)
  U.solve = function (A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) continue;
      for (let r = 0; r < n; r++) if (r !== c) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((r, i) => r[n] / r[i]);
  };

  // animation loop that only runs while its slide is visible
  U.loop = function (fn) {
    let id = null, last = 0;
    const tick = (t) => { const dt = last ? Math.min(0.05, (t - last) / 1000) : 0; last = t; fn(dt, t); id = requestAnimationFrame(tick); };
    return {
      start() { if (id == null) { last = 0; id = requestAnimationFrame(tick); } },
      stop() { if (id != null) cancelAnimationFrame(id); id = null; },
      get running() { return id != null; },
    };
  };

  // ------------------------------------------------------------------ controls
  U.slider = function ({ label, min, max, step, value, fmt, oninput, width }) {
    const inp = U.el("input", { type: "range", min, max, step, value });
    if (width) inp.style.width = width + "px";
    const out = U.el("span", { class: "val" });
    const show = () => (out.textContent = fmt ? fmt(+inp.value) : inp.value);
    inp.addEventListener("input", () => { show(); oninput && oninput(+inp.value); });
    inp.addEventListener("pointerup", () => inp.blur());
    inp.addEventListener("change", () => inp.blur());
    show();
    const wrap = U.el("label", {}, label ? U.el("span", { html: label }) : null, inp, out);
    wrap.input = inp;
    wrap.set = (v) => { inp.value = v; show(); };
    wrap.get = () => +inp.value;
    return wrap;
  };
  U.button = function (label, onclick, cls = "") {
    const b = U.el("button", { class: "btn " + cls, html: label });
    b.addEventListener("click", (e) => { b.blur(); onclick(e); });
    return b;
  };
  U.toggle = function (labels, onchange, initial = 0) {
    const wrap = U.el("span", { style: { display: "inline-flex", gap: "0" } });
    const btns = labels.map((l, i) => {
      const b = U.el("button", { class: "btn", html: l, style: { borderRadius: i === 0 ? "8px 0 0 8px" : i === labels.length - 1 ? "0 8px 8px 0" : "0", marginLeft: i ? "-1.5px" : "0" } });
      b.addEventListener("click", () => { b.blur(); set(i); onchange(i); });
      wrap.append(b);
      return b;
    });
    const set = (i) => btns.forEach((b, j) => b.classList.toggle("on", i === j));
    set(initial);
    wrap.set = set;
    return wrap;
  };

  // hi-dpi canvas (the deck is CSS-scaled, so render at 2x)
  U.canvas = function (w, h, parent) {
    const c = U.el("canvas", { width: w * 2, height: h * 2 });
    c.style.width = w + "px"; c.style.height = h + "px";
    const ctx = c.getContext("2d"); ctx.scale(2, 2);
    c.W = w; c.H = h; c.ctx = ctx;
    c.pos = (e) => { const r = c.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * w, ((e.clientY - r.top) / r.height) * h]; };
    if (parent) parent.append(c);
    return c;
  };

  // ------------------------------------------------------------------ plotting
  // Minimal canvas chart: lines, points, bars, bands. Coordinates in data space.
  U.Plot = class {
    constructor(parent, w, h, opts = {}) {
      this.cv = U.canvas(w, h, parent);
      this.ctx = this.cv.ctx; this.w = w; this.h = h;
      this.o = Object.assign({ pad: [18, 16, 44, 62], xlim: [0, 1], ylim: [0, 1], xlabel: "", ylabel: "", xticks: 5, yticks: 5, grid: true, xfmt: null, yfmt: null, title: "" }, opts);
      this.series = [];
    }
    get px() { const [t, r, b, l] = this.o.pad; return { l, r: this.w - r, t, b: this.h - b }; }
    X(x) { const p = this.px, [a, b] = this.o.xlim; return p.l + ((x - a) / (b - a)) * (p.r - p.l); }
    Y(y) { const p = this.px, [a, b] = this.o.ylim; return p.b - ((y - a) / (b - a)) * (p.b - p.t); }
    ticks(a, b, n) {
      const span = b - a, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
      const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 0.5) || raw;
      const out = []; for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
      return out;
    }
    axes() {
      const c = this.ctx, p = this.px, o = this.o;
      c.clearRect(0, 0, this.w, this.h);
      c.font = "13px " + U.css("--font"); c.fillStyle = U.C.muted; c.strokeStyle = U.C.faint; c.lineWidth = 1;
      const xf = o.xfmt || ((v) => +v.toFixed(3)), yf = o.yfmt || ((v) => +v.toFixed(3));
      c.textAlign = "center"; c.textBaseline = "top";
      for (const v of this.ticks(o.xlim[0], o.xlim[1], o.xticks)) {
        const x = this.X(v);
        if (o.grid) { c.beginPath(); c.moveTo(x, p.t); c.lineTo(x, p.b); c.stroke(); }
        c.fillText(xf(v), x, p.b + 6);
      }
      c.textAlign = "right"; c.textBaseline = "middle";
      for (const v of this.ticks(o.ylim[0], o.ylim[1], o.yticks)) {
        const y = this.Y(v);
        if (o.grid) { c.beginPath(); c.moveTo(p.l, y); c.lineTo(p.r, y); c.stroke(); }
        c.fillText(yf(v), p.l - 8, y);
      }
      c.strokeStyle = U.C.muted; c.beginPath(); c.moveTo(p.l, p.t); c.lineTo(p.l, p.b); c.lineTo(p.r, p.b); c.stroke();
      c.fillStyle = U.C.ink; c.font = "15px " + U.css("--font");
      if (o.xlabel) { c.textAlign = "center"; c.textBaseline = "bottom"; c.fillText(o.xlabel, (p.l + p.r) / 2, this.h - 2); }
      if (o.ylabel) { c.save(); c.translate(14, (p.t + p.b) / 2); c.rotate(-Math.PI / 2); c.textAlign = "center"; c.textBaseline = "middle"; c.fillText(o.ylabel, 0, 0); c.restore(); }
      if (o.title) { c.textAlign = "left"; c.textBaseline = "top"; c.font = "600 15px " + U.css("--font"); c.fillText(o.title, p.l + 6, 2); }
    }
    clip(fn) { const c = this.ctx, p = this.px; c.save(); c.beginPath(); c.rect(p.l, p.t - 2, p.r - p.l, p.b - p.t + 4); c.clip(); fn(c); c.restore(); }
    line(pts, color = U.C.blue, width = 2.5, dash = null, alpha = 1) {
      if (!pts.length) return;
      this.clip((c) => {
        c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = width; c.setLineDash(dash || []); c.lineJoin = "round";
        c.beginPath(); pts.forEach(([x, y], i) => (i ? c.lineTo(this.X(x), this.Y(y)) : c.moveTo(this.X(x), this.Y(y)))); c.stroke();
        c.setLineDash([]); c.globalAlpha = 1;
      });
    }
    points(pts, color = U.C.blue, r = 4, fill = true) {
      this.clip((c) => { for (const [x, y] of pts) { c.beginPath(); c.arc(this.X(x), this.Y(y), r, 0, 7); if (fill) { c.fillStyle = color; c.fill(); } else { c.strokeStyle = color; c.lineWidth = 2; c.stroke(); } } });
    }
    bars(xs, ys, color = U.C.blue, width = 0.8, alpha = 1) {
      this.clip((c) => { c.globalAlpha = alpha; c.fillStyle = color; xs.forEach((x, i) => { const x0 = this.X(x - width / 2), x1 = this.X(x + width / 2), y0 = this.Y(Math.max(0, this.o.ylim[0])), y1 = this.Y(ys[i]); c.fillRect(x0, Math.min(y0, y1), x1 - x0, Math.abs(y1 - y0)); }); c.globalAlpha = 1; });
    }
    band(xs, lo, hi, color, alpha = 0.18) {
      this.clip((c) => {
        c.globalAlpha = alpha; c.fillStyle = color; c.beginPath();
        xs.forEach((x, i) => (i ? c.lineTo(this.X(x), this.Y(hi[i])) : c.moveTo(this.X(x), this.Y(hi[i]))));
        for (let i = xs.length - 1; i >= 0; i--) c.lineTo(this.X(xs[i]), this.Y(lo[i]));
        c.fill(); c.globalAlpha = 1;
      });
    }
    hline(y, color = U.C.muted, dash = [5, 5], label = "") {
      const p = this.px; this.line([[this.o.xlim[0], y], [this.o.xlim[1], y]], color, 1.5, dash);
      if (label) { const c = this.ctx; c.fillStyle = color; c.font = "13px " + U.css("--font"); c.textAlign = "right"; c.textBaseline = "bottom"; c.fillText(label, p.r - 4, this.Y(y) - 3); }
    }
    vline(x, color = U.C.muted, dash = [5, 5]) { this.line([[x, this.o.ylim[0]], [x, this.o.ylim[1]]], color, 1.5, dash); }
    text(x, y, s, color = U.C.ink, align = "left", font = "14px") {
      const c = this.ctx; c.fillStyle = color; c.font = font + " " + U.css("--font"); c.textAlign = align; c.textBaseline = "middle"; c.fillText(s, this.X(x), this.Y(y));
    }
  };

  // moving average for noisy learning curves
  U.smooth = (arr, k = 10) => arr.map((_, i) => { const s = Math.max(0, i - k + 1); let t = 0; for (let j = s; j <= i; j++) t += arr[j]; return t / (i - s + 1); });

  // diverging colour: negative → red, 0 → white, positive → blue
  U.divColor = function (v, vmax) {
    const t = U.clamp(v / (vmax || 1), -1, 1);
    const [r, g, b] = t >= 0 ? [255 - t * (255 - 31), 255 - t * (255 - 95), 255 - t * (255 - 174)] : [255 + t * (255 - 201), 255 + t * (255 - 42), 255 + t * (255 - 42)];
    return `rgb(${r | 0},${g | 0},${b | 0})`;
  };

  // ------------------------------------------------------------------ graph (MDP / Markov chain) renderer
  U.Graph = class {
    constructor(parent, w, h, nodes, edges, opts = {}) {
      this.opts = Object.assign({ nodeR: 38, font: 18 }, opts);
      this.svg = U.svg("svg", { width: w, height: h, viewBox: `0 0 ${w} ${h}` });
      this.svg.style.overflow = "visible";
      const defs = U.svg("defs");
      for (const [name, col] of [["a-ink", U.C.ink], ["a-hl", U.C.orange], ["a-mut", "#aab0bb"], ["a-blue", U.C.blue], ["a-green", U.C.green]]) {
        defs.append(U.svg("marker", { id: name + this.uid(), viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, U.svg("path", { d: "M0,0 L10,5 L0,10 z", fill: col })));
      }
      this.svg.append(defs);
      this.gE = U.svg("g"); this.gN = U.svg("g"); this.gT = U.svg("g");
      this.svg.append(this.gE, this.gN, this.gT);
      this.nodes = {}; this.edges = {};
      nodes.forEach((n) => this.addNode(n));
      edges.forEach((e, i) => this.addEdge(Object.assign({ id: e.id || `${e.from}>${e.to}${e.tag ? ":" + e.tag : ""}` }, e)));
      parent.append(this.svg);
    }
    uid() { if (!this._uid) this._uid = "g" + Math.random().toString(36).slice(2, 7); return this._uid; }
    marker(name) { return `url(#${name}${this.uid()})`; }
    addNode(n) {
      const r = n.r || (n.shape === "dot" ? 7 : this.opts.nodeR);
      const g = U.svg("g", { transform: `translate(${n.x},${n.y})`, style: "cursor:pointer" });
      let shape;
      if (n.shape === "square") shape = U.svg("rect", { x: -r, y: -r * 0.75, width: 2 * r, height: 1.5 * r, rx: 8, fill: n.fill || "#fff", stroke: U.C.ink, "stroke-width": 2.5 });
      else if (n.shape === "dot") shape = U.svg("circle", { r, fill: U.C.ink });
      else shape = U.svg("circle", { r, fill: n.fill || "#fff", stroke: U.C.ink, "stroke-width": 2.5 });
      g.append(shape);
      const lab = U.svg("text", { "text-anchor": "middle", "dominant-baseline": "central", "font-size": n.fontSize || this.opts.font, "font-weight": 600, fill: U.C.ink, y: 0 }, n.label || "");
      const sub = U.svg("text", { "text-anchor": "middle", "dominant-baseline": "central", "font-size": 15, fill: U.C.blue, "font-weight": 700, y: 14 }, "");
      const ext = U.svg("text", { "text-anchor": "middle", "font-size": 15, fill: U.C.muted, y: -(r + 8) }, n.above || "");
      if (n.below) ext.setAttribute("y", r + 20), (ext.textContent = n.below);
      g.append(lab, sub, ext);
      this.gN.append(g);
      this.nodes[n.id] = Object.assign({}, n, { kind: n.shape || "circle", r, g, shape, lab, sub, ext });
      if (n.onclick) g.addEventListener("click", () => n.onclick(n.id));
    }
    geom(e) {
      const a = this.nodes[e.from], b = this.nodes[e.to];
      if (e.from === e.to) {
        const ang = ((e.loop ?? -90) * Math.PI) / 180, r = a.r, s = 1.9 * r;
        const p1 = [a.x + r * Math.cos(ang - 0.45), a.y + r * Math.sin(ang - 0.45)], p2 = [a.x + r * Math.cos(ang + 0.45), a.y + r * Math.sin(ang + 0.45)];
        const c1 = [a.x + s * 1.3 * Math.cos(ang - 0.6), a.y + s * 1.3 * Math.sin(ang - 0.6)], c2 = [a.x + s * 1.3 * Math.cos(ang + 0.6), a.y + s * 1.3 * Math.sin(ang + 0.6)];
        return { d: `M${p1} C${c1} ${c2} ${p2}`, mid: [a.x + s * 1.12 * Math.cos(ang), a.y + s * 1.12 * Math.sin(ang)] };
      }
      const dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy), bend = e.bend || 0;
      const cx = (a.x + b.x) / 2 - (dy / L) * bend * L, cy = (a.y + b.y) / 2 + (dx / L) * bend * L;
      const trim = (n, tx, ty, extra) => { const ux = tx - n.x, uy = ty - n.y, l = Math.hypot(ux, uy); const rr = n.kind === "square" ? n.r * 0.95 : n.r; return [n.x + (ux / l) * (rr + extra), n.y + (uy / l) * (rr + extra)]; };
      const p1 = trim(a, cx, cy, 2), p2 = trim(b, cx, cy, 4);
      const mx = 0.25 * p1[0] + 0.5 * cx + 0.25 * p2[0], my = 0.25 * p1[1] + 0.5 * cy + 0.25 * p2[1];
      return { d: `M${p1} Q${cx},${cy} ${p2}`, mid: [mx, my] };
    }
    addEdge(e) {
      const { d, mid } = this.geom(e);
      const path = U.svg("path", { d, fill: "none", stroke: U.C.ink, "stroke-width": 2, "marker-end": this.marker("a-ink") });
      this.gE.append(path);
      let lab = null;
      if (e.label != null) {
        const off = e.labelOff || [0, 0];
        lab = U.svg("text", { x: mid[0] + off[0], y: mid[1] + off[1], "text-anchor": "middle", "dominant-baseline": "central", "font-size": e.fontSize || 16, fill: U.C.ink, "paint-order": "stroke", stroke: "#fff", "stroke-width": 5 }, e.label);
        this.gT.append(lab);
      }
      this.edges[e.id] = Object.assign({}, e, { path, lab });
    }
    edgeBetween(a, b) { return Object.values(this.edges).find((e) => e.from === a && e.to === b); }
    styleEdge(id, mode) {
      const e = this.edges[id]; if (!e) return;
      const m = { hl: [U.C.orange, 4, "a-hl"], dim: ["#c9ccd3", 1.5, "a-mut"], blue: [U.C.blue, 3.5, "a-blue"], green: [U.C.green, 4, "a-green"], normal: [U.C.ink, 2, "a-ink"] }[mode || "normal"];
      e.path.setAttribute("stroke", m[0]); e.path.setAttribute("stroke-width", m[1]); e.path.setAttribute("marker-end", this.marker(m[2]));
      if (e.lab) e.lab.setAttribute("fill", mode === "dim" ? "#aab0bb" : mode === "normal" || !mode ? U.C.ink : m[0]);
    }
    styleNode(id, fill, stroke) {
      const n = this.nodes[id]; if (!n || n.kind === "dot") return;
      n.shape.setAttribute("fill", fill || n.fill || "#fff"); n.shape.setAttribute("stroke", stroke || U.C.ink);
    }
    resetStyles() { Object.keys(this.edges).forEach((id) => this.styleEdge(id, "normal")); Object.keys(this.nodes).forEach((id) => this.styleNode(id)); }
    setSub(id, text, color) { const n = this.nodes[id]; n.sub.textContent = text; if (color) n.sub.setAttribute("fill", color); n.lab.setAttribute("y", text ? -8 : 0); }
    setExt(id, text) { this.nodes[id].ext.textContent = text; }
    // animate a token along an edge; returns a promise
    token(id, ms = 500, color = U.C.orange) {
      const e = this.edges[id]; if (!e) return Promise.resolve();
      if (!this.tok) { this.tok = U.svg("circle", { r: 11, fill: color, stroke: "#fff", "stroke-width": 3 }); this.gT.append(this.tok); }
      this.tok.setAttribute("fill", color); this.tok.style.display = "";
      const L = e.path.getTotalLength(), t0 = performance.now();
      return new Promise((res) => {
        const f = (t) => {
          const k = Math.min(1, (t - t0) / ms), ease = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
          const p = e.path.getPointAtLength(ease * L); this.tok.setAttribute("cx", p.x); this.tok.setAttribute("cy", p.y);
          k < 1 ? requestAnimationFrame(f) : res();
        };
        requestAnimationFrame(f);
      });
    }
    placeToken(nodeId, color = U.C.orange) {
      const n = this.nodes[nodeId];
      if (!this.tok) { this.tok = U.svg("circle", { r: 11, fill: color, stroke: "#fff", "stroke-width": 3 }); this.gT.append(this.tok); }
      this.tok.style.display = ""; this.tok.setAttribute("fill", color); this.tok.setAttribute("cx", n.x); this.tok.setAttribute("cy", n.y - n.r);
    }
    hideToken() { if (this.tok) this.tok.style.display = "none"; }
  };

  // ------------------------------------------------------------------ deck engine
  const Deck = { widgets: {}, U };
  window.Deck = Deck; window.U = U;
  Deck.widget = (name, factory) => (Deck.widgets[name] = factory);

  let slides = [], cur = 0, deckEl, notesPanel, progress, presenter = null, t0 = null;

  function fragsOf(s) { return U.$$(".fragment", s).sort((a, b) => (+a.dataset.f || 0) - (+b.dataset.f || 0)); }

  function initWidgets(s) {
    if (s._widgets) return;
    s._widgets = [];
    for (const el of U.$$("[data-widget]", s)) {
      const f = Deck.widgets[el.dataset.widget];
      if (!f) { console.warn("missing widget", el.dataset.widget); continue; }
      try { s._widgets.push(f(el) || {}); } catch (err) { console.error("widget", el.dataset.widget, err); el.textContent = "widget error: " + err.message; }
    }
  }

  function show(i, fragMode = "none") {
    i = U.clamp(i, 0, slides.length - 1);
    const old = slides[cur];
    if (i !== cur) Deck.cameFrom = cur;
    if (old && old !== slides[i]) { old.classList.remove("active"); (old._widgets || []).forEach((w) => w.stop && w.stop()); }
    cur = i;
    const s = slides[cur];
    s.classList.add("active");
    initWidgets(s);   // first, so fragments created by widgets are included below
    const fr = fragsOf(s);
    fr.forEach((f, k) => f.classList.toggle("visible", fragMode === "all" || (typeof fragMode === "number" && k < fragMode)));
    (s._widgets || []).forEach((w) => w.start && w.start());
    if (location.hash !== "#" + (cur + 1)) history.replaceState(null, "", "#" + (cur + 1));
    progress.style.width = ((cur + 1) / slides.length) * 100 + "%";
    progress.style.background = getComputedStyle(s).getPropertyValue("--accent") || "";
    const tn = document.getElementById("touchnum"); if (tn) tn.textContent = `${cur + 1} / ${slides.length}`;
    updateNotes();
    if (!t0 && cur > 0) t0 = Date.now();
  }

  function fragEvent(f, visible) { f.dispatchEvent(new CustomEvent("fragment", { bubbles: true, detail: { visible } })); }
  function next() {
    const hidden = fragsOf(slides[cur]).filter((f) => !f.classList.contains("visible"));
    if (hidden.length) { hidden[0].classList.add("visible"); fragEvent(hidden[0], true); updateNotes(); return; }
    if (cur < slides.length - 1) show(cur + 1);
  }
  function prev() {
    const vis = fragsOf(slides[cur]).filter((f) => f.classList.contains("visible"));
    if (vis.length) { const f = vis[vis.length - 1]; f.classList.remove("visible"); fragEvent(f, false); updateNotes(); return; }
    if (cur > 0) show(cur - 1, "all");
  }

  function sectionOf(i) { for (let k = i; k >= 0; k--) if (slides[k].dataset.section) return slides[k].dataset.section; return ""; }
  function clock() { if (!t0) return "00:00"; const s = Math.floor((Date.now() - t0) / 1000); return `${String(Math.floor(s / 3600)).padStart(1, "0")}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; }
  function notesHTML(i) { const n = U.$("aside.notes", slides[i]); return n ? n.innerHTML : "<em>No notes for this slide.</em>"; }
  function titleOf(i) { if (i >= slides.length) return "— end —"; const h = U.$("h2, h1", slides[i]); return h ? h.textContent.trim() : "(untitled)"; }

  function updateNotes() {
    const s = slides[cur];
    const fr = fragsOf(s), vis = fr.filter((f) => f.classList.contains("visible")).length;
    const meta = `Slide ${cur + 1}/${slides.length}${fr.length ? ` · step ${vis}/${fr.length}` : ""} · ${sectionOf(cur)} · ⏱ ${clock()}${s.dataset.time ? ` · suggested ${s.dataset.time}` : ""}${s.dataset.optional ? " · OPTIONAL — skip if behind" : ""}`;
    notesPanel.innerHTML = `<div class="meta">${meta}</div>${notesHTML(cur)}`;
    if (presenter && !presenter.closed) {
      const d = presenter.document;
      d.getElementById("meta").textContent = meta;
      d.getElementById("notes").innerHTML = notesHTML(cur);
      d.getElementById("now").textContent = titleOf(cur);
      d.getElementById("next").textContent = vis < fr.length ? "(more steps on this slide)" : titleOf(cur + 1);
    }
  }

  function openPresenter() {
    presenter = window.open("", "cce5502-presenter", "width=900,height=700");
    if (!presenter) return;
    const d = presenter.document;
    d.open();
    d.write(`<!doctype html><title>Presenter — ${document.title}</title><style>
      body{font:18px/1.5 system-ui,sans-serif;background:#16181d;color:#e8eaf0;margin:0;padding:18px 26px}
      #meta{color:#e8a100;font-weight:600;font-size:16px} #clock{font:700 44px ui-monospace,monospace;color:#fff;float:right}
      h3{margin:14px 0 4px;color:#9fc3f0;font-size:15px;text-transform:uppercase;letter-spacing:.08em}
      #now{font-size:26px;font-weight:700} #next{font-size:20px;color:#b9bfcc} #notes{font-size:20px}
      #notes ul{padding-left:22px} code{background:#2a2e38;padding:1px 5px;border-radius:4px}
      .hint{color:#777;font-size:13px;margin-top:20px}</style>
      <div id="clock">0:00:00</div><div id="meta"></div><h3>Now</h3><div id="now"></div><h3>Next</h3><div id="next"></div><h3>Notes</h3><div id="notes"></div>
      <div class="hint">Arrow keys / clicker work in this window too. T resets the timer.</div>`);
    d.close();
    presenter.addEventListener("keydown", onKey);
    const tick = () => { if (!presenter || presenter.closed) return; d.getElementById("clock").textContent = clock(); setTimeout(tick, 1000); };
    tick();
    updateNotes();
  }

  let gotoBuf = null;
  function onKey(e) {
    const tag = (e.target && e.target.tagName) || "";
    const typing = tag === "INPUT" && e.target.type !== "range" || tag === "TEXTAREA" || tag === "SELECT";
    if (typing) return;
    if (gotoBuf !== null) {
      if (/^\d$/.test(e.key)) { gotoBuf += e.key; U.$("#goto").textContent = "Go to: " + gotoBuf; return; }
      if (e.key === "Enter" && gotoBuf) show(+gotoBuf - 1);
      gotoBuf = null; U.$("#goto").style.display = "none"; return;
    }
    const onRange = tag === "INPUT"; // range sliders keep arrows; clickers use PgUp/PgDn
    switch (e.key) {
      case "ArrowRight": case "ArrowDown": if (onRange) return; // fallthrough
      case "PageDown": case " ": e.preventDefault(); next(); break;
      case "ArrowLeft": case "ArrowUp": if (onRange) return; // fallthrough
      case "PageUp": e.preventDefault(); prev(); break;
      case "Home": show(0); break;
      case "End": show(slides.length - 1); break;
      case "n": case "N": notesPanel.classList.toggle("show"); break;
      case "p": case "P": openPresenter(); break;
      case "o": case "O": toggleOverview(); break;
      case "f": case "F": document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); break;
      case "b": case "B": case ".": { const b = U.$("#blackout"); b.style.display = b.style.display === "block" ? "none" : "block"; break; }
      case "g": case "G": gotoBuf = ""; U.$("#goto").style.display = "block"; U.$("#goto").textContent = "Go to: "; break;
      case "t": case "T": t0 = Date.now(); updateNotes(); break;
      case "i": case "I": location.href = deckEl.dataset.index || "index.html"; break;
      case "m": case "M": { const k = slides.findIndex((x) => x.querySelector('[data-widget="mindmap"]')); if (k >= 0) show(k, "all"); break; }
      case "?": case "h": case "H": { const h = U.$("#help"); h.style.display = h.style.display === "block" ? "none" : "block"; break; }
      case "Escape": U.$("#help").style.display = "none"; if (document.body.classList.contains("overview")) toggleOverview(); break;
    }
  }

  function toggleOverview() {
    const on = document.body.classList.toggle("overview");
    if (on) slides.forEach((s) => (s.style.display = "flex"));
    else { slides.forEach((s) => (s.style.display = "")); fit(); }
    if (on) slides[cur].scrollIntoView({ block: "center" });
  }

  function fit() {
    if (document.body.classList.contains("overview")) return;
    const s = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
    deckEl.style.transform = `scale(${s})`;
  }

  function boot() {
    deckEl = U.$(".deck");
    slides = U.$$(".deck > section");
    slides.forEach((s) => s.classList.add("slide"));
    const vp = U.el("div", { id: "viewport" }); deckEl.before(vp); vp.append(deckEl);
    progress = U.el("div", { id: "progress" }); document.body.append(progress);
    notesPanel = U.el("div", { id: "notes-panel" }); document.body.append(notesPanel);
    document.body.append(U.el("div", { id: "goto" }), U.el("div", { id: "blackout" }));
    document.body.append(U.el("div", { id: "help", html: `<table>
      <tr><td><kbd>→</kbd> <kbd>Space</kbd> <kbd>PgDn</kbd></td><td>next step / slide</td></tr>
      <tr><td><kbd>←</kbd> <kbd>PgUp</kbd></td><td>previous</td></tr>
      <tr><td><kbd>N</kbd></td><td>toggle speaker notes (bottom)</td></tr>
      <tr><td><kbd>P</kbd></td><td>presenter window (notes, next, timer)</td></tr>
      <tr><td><kbd>O</kbd></td><td>overview (click a slide)</td></tr>
      <tr><td><kbd>G</kbd> 12 <kbd>Enter</kbd></td><td>go to slide 12</td></tr>
      <tr><td><kbd>F</kbd> / <kbd>B</kbd> / <kbd>T</kbd></td><td>fullscreen / blackout / reset timer</td></tr>
      <tr><td><kbd>M</kbd></td><td>back to the mind map (fully drawn)</td></tr>
      <tr><td><kbd>I</kbd></td><td>back to the index page</td></tr></table>` }));
    document.body.append(U.el("a", { id: "home", href: deckEl.dataset.index || "index.html", title: "Back to index (I)" }, "⌂"));
    const footer = deckEl.dataset.footer || document.title;
    slides.forEach((s, i) => {
      if (!s.classList.contains("title")) s.append(U.el("div", { class: "footer" }, U.el("span", {}, footer), U.el("span", {}, String(i + 1))));
      s.addEventListener("click", () => { if (document.body.classList.contains("overview")) { toggleOverview(); show(i); } });
    });
    bootHooks.forEach((fn) => { try { fn(slides); } catch (err) { console.error(err); } });
    if (window.renderMathInElement) {
      renderMathInElement(deckEl, Object.assign({ delimiters: [{ left: "$$", right: "$$", display: true }, { left: "$", right: "$", display: false }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }, U.KATEX));
    }
    // ---- touch: swipe left/right, plus on-screen buttons on touch devices
    const nav = U.el("div", { id: "touchnav" },
      U.el("button", { "aria-label": "previous slide", onclick: (e) => { e.stopPropagation(); prev(); } }, "‹"),
      U.el("span", { id: "touchnum" }),
      U.el("button", { "aria-label": "next slide", onclick: (e) => { e.stopPropagation(); next(); } }, "›"));
    document.body.append(nav);
    let tx = null, ty = null, tt = 0;
    const interactive = (el) => el && el.closest && el.closest("input, select, textarea, button, a");
    document.addEventListener("touchstart", (e) => {
      if (e.touches.length !== 1 || interactive(e.target)) { tx = null; return; }
      tx = e.touches[0].clientX; ty = e.touches[0].clientY; tt = Date.now();
    }, { passive: true });
    document.addEventListener("touchend", (e) => {
      if (tx === null) return;
      const dx = e.changedTouches[0].clientX - tx, dy = e.changedTouches[0].clientY - ty;
      tx = null;
      if (Date.now() - tt < 800 && Math.abs(dx) > 40 && Math.abs(dx) > 1.5 * Math.abs(dy)) (dx < 0 ? next : prev)();
    }, { passive: true });
    window.addEventListener("resize", fit);
    window.addEventListener("hashchange", () => { const k = parseInt(location.hash.slice(1), 10); if (!isNaN(k) && k - 1 !== cur) show(k - 1); });
    document.addEventListener("keydown", onKey);
    fit();
    const h = parseInt(location.hash.slice(1), 10);
    show(isNaN(h) ? 0 : h - 1);
    setInterval(() => { if (notesPanel.classList.contains("show")) updateNotes(); }, 1000);
  }
  // 30-second "think time" ring for quiz slides: click to start / restart
  Deck.widget("quiztimer", (root) => {
    const R = 30, C = 2 * Math.PI * R, total = +(root.dataset.seconds || 30);
    root.classList.add("quiz-timer");
    root.innerHTML = `<svg width="80" height="80" viewBox="0 0 80 80"><circle cx="40" cy="40" r="${R}" fill="none" stroke="rgba(255,255,255,.15)" stroke-width="7"/>
      <circle class="arc" cx="40" cy="40" r="${R}" fill="none" stroke="#e8a100" stroke-width="7" stroke-linecap="round" transform="rotate(-90 40 40)" stroke-dasharray="${C}" stroke-dashoffset="0"/>
      <text x="40" y="47" text-anchor="middle" font-size="22" font-weight="800" fill="#fff">${total}</text></svg><div>think time</div>`;
    const arc = root.querySelector(".arc"), txt = root.querySelector("text"), lab = root.querySelector("div");
    let t0 = null, raf = null;
    const tick = (t) => {
      const left = Math.max(0, total - (t - t0) / 1000);
      arc.setAttribute("stroke-dashoffset", C * (1 - left / total)); arc.setAttribute("stroke", left < 5 ? "#ff6b6b" : "#e8a100");
      txt.textContent = Math.ceil(left); lab.textContent = left > 0 ? "think time" : "time's up!";
      raf = left > 0 ? requestAnimationFrame(tick) : null;
    };
    root.addEventListener("click", (e) => { e.stopPropagation(); if (raf) cancelAnimationFrame(raf); t0 = performance.now(); raf = requestAnimationFrame(tick); });
    return { stop() { if (raf) cancelAnimationFrame(raf); raf = null; arc.setAttribute("stroke-dashoffset", 0); txt.textContent = total; lab.textContent = "think time"; } };
  });

  Deck.boot = boot;
  Deck.go = (i, step) => show(i, step == null ? "none" : step);   // step: number of build steps to reveal, or "all"
  Deck.slides = () => slides;
  const bootHooks = [];
  Deck.onBoot = (fn) => bootHooks.push(fn);
  Deck.current = () => cur;
  // index of the first slide whose heading contains `text` (case-insensitive), or -1
  Deck.findSlide = (text) => { const t = text.toLowerCase(); return slides.findIndex((s) => { const h = s.querySelector("h2, h1"); return h && h.textContent.toLowerCase().includes(t); }); };
  // wait for the bundled serif font so canvas figures are drawn with it (max 1.5 s)
  document.addEventListener("DOMContentLoaded", () => {
    let done = false; const go = () => { if (!done) { done = true; boot(); } };
    (document.fonts ? document.fonts.load('16px "Source Serif 4"').then(() => document.fonts.ready) : Promise.resolve()).then(go, go);
    setTimeout(go, 1500);
  });
})();

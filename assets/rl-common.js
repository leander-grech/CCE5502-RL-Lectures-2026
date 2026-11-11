/* Shared RL environments + renderers for the CCE5502 decks. Depends on deck.js (window.U). */
(function () {
  "use strict";
  const RL = {};
  window.RL = RL;
  const C = U.C;

  // ------------------------------------------------------------------ Student MRP (Silver, Lecture 2)
  RL.studentMRP = {
    states: ["C1", "C2", "C3", "Pass", "Pub", "FB", "Sleep"],
    R: { C1: -2, C2: -2, C3: -2, Pass: 10, Pub: 1, FB: -1, Sleep: 0 },
    P: {
      C1: { C2: 0.5, FB: 0.5 },
      C2: { C3: 0.8, Sleep: 0.2 },
      C3: { Pass: 0.6, Pub: 0.4 },
      Pass: { Sleep: 1.0 },
      Pub: { C1: 0.2, C2: 0.4, C3: 0.4 },
      FB: { FB: 0.9, C1: 0.1 },
      Sleep: {},
    },
    terminal: "Sleep",
    layout: { FB: [140, 80], C1: [140, 270], C2: [390, 270], C3: [640, 270], Pass: [860, 400], Pub: [520, 440], Sleep: [760, 90] },
    // exact value function: v = (I - γP)^-1 R over non-terminal states
    values(gamma) {
      const S = this.states.filter((s) => s !== this.terminal), n = S.length;
      const A = S.map((s, i) => S.map((t, j) => (i === j ? 1 : 0) - gamma * (this.P[s][t] || 0)));
      const b = S.map((s) => this.R[s]);
      if (gamma >= 1) { /* FB loop still terminates w.p.1, system is non-singular */ }
      const v = U.solve(A, b), out = { Sleep: 0 };
      S.forEach((s, i) => (out[s] = v[i]));
      return out;
    },
    sample(rng, start = "C1", maxLen = 60) {
      const ep = [start];
      let s = start;
      while (s !== this.terminal && ep.length < maxLen) {
        const nx = Object.keys(this.P[s]), ps = nx.map((k) => this.P[s][k]);
        s = nx[rng.categorical(ps)];
        ep.push(s);
      }
      return ep;
    },
  };

  RL.buildChainGraph = function (parent, { w = 960, h = 500, rewards = false, onclick } = {}) {
    const M = RL.studentMRP, L = M.layout;
    const names = { C1: "Class 1", C2: "Class 2", C3: "Class 3", Pass: "Pass", Pub: "Pub", FB: "Facebook", Sleep: "Sleep" };
    const nodes = M.states.map((s) => ({
      id: s, x: L[s][0], y: L[s][1], label: names[s], shape: s === "Sleep" ? "square" : "circle", fontSize: 16, r: 44,
      fill: s === "Sleep" ? "#eef0f3" : "#fff", above: rewards ? `R = ${M.R[s]}` : "", onclick,
    }));
    const E = [
      ["C1", "C2", 0.5, 0], ["C1", "FB", 0.5, 0.15], ["FB", "C1", 0.1, 0.15], ["FB", "FB", 0.9, 0, { loop: 180 }],
      ["C2", "C3", 0.8, 0], ["C2", "Sleep", 0.2, -0.1], ["C3", "Pass", 0.6, -0.1], ["C3", "Pub", 0.4, 0.22],
      ["Pass", "Sleep", 1.0, 0.1], ["Pub", "C1", 0.2, -0.15], ["Pub", "C2", 0.4, -0.08], ["Pub", "C3", 0.4, 0.22],
    ];
    const edges = E.map(([f, t, p, b, x]) => Object.assign({ from: f, to: t, label: String(p), bend: b }, x || {}));
    // loop for FB is drawn to the left; move the "R =" labels out of the way
    const g = new U.Graph(parent, w, h, nodes, edges);
    if (rewards) {
      const off = { C1: [0, 66], C2: [0, -54], C3: [0, -54], Pass: [0, 66], Pub: [-84, 5], FB: [0, -56], Sleep: [0, -44] };
      for (const [id, [dx, dy]] of Object.entries(off)) { const t = g.nodes[id].ext; t.setAttribute("x", dx); t.setAttribute("y", dy); t.setAttribute("fill", U.C.orange); t.setAttribute("font-weight", 700); }
    }
    return g;
  };

  // ------------------------------------------------------------------ Student MDP
  // Each state has actions; each action has a reward and a distribution over next states.
  RL.studentMDP = {
    states: ["FB", "C1", "C2", "C3", "Sleep"],
    terminal: "Sleep",
    A: {
      FB: { Facebook: { r: -1, P: { FB: 1 } }, Quit: { r: 0, P: { C1: 1 } } },
      C1: { Facebook: { r: -1, P: { FB: 1 } }, Study: { r: -2, P: { C2: 1 } } },
      C2: { Sleep: { r: 0, P: { Sleep: 1 } }, Study: { r: -2, P: { C3: 1 } } },
      C3: { Study: { r: 10, P: { Sleep: 1 } }, Pub: { r: 1, P: { C1: 0.2, C2: 0.4, C3: 0.4 } } },
      Sleep: {},
    },
    layout: { FB: [170, 110], C1: [170, 330], C2: [440, 330], C3: [710, 330], Sleep: [580, 100], PubDot: [540, 470] },
    // v_pi via linear solve; pi[s][a] = probability
    evaluate(pi, gamma = 1) {
      const S = this.states.filter((s) => s !== this.terminal);
      const A = S.map(() => S.map(() => 0)), b = S.map(() => 0);
      S.forEach((s, i) => {
        A[i][i] += 1;
        for (const [a, d] of Object.entries(this.A[s])) {
          const p = pi[s][a] || 0;
          b[i] += p * d.r;
          for (const [t, pt] of Object.entries(d.P)) if (t !== this.terminal) A[i][S.indexOf(t)] -= gamma * p * pt;
        }
      });
      const v = U.solve(A, b), out = { Sleep: 0 };
      S.forEach((s, i) => (out[s] = v[i]));
      return out;
    },
    q(v, s, a, gamma = 1) { const d = this.A[s][a]; let x = d.r; for (const [t, p] of Object.entries(d.P)) x += gamma * p * v[t]; return x; },
    optimal(gamma = 1) {
      let v = { FB: 0, C1: 0, C2: 0, C3: 0, Sleep: 0 };
      for (let k = 0; k < 500; k++) {
        const nv = { Sleep: 0 };
        for (const s of this.states) if (s !== this.terminal) nv[s] = Math.max(...Object.keys(this.A[s]).map((a) => this.q(v, s, a, gamma)));
        v = nv;
      }
      return v;
    },
  };

  RL.buildMDPGraph = function (parent, { w = 900, h = 540, onclick } = {}) {
    const M = RL.studentMDP, L = M.layout;
    const names = { FB: "Facebook", C1: "Class 1", C2: "Class 2", C3: "Class 3", Sleep: "Sleep" };
    const nodes = M.states.map((s) => ({ id: s, x: L[s][0], y: L[s][1], label: names[s], shape: s === "Sleep" ? "square" : "circle", r: 46, fontSize: 16, fill: s === "Sleep" ? "#eef0f3" : "#fff", onclick }));
    nodes.push({ id: "PubDot", x: L.PubDot[0], y: L.PubDot[1], shape: "dot", r: 8 });
    const edges = [
      { from: "FB", to: "FB", tag: "Facebook", label: "Facebook, R=−1", loop: 180, labelOff: [-6, -52] },
      { from: "FB", to: "C1", tag: "Quit", label: "Quit, R=0", bend: 0.18, labelOff: [-42, 0] },
      { from: "C1", to: "FB", tag: "Facebook", label: "Facebook, R=−1", bend: 0.18, labelOff: [66, 0] },
      { from: "C1", to: "C2", tag: "Study", label: "Study, R=−2", labelOff: [0, -14] },
      { from: "C2", to: "Sleep", tag: "Sleep", label: "Sleep, R=0", bend: -0.1, labelOff: [-30, 0] },
      { from: "C2", to: "C3", tag: "Study", label: "Study, R=−2", labelOff: [0, -14] },
      { from: "C3", to: "Sleep", tag: "Study", label: "Study, R=+10", bend: 0.1, labelOff: [40, 0] },
      { from: "C3", to: "PubDot", tag: "Pub", label: "Pub, R=+1", bend: -0.15, labelOff: [52, 16] },
      { from: "PubDot", to: "C1", tag: "p", label: "0.2", bend: -0.15 },
      { from: "PubDot", to: "C2", tag: "p", label: "0.4", bend: -0.05 },
      { from: "PubDot", to: "C3", tag: "p", label: "0.4", bend: -0.2 },
    ];
    edges.forEach((e) => (e.id = `${e.from}:${e.tag}${e.tag === "p" ? ">" + e.to : ""}`));
    return new U.Graph(parent, w, h, nodes, edges, { nodeR: 46 });
  };

  // ------------------------------------------------------------------ Grid worlds
  // cell types: "" empty, "#" wall, "G" goal (+), "X" pit (−), "S" start, "C" cliff
  RL.ACTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]]; // up, right, down, left
  RL.ARROWS = ["↑", "→", "↓", "←"];

  RL.Grid = class {
    constructor(rows, cols, opts = {}) {
      this.rows = rows; this.cols = cols; this.n = rows * cols;
      this.type = Array(this.n).fill("");
      this.reward = Array(this.n).fill(0); // reward on *entering* a terminal / special cell
      this.step = opts.step ?? -1;
      this.gamma = opts.gamma ?? 1;
      this.start = opts.start ?? 0;
      this.slip = opts.slip ?? 0;
    }
    rc(s) { return [Math.floor(s / this.cols), s % this.cols]; }
    idx(r, c) { return r * this.cols + c; }
    isTerminal(s) { return this.type[s] === "G" || this.type[s] === "X" || this.type[s] === "T"; }
    isWall(s) { return this.type[s] === "#"; }
    move(s, a) {
      const [r, c] = this.rc(s), [dr, dc] = RL.ACTIONS[a];
      const nr = r + dr, nc = c + dc;
      if (nr < 0 || nr >= this.rows || nc < 0 || nc >= this.cols) return s;
      const t = this.idx(nr, nc);
      return this.isWall(t) ? s : t;
    }
    // transition list: [[prob, next, reward, done]]
    trans(s, a) {
      const outcomes = this.slip ? [[1 - this.slip, a], [this.slip / 2, (a + 1) % 4], [this.slip / 2, (a + 3) % 4]] : [[1, a]];
      return outcomes.map(([p, aa]) => { const t = this.move(s, aa); return [p, t, this.step + this.reward[t], this.isTerminal(t)]; });
    }
    // sample step for model-free agents (cliff handled here)
    sample(s, a, rng) {
      let aa = a;
      if (this.slip && rng() < this.slip) aa = rng() < 0.5 ? (a + 1) % 4 : (a + 3) % 4;
      const t = this.move(s, aa);
      if (this.type[t] === "C") return [this.start, this.step + this.reward[t], false, true];
      return [t, this.step + this.reward[t], this.isTerminal(t), false];
    }
    states() { const out = []; for (let s = 0; s < this.n; s++) if (!this.isWall(s) && !this.isTerminal(s)) out.push(s); return out; }
    evalSweep(V, pi) { // pi[s] = [p_up, p_right, p_down, p_left]
      const nV = V.slice();
      for (const s of this.states()) {
        let v = 0;
        for (let a = 0; a < 4; a++) if (pi[s][a]) for (const [p, t, r, d] of this.trans(s, a)) v += pi[s][a] * p * (r + (d ? 0 : this.gamma * V[t]));
        nV[s] = v;
      }
      return nV;
    }
    qFrom(V, s) { return [0, 1, 2, 3].map((a) => this.trans(s, a).reduce((acc, [p, t, r, d]) => acc + p * (r + (d ? 0 : this.gamma * V[t])), 0)); }
    viSweep(V) { const nV = V.slice(); for (const s of this.states()) nV[s] = Math.max(...this.qFrom(V, s)); return nV; }
    greedy(V, tol = 1e-6) { // returns pi with ties split
      const pi = Array.from({ length: this.n }, () => [0.25, 0.25, 0.25, 0.25]);
      for (const s of this.states()) { const q = this.qFrom(V, s), m = Math.max(...q), best = q.map((x) => (x > m - tol ? 1 : 0)), k = best.reduce((a, b) => a + b); pi[s] = best.map((b) => b / k); }
      return pi;
    }
    randomPolicy() { return Array.from({ length: this.n }, () => [0.25, 0.25, 0.25, 0.25]); }
  };

  RL.GridView = class {
    constructor(parent, grid, cell = 70, opts = {}) {
      this.g = grid; this.cell = cell;
      this.o = Object.assign({ showValues: true, showArrows: true, vmax: null, qTriangles: false, fontSize: 15, labels: {} }, opts);
      this.cv = U.canvas(grid.cols * cell + 2, grid.rows * cell + 2, parent);
      this.ctx = this.cv.ctx;
      this.V = null; this.pi = null; this.Q = null; this.agent = null; this.trail = []; this.paths = [];
    }
    cellAt(e) { const [x, y] = this.cv.pos(e); const c = Math.floor((x - 1) / this.cell), r = Math.floor((y - 1) / this.cell); return r >= 0 && r < this.g.rows && c >= 0 && c < this.g.cols ? this.g.idx(r, c) : -1; }
    draw() {
      const { ctx: c, g, cell: z } = this, o = this.o;
      c.clearRect(0, 0, this.cv.W, this.cv.H);
      let vmax = o.vmax;
      if (!vmax && this.V) vmax = Math.max(1e-6, ...this.V.map(Math.abs));
      if (!vmax && this.Q) vmax = Math.max(1e-6, ...this.Q.flat().map(Math.abs));
      for (let s = 0; s < g.n; s++) {
        const [r, col] = g.rc(s), x = 1 + col * z, y = 1 + r * z, t = g.type[s];
        let fill = "#fff";
        if (t === "#") fill = "#3b4252";
        else if (t === "C") fill = "#6b3a3a";
        else if (t === "G") fill = "#c9ebd0";
        else if (t === "X") fill = "#f6c9c9";
        else if (t === "T") fill = "#e3e6ea";
        else if (this.V && o.showValues) fill = U.divColor(this.V[s], vmax);
        c.fillStyle = fill; c.fillRect(x, y, z, z);
        if (this.Q && o.qTriangles && !g.isWall(s) && !g.isTerminal(s) && t !== "C") {
          const cx = x + z / 2, cy = y + z / 2, corners = [[x, y], [x + z, y], [x + z, y + z], [x, y + z]];
          for (let a = 0; a < 4; a++) {
            const p1 = corners[a], p2 = corners[(a + 1) % 4];
            c.fillStyle = U.divColor(this.Q[s][a], vmax); c.beginPath(); c.moveTo(cx, cy); c.lineTo(...p1); c.lineTo(...p2); c.closePath(); c.fill();
            c.strokeStyle = "rgba(0,0,0,.08)"; c.stroke();
          }
        }
        c.strokeStyle = "#c8ccd4"; c.lineWidth = 1; c.strokeRect(x, y, z, z);
        c.textAlign = "center"; c.textBaseline = "middle";
        const lab = o.labels[s] || (t === "G" ? "+" + g.reward[s] : t === "X" ? String(g.reward[s]) : t === "C" ? "cliff" : t === "T" ? "" : "");
        if (lab) { c.fillStyle = t === "C" ? "#fff" : C.ink; c.font = `600 ${o.fontSize}px ${U.css("--font")}`; c.fillText(lab, x + z / 2, y + z / 2); continue; }
        if (t === "#") continue;
        if (this.V && o.showValues && !g.isTerminal(s)) {
          c.fillStyle = C.ink; c.font = `${o.fontSize}px ${U.css("--mono")}`;
          c.fillText(U.fmt(this.V[s], Math.abs(this.V[s]) >= 100 ? 0 : 1), x + z / 2, y + (this.pi && o.showArrows ? z * 0.3 : z / 2));
        }
        if (this.pi && o.showArrows && !g.isTerminal(s)) this.arrows(x, y, this.pi[s], this.V && o.showValues ? 0.62 : 0.5);
      }
      for (const p of this.paths) this.path(p.cells, p.color, p.width || 4);
      if (this.trail.length > 1) this.path(this.trail, C.orange, 3, 0.5);
      if (this.agent != null) {
        const [r, col] = g.rc(this.agent);
        c.beginPath(); c.arc(1 + col * z + z / 2, 1 + r * z + z / 2, z * 0.22, 0, 7); c.fillStyle = C.orange; c.fill(); c.strokeStyle = "#fff"; c.lineWidth = 3; c.stroke();
      }
    }
    arrows(x, y, p, fy) {
      const c = this.ctx, z = this.cell, cx = x + z / 2, cy = y + z * fy, m = Math.max(...p);
      c.strokeStyle = C.ink; c.fillStyle = C.ink;
      for (let a = 0; a < 4; a++) {
        if (p[a] < 1e-9 || p[a] < m - 1e-9) continue;
        const [dy, dx] = RL.ACTIONS[a], L = z * 0.2;
        const ex = cx + dx * L, ey = cy + dy * L;
        c.lineWidth = 2; c.beginPath(); c.moveTo(cx, cy); c.lineTo(ex, ey); c.stroke();
        c.beginPath(); c.moveTo(ex + dx * 6, ey + dy * 6); c.lineTo(ex - dy * 5, ey + dx * 5); c.lineTo(ex + dy * 5, ey - dx * 5); c.closePath(); c.fill();
      }
    }
    path(cells, color, width, alpha = 1) {
      const c = this.ctx, z = this.cell;
      c.globalAlpha = alpha; c.strokeStyle = color; c.lineWidth = width; c.lineJoin = "round"; c.lineCap = "round"; c.beginPath();
      cells.forEach((s, i) => { const [r, col] = this.g.rc(s), x = 1 + col * z + z / 2, y = 1 + r * z + z / 2; i ? c.lineTo(x, y) : c.moveTo(x, y); });
      c.stroke(); c.globalAlpha = 1;
    }
  };

  // ------------------------------------------------------------------ CartPole (Gymnasium CartPole-v1 dynamics, Euler)
  RL.CartPole = class {
    constructor(opts = {}) {
      Object.assign(this, { g: 9.8, mc: 1.0, mp: 0.1, l: 0.5, F: 10.0, tau: 0.02, thMax: (12 * 2 * Math.PI) / 360, xMax: 2.4, maxSteps: 500 }, opts);
      this.rng = U.rng(opts.seed || 7);
      this.reset();
    }
    reset() { this.s = [0, 0, 0, 0].map(() => (this.rng() - 0.5) * 0.1); this.t = 0; this.done = false; return this.s.slice(); }
    stepPhys(a) {
      let [x, xd, th, thd] = this.s;
      const f = a === 1 ? this.F : -this.F, ct = Math.cos(th), st = Math.sin(th), M = this.mc + this.mp, pml = this.mp * this.l;
      const tmp = (f + pml * thd * thd * st) / M;
      const thacc = (this.g * st - ct * tmp) / (this.l * (4 / 3 - (this.mp * ct * ct) / M));
      const xacc = tmp - (pml * thacc * ct) / M;
      x += this.tau * xd; xd += this.tau * xacc; th += this.tau * thd; thd += this.tau * thacc;
      this.s = [x, xd, th, thd];
    }
    step(a) {
      this.stepPhys(a); this.t++;
      const [x, , th] = this.s;
      const term = Math.abs(x) > this.xMax || Math.abs(th) > this.thMax, trunc = this.t >= this.maxSteps;
      this.done = term || trunc;
      return [this.s.slice(), 1, term, trunc];
    }
    draw(ctx, W, H, opts = {}) {
      const [x, , th] = this.s, scale = W / (2 * this.xMax * 1.15), cy = H * 0.72, cx = W / 2 + x * scale;
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#f6f4ef"; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = "#aab0bb"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(0, cy + 16); ctx.lineTo(W, cy + 16); ctx.stroke();
      ctx.fillStyle = "rgba(201,42,42,.12)";
      ctx.fillRect(0, 0, W / 2 - this.xMax * scale, H); ctx.fillRect(W / 2 + this.xMax * scale, 0, W, H);
      ctx.fillStyle = C.ink; ctx.fillRect(cx - 34, cy - 14, 68, 30);
      ctx.fillStyle = "#555"; [[-20], [20]].forEach(([o]) => { ctx.beginPath(); ctx.arc(cx + o, cy + 16, 6, 0, 7); ctx.fill(); });
      const L = this.l * 2 * scale;
      ctx.strokeStyle = this.done && Math.abs(th) > this.thMax ? C.red : "#c98a3a"; ctx.lineWidth = 10; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(cx, cy - 6); ctx.lineTo(cx + L * Math.sin(th), cy - 6 - L * Math.cos(th)); ctx.stroke();
      ctx.fillStyle = C.blue; ctx.beginPath(); ctx.arc(cx, cy - 6, 5, 0, 7); ctx.fill();
      if (opts.action != null) {
        ctx.fillStyle = C.orange; ctx.font = "bold 26px " + U.css("--font"); ctx.textAlign = "center";
        ctx.fillText(opts.action === 1 ? "→" : "←", cx + (opts.action === 1 ? 52 : -52), cy + 4);
      }
      ctx.fillStyle = C.ink; ctx.font = "16px " + U.css("--mono"); ctx.textAlign = "left"; ctx.textBaseline = "top";
      ctx.fillText(`t = ${this.t}   return = ${this.t}`, 12, 10);
      if (opts.label) { ctx.textAlign = "right"; ctx.fillText(opts.label, W - 12, 10); }
    }
  };
})();

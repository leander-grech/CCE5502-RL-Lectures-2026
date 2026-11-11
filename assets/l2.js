/* Lecture 2 widgets — RL algorithms */
(function () {
  "use strict";
  const { el, C } = U;
  const argmax = (a) => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
  const argmaxRand = (a, rng) => { const m = Math.max(...a), idx = []; a.forEach((x, i) => { if (x >= m - 1e-12) idx.push(i); }); return idx[Math.floor(rng() * idx.length)]; };

  // ================================================================ GPI zig-zag
  Deck.widget("gpi", (root) => {
    const W = 520, H = 330, cv = U.canvas(W, H, root), ctx = cv.ctx;
    let t = 0;
    const loop = U.loop((dt) => {
      t += dt;
      ctx.clearRect(0, 0, W, H);
      // two lines converging to the right
      const P = [470, 165];
      const top = (x) => [x, 40 + ((x - 40) / (P[0] - 40)) * (P[1] - 40)], bot = (x) => [x, 300 - ((x - 40) / (P[0] - 40)) * (300 - P[1])];
      ctx.lineWidth = 3; ctx.strokeStyle = C.blue; ctx.beginPath(); ctx.moveTo(...top(40)); ctx.lineTo(...P); ctx.stroke();
      ctx.strokeStyle = C.orange; ctx.beginPath(); ctx.moveTo(...bot(40)); ctx.lineTo(...P); ctx.stroke();
      ctx.font = "600 17px " + U.css("--font"); ctx.fillStyle = C.blue; ctx.fillText("v = vπ", 50, 32);
      ctx.fillStyle = C.orange; ctx.fillText("π = greedy(v)", 50, 322);
      ctx.fillStyle = C.green; ctx.beginPath(); ctx.arc(...P, 8, 0, 7); ctx.fill(); ctx.fillText("v*, π*", P[0] - 20, P[1] - 18);
      // zigzag
      const n = Math.min(12, Math.floor(t * 2.2) % 16);
      let x = 70; const pts = [bot(x)];
      for (let i = 0; i < n; i++) { const up = i % 2 === 0; x += (P[0] - x) * 0.32; pts.push(up ? top(x) : bot(x)); }
      ctx.strokeStyle = C.ink; ctx.lineWidth = 2;
      for (let i = 1; i < pts.length; i++) {
        const [a, b] = [pts[i - 1], pts[i]];
        ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
        const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
        ctx.beginPath(); ctx.moveTo(...b); ctx.lineTo(b[0] - 10 * Math.cos(ang - 0.4), b[1] - 10 * Math.sin(ang - 0.4)); ctx.lineTo(b[0] - 10 * Math.cos(ang + 0.4), b[1] - 10 * Math.sin(ang + 0.4)); ctx.fill();
      }
      ctx.fillStyle = C.muted; ctx.font = "15px " + U.css("--font");
      ctx.fillText("↑ evaluation", 120, 120); ctx.fillText("↓ improvement", 150, 235);
    });
    return { start: loop.start, stop: loop.stop };
  });

  // ================================================================ DP grid world: policy evaluation / iteration / value iteration
  Deck.widget("dpgrid", (root) => {
    const preset = root.dataset.preset || "maze";
    let G, view;
    const holder = el("div");
    const info = el("div", { class: "readout", style: { fontSize: "16px", minWidth: "260px" } });
    const stepSl = U.slider({ label: "step reward", min: -1, max: 0, step: 0.01, value: preset === "sutton" ? -1 : -0.04, fmt: (v) => v.toFixed(2), oninput: (v) => { G.step = v; draw(); } });
    const gSl = U.slider({ label: "γ", min: 0.5, max: 1, step: 0.01, value: preset === "sutton" ? 1 : 0.95, fmt: (v) => v.toFixed(2), oninput: (v) => { G.gamma = v; draw(); } });
    const slipSl = U.slider({ label: "slip", min: 0, max: 0.6, step: 0.05, value: 0, fmt: (v) => v.toFixed(2), oninput: (v) => { G.slip = v; draw(); } });
    let V, pi, k = 0, msg = "", running = null;
    function build() {
      if (preset === "sutton") {
        G = new RL.Grid(4, 4, { step: -1, gamma: 1 }); G.type[0] = "T"; G.type[15] = "T";
        view = new RL.GridView(holder, G, 92, { fontSize: 17 });
      } else {
        const map = ["........", ".##..#..", "...#.#X.", ".#...#..", ".#.#...G", "...#...."];
        G = new RL.Grid(map.length, map[0].length, { step: stepSl.get(), gamma: gSl.get() });
        map.join("").split("").forEach((ch, i) => { if (ch === "#") G.type[i] = "#"; if (ch === "G") { G.type[i] = "G"; G.reward[i] = 10; } if (ch === "X") { G.type[i] = "X"; G.reward[i] = -10; } });
        view = new RL.GridView(holder, G, 66, { fontSize: 14 });
        view.cv.addEventListener("click", (e) => {
          const s = view.cellAt(e); if (s < 0) return;
          const cyc = { "": "#", "#": "G", G: "X", X: "" }; G.type[s] = cyc[G.type[s]];
          G.reward[s] = G.type[s] === "G" ? 10 : G.type[s] === "X" ? -10 : 0; msg = "grid edited — values reset"; reset();
        });
      }
      G.slip = slipSl.get();
    }
    function reset() { V = Array(G.n).fill(0); pi = G.randomPolicy(); k = 0; draw(); }
    const evalOnce = () => { V = G.evalSweep(V, pi); k++; };
    function draw() {
      view.V = V; view.pi = pi; view.o.vmax = preset === "sutton" ? 22 : 10; view.draw();
      info.innerHTML = `sweeps k = <b>${k}</b><br>${msg}`;
    }
    const acts = el("div", { class: "controls" },
      U.button("Evaluate: 1 sweep", () => { evalOnce(); msg = "v ← Bellman expectation backup (policy fixed)"; draw(); }, "primary"),
      U.button("Evaluate ⟳", () => { stopRun(); running = setInterval(() => { const old = V; evalOnce(); draw(); if (Math.max(...V.map((v, i) => Math.abs(v - old[i]))) < 1e-3 || k > 2000) { stopRun(); msg = "v<sub>π</sub> converged"; draw(); } }, 40); }),
      U.button("Improve π (greedy)", () => { const old = JSON.stringify(pi.map((p) => p.map((x) => +x.toFixed(3)))); pi = G.greedy(V); const stable = old === JSON.stringify(pi.map((p) => p.map((x) => +x.toFixed(3)))); msg = stable ? "<span class='good'>policy stable → optimal!</span>" : "π ← greedy(v)"; draw(); }, "warn"),
      U.button("Value iteration ⟳", () => { stopRun(); running = setInterval(() => { const old = V; V = G.viSweep(V); pi = G.greedy(V); k++; msg = "v ← max<sub>a</sub> backup"; draw(); if (Math.max(...V.map((v, i) => Math.abs(v - old[i]))) < 1e-3 || k > 500) { stopRun(); msg = "value iteration converged"; draw(); } }, 120); }),
      U.button("Reset", () => { stopRun(); msg = ""; reset(); }));
    const stopRun = () => { if (running) clearInterval(running); running = null; };
    const params = el("div", { class: "controls" }, ...(preset === "sutton" ? [] : [stepSl, gSl, slipSl, el("span", { class: "hint" }, "click cells to edit: wall → +10 → −10 → empty")]));
    root.append(acts, params, el("div", { class: "row" }, holder, info));
    build(); reset();
    return { stop: stopRun };
  });

  // ================================================================ 10-armed bandit
  Deck.widget("bandit", (root) => {
    const K = 10;
    let rng = U.rng(21), q = [], est, cnt, total, pulls, reveal = false;
    const newProblem = () => { q = Array.from({ length: K }, () => rng.normal()); est = Array(K).fill(0); cnt = Array(K).fill(0); total = 0; pulls = 0; reveal = false; drawHuman(); };
    const hp = el("div");
    const plotH = new U.Plot(hp, 520, 250, { xlim: [0.4, K + 0.6], ylim: [-3, 3], xticks: 10, xlabel: "arm", ylabel: "reward", yticks: 6 });
    const armRow = el("div", { class: "controls", style: { marginLeft: "58px", gap: "4px" } });
    for (let a = 0; a < K; a++) armRow.append(U.button(String(a + 1), () => pull(a)));
    const hInfo = el("div", { class: "readout" });
    const pp = el("div");
    const plotA = new U.Plot(pp, 520, 280, { xlim: [0, 1000], ylim: [0, 1.6], xlabel: "step", ylabel: "average reward", yticks: 4 });
    const epsSl = U.slider({ label: "your ε", min: 0, max: 0.5, step: 0.01, value: 0.3, fmt: (v) => v.toFixed(2) });
    root.append(el("div", { class: "row" },
      el("div", { class: "stack" }, el("b", {}, "You are the agent: click arms"), hp, armRow, el("div", { class: "controls" }, U.button("Reveal true means", () => { reveal = true; drawHuman(); }), U.button("New problem", newProblem)), hInfo),
      el("div", { class: "stack" }, el("b", {}, "ε-greedy agents, averaged over 200 problems"), el("div", { class: "controls" }, U.button("Run", runAgents, "primary"), epsSl), pp,
        el("div", { class: "legend" }, el("span", { style: "--c:" + C.red }, "ε = 0 (greedy)"), el("span", { style: "--c:" + C.blue }, "ε = 0.01"), el("span", { style: "--c:" + C.green }, "ε = 0.1"), el("span", { style: "--c:" + C.purple }, "your ε")))));
    function pull(a) { const r = q[a] + rng.normal(); cnt[a]++; est[a] += (r - est[a]) / cnt[a]; total += r; pulls++; drawHuman(r, a); }
    function drawHuman(r, a) {
      plotH.axes(); plotH.hline(0, C.muted, [2, 3]);
      const xs = [...Array(K).keys()].map((i) => i + 1);
      plotH.bars(xs, est, C.blue, 0.55, 0.8);
      if (reveal) plotH.points(xs.map((x, i) => [x, q[i]]), C.orange, 6);
      if (a != null) plotH.points([[a + 1, r]], C.ink, 4);
      const best = Math.max(...q);
      hInfo.innerHTML = `pulls: ${pulls} · average reward: <b>${pulls ? (total / pulls).toFixed(2) : "–"}</b>${reveal ? ` · best possible ≈ ${best.toFixed(2)}` : ""}<br><span style="color:${C.blue}">bars</span> = your estimates Q(a)${reveal ? `, <span style="color:${C.orange}">dots</span> = true q*(a)` : ""}`;
    }
    function runAgents() {
      const epsList = [0, 0.01, 0.1, epsSl.get()], cols = [C.red, C.blue, C.green, C.purple], T = 1000, N = 200;
      const avg = epsList.map(() => new Float64Array(T));
      const r2 = U.rng(99);
      for (let n = 0; n < N; n++) {
        const qq = Array.from({ length: K }, () => r2.normal());
        epsList.forEach((eps, j) => {
          const Q = new Float64Array(K), Nn = new Float64Array(K);
          for (let t = 0; t < T; t++) {
            const a = r2() < eps ? r2.int(K) : argmaxRand(Q, r2);
            const r = qq[a] + r2.normal(); Nn[a]++; Q[a] += (r - Q[a]) / Nn[a]; avg[j][t] += r / N;
          }
        });
      }
      plotA.axes();
      avg.forEach((a, j) => plotA.line(U.smooth(Array.from(a), 15).map((y, t) => [t + 1, y]), cols[j], 2.2));
    }
    newProblem(); plotA.axes();
    return {};
  });

  // ================================================================ Driving home: MC vs TD targets
  Deck.widget("driving", (root) => {
    const S = ["leave office", "reach car (rain)", "exit highway", "behind truck", "home street", "arrive home"];
    const elapsed = [0, 5, 20, 30, 40, 43], togo = [30, 35, 15, 10, 3, 0], total = elapsed.map((e, i) => e + togo[i]);
    let mode = 0;
    const plot = new U.Plot(el("div"), 820, 340, { xlim: [-0.4, 5.4], ylim: [25, 46], xticks: 6, xfmt: (i) => S[Math.round(i)] || "", ylabel: "predicted total travel time (min)", yticks: 5, pad: [18, 16, 44, 70] });
    const out = el("p", { class: "small" });
    root.append(el("div", { class: "controls" }, U.toggle(["Predictions", "Monte-Carlo updates", "TD updates"], (i) => { mode = i; draw(); })), plot.cv, out);
    function arrow(x0, y0, x1, y1, col) {
      plot.line([[x0, y0], [x1, y1]], col, 2.5);
      const c = plot.ctx, X1 = plot.X(x1), Y1 = plot.Y(y1), X0 = plot.X(x0), Y0 = plot.Y(y0), a = Math.atan2(Y1 - Y0, X1 - X0);
      c.fillStyle = col; c.beginPath(); c.moveTo(X1, Y1); c.lineTo(X1 - 11 * Math.cos(a - 0.4), Y1 - 11 * Math.sin(a - 0.4)); c.lineTo(X1 - 11 * Math.cos(a + 0.4), Y1 - 11 * Math.sin(a + 0.4)); c.fill();
    }
    function draw() {
      plot.axes();
      plot.hline(43, C.green, [6, 4], "actual outcome: 43 min");
      const pts = total.map((y, i) => [i, y]);
      plot.line(pts, C.ink, 2.5); plot.points(pts, C.ink, 5);
      for (let i = 0; i < 5; i++) {
        if (mode === 1 && Math.abs(total[i] - 43) > 0.1) arrow(i, total[i], i, 43 + (total[i] < 43 ? -0.4 : 0.4), C.green);
        if (mode === 2 && Math.abs(total[i] - total[i + 1]) > 0.1) arrow(i, total[i], i, total[i + 1] + (total[i] < total[i + 1] ? -0.4 : 0.4), C.blue);
      }
      out.innerHTML = ["Each state: elapsed time + <i>predicted</i> time to go. The prediction changes as the journey unfolds.",
        "<b style='color:#2b8a3e'>Monte-Carlo:</b> wait until you arrive home, then move <b>every</b> prediction towards the <b style='color:#2b8a3e'>actual outcome</b> (43), which is real experience.",
        "<b style='color:#1f5fae'>TD:</b> move each prediction towards the <b style='color:#1f5fae'>next prediction</b> (an estimate of the next state), immediately, without waiting for the end."][mode];
    }
    draw();
    return {};
  });

  // ================================================================ Random walk: MC vs TD(0)
  Deck.widget("randomwalk", (root) => {
    const N = 5, truth = [1, 2, 3, 4, 5].map((i) => i / 6), names = ["A", "B", "C", "D", "E"];
    // chain picture
    const pic = U.svg("svg", { width: 760, height: 70, viewBox: "0 0 760 70" });
    const px = (i) => 80 + i * 120;
    pic.append(U.svg("rect", { x: 10, y: 20, width: 40, height: 30, fill: "#8a8f99" }), U.svg("rect", { x: 710, y: 20, width: 40, height: 30, fill: "#8a8f99" }));
    pic.append(U.svg("line", { x1: 50, y1: 35, x2: 710, y2: 35, stroke: C.ink, "stroke-width": 2 }));
    for (let i = 0; i < N; i++) { pic.append(U.svg("circle", { cx: px(i) + 30, cy: 35, r: 20, fill: "#fff", stroke: C.ink, "stroke-width": 2.5 }), U.svg("text", { x: px(i) + 30, y: 41, "text-anchor": "middle", "font-size": 17, "font-weight": 700 }, names[i])); }
    pic.append(U.svg("text", { x: 690, y: 15, "font-size": 15, fill: C.green, "font-weight": 700 }, "+1"), U.svg("text", { x: 58, y: 15, "font-size": 15, fill: C.muted }, "0"), U.svg("text", { x: 330, y: 68, "font-size": 14, fill: C.muted }, "start at C · move left/right with prob ½"));
    const pv = el("div"), pe = el("div");
    const plotV = new U.Plot(pv, 440, 280, { xlim: [-0.4, 4.4], ylim: [0, 1], xticks: 5, xfmt: (i) => names[Math.round(i)] || "", ylabel: "estimated value", yticks: 5 });
    const plotE = new U.Plot(pe, 440, 280, { xlim: [0, 100], ylim: [0, 0.25], xlabel: "episodes", ylabel: "RMS error (avg 100 runs)", yticks: 5 });
    const aTD = U.slider({ label: "α<sub>TD</sub>", min: 0.01, max: 0.3, step: 0.01, value: 0.1, fmt: (v) => v.toFixed(2) });
    const aMC = U.slider({ label: "α<sub>MC</sub>", min: 0.01, max: 0.3, step: 0.01, value: 0.02, fmt: (v) => v.toFixed(2) });
    const cnt = el("span", { class: "val" });
    root.append(pic, el("div", { class: "controls" }, U.button("1 episode", () => { run(1); }, "primary"), U.button("+10", () => run(10)), U.button("+100", () => run(100)), U.button("Reset", reset), aTD, aMC, U.button("Learning curves", curves, "warn"), cnt),
      el("div", { class: "row" }, pv, pe), el("div", { class: "legend" }, el("span", { style: "--c:" + C.blue }, "TD(0)"), el("span", { style: "--c:" + C.orange }, "Monte-Carlo"), el("span", { style: "--c:" + C.ink }, "true value")));
    const rng = U.rng(17);
    let Vtd, Vmc, eps;
    function episode(r) { let s = 2; const traj = [s]; let R = 0; while (true) { s += r() < 0.5 ? -1 : 1; if (s < 0) { R = 0; break; } if (s >= N) { R = 1; break; } traj.push(s); } return [traj, R]; }
    function learn(Vt, Vm, traj, R, at, am) {
      for (let i = 0; i < traj.length; i++) { const s = traj[i], nxt = i + 1 < traj.length ? Vt[traj[i + 1]] : R; Vt[s] += at * (nxt - Vt[s]); }
      for (const s of traj) Vm[s] += am * (R - Vm[s]);
    }
    function reset() { Vtd = Array(N).fill(0.5); Vmc = Array(N).fill(0.5); eps = 0; drawV(); }
    function run(n) { for (let i = 0; i < n; i++) { const [t, R] = episode(rng); learn(Vtd, Vmc, t, R, aTD.get(), aMC.get()); eps++; } drawV(); }
    function drawV() {
      plotV.axes(); const xs = [0, 1, 2, 3, 4];
      plotV.line(xs.map((i) => [i, truth[i]]), C.ink, 2); plotV.points(xs.map((i) => [i, truth[i]]), C.ink, 4);
      plotV.line(xs.map((i) => [i, Vtd[i]]), C.blue, 2.5); plotV.points(xs.map((i) => [i, Vtd[i]]), C.blue, 4);
      plotV.line(xs.map((i) => [i, Vmc[i]]), C.orange, 2.5); plotV.points(xs.map((i) => [i, Vmc[i]]), C.orange, 4);
      cnt.textContent = `episodes: ${eps}`;
    }
    function curves() {
      const R = 100, E = 100, rt = new Float64Array(E), rm = new Float64Array(E), r2 = U.rng(5);
      for (let k = 0; k < R; k++) {
        const vt = Array(N).fill(0.5), vm = Array(N).fill(0.5);
        for (let e = 0; e < E; e++) {
          const [t, Rr] = episode(r2); learn(vt, vm, t, Rr, aTD.get(), aMC.get());
          rt[e] += Math.sqrt(vt.reduce((a, v, i) => a + (v - truth[i]) ** 2, 0) / N) / R;
          rm[e] += Math.sqrt(vm.reduce((a, v, i) => a + (v - truth[i]) ** 2, 0) / N) / R;
        }
      }
      plotE.axes(); plotE.line(Array.from(rt).map((y, i) => [i + 1, y]), C.blue, 2.5); plotE.line(Array.from(rm).map((y, i) => [i + 1, y]), C.orange, 2.5);
      plotE.text(60, 0.23, `TD α=${aTD.get()}  MC α=${aMC.get()}`, C.muted);
    }
    reset(); plotE.axes();
    return {};
  });

  // ================================================================ Unified view: which part of the tree does each method back up?
  Deck.widget("unified", (root) => {
    const W = 660, H = 380, svg = U.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const desc = el("div", { class: "box", style: { flex: 1 } });
    let mode = 0;
    const tog = U.toggle(["Temporal-Difference", "Monte-Carlo", "Dynamic Programming", "Exhaustive search"], (i) => { mode = i; draw(); });
    root.append(el("div", { class: "controls" }, tog), el("div", { class: "row" }, svg, desc));
    // binary tree depth 4
    const nodes = [], edges = [];
    (function build(id, depth, x, w, path) {
      nodes.push({ id, depth, x, y: 30 + depth * 80, path });
      if (depth === 4) return;
      [0, 1].forEach((b) => { const cid = id + b; edges.push([id, cid, depth]); build(cid, depth + 1, x + (b ? w / 4 : -w / 4), w / 2, path + b); });
    })("r", 0, W / 2, W - 40, "");
    const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
    const samplePath = "r0110";
    const inSel = (id, depth) => {
      if (mode === 0) return depth <= 1 && samplePath.startsWith(id);
      if (mode === 1) return samplePath.startsWith(id);
      if (mode === 2) return depth <= 1;
      return true;
    };
    const texts = [
      ["TD(0)", "Samples <b>one step</b>, then <b>bootstraps</b> from the estimate V(S<sub>t+1</sub>).", "$\\est{V(S_t)} \\leftarrow \\est{V(S_t)} + \\alpha\\,[\\rew{R_{t+1}} + \\gamma \\nxt{V(S_{t+1})} - \\est{V(S_t)}]$"],
      ["Monte-Carlo", "Samples a <b>whole episode</b> to termination. No bootstrapping.", "$\\est{V(S_t)} \\leftarrow \\est{V(S_t)} + \\alpha\\,[\\rew{G_t} - \\est{V(S_t)}]$"],
      ["Dynamic programming", "Full <b>one-step</b> look-ahead over <b>all</b> successors — needs the model. Bootstraps.", "$\\est{V(S_t)} \\leftarrow \\mathbb{E}_\\pi[\\rew{R_{t+1}} + \\gamma \\nxt{V(S_{t+1})}]$"],
      ["Exhaustive search", "Every branch to the end. Exact — and exponentially expensive.", "$\\text{cost} \\sim |\\mathcal{A}|^{T}$"],
    ];
    function draw() {
      svg.innerHTML = "";
      for (const [a, b, d] of edges) { const A = byId[a], B = byId[b], on = inSel(b, d + 1); svg.append(U.svg("line", { x1: A.x, y1: A.y, x2: B.x, y2: B.y, stroke: on ? C.orange : "#d0d3d9", "stroke-width": on ? 4 : 1.5 })); }
      for (const n of nodes) {
        const on = inSel(n.id, n.depth);
        if (n.depth === 4) svg.append(U.svg("rect", { x: n.x - 9, y: n.y - 9, width: 18, height: 18, fill: on ? C.orange : "#8a8f99" }));
        else svg.append(U.svg("circle", { cx: n.x, cy: n.y, r: n.depth === 0 ? 14 : 10, fill: n.depth === 0 ? "#fff4d6" : on ? "#fff" : "#f3f3f3", stroke: on ? C.orange : "#b5b9c1", "stroke-width": on ? 3 : 1.5 }));
      }
      svg.append(U.svg("text", { x: W / 2 + 22, y: 36, "font-size": 15, fill: C.ink }, "Sₜ"), U.svg("text", { x: 8, y: H - 8, "font-size": 13, fill: C.muted }, "■ terminal"));
      const [h, t, eq] = texts[mode];
      desc.innerHTML = `<h3>${h}</h3><p>${t}</p><p style="font-size:19px">${eq}</p><table class="t" style="font-size:16px;margin-top:10px"><tr><th></th><th>bootstraps</th><th>samples</th></tr><tr><td>MC</td><td>✗</td><td>✓</td></tr><tr><td>TD</td><td>✓</td><td>✓</td></tr><tr><td>DP</td><td>✓</td><td>✗</td></tr></table>`;
      renderMathInElement(desc, Object.assign({ delimiters: [{ left: "$", right: "$", display: false }] }, U.KATEX));
    }
    draw();
    return {};
  });

  // ================================================================ n-step target
  Deck.widget("nstep", (root) => {
    const T = 8, W = 980, H = 150, svg = U.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const out = el("div", { class: "eq", style: { fontSize: "22px", minHeight: "44px" } });
    const sl = U.slider({ label: "n", min: 1, max: T, step: 1, value: 1, fmt: (v) => (v === T ? "∞ (to end)" : String(v)), oninput: draw });
    root.append(el("div", { class: "controls" }, sl), svg, out);
    function draw() {
      const n = sl.get(); svg.innerHTML = "";
      for (let i = 0; i <= T; i++) {
        const x = 50 + i * 112, used = i < n, boot = i === n && n < T, term = i === T;
        if (i < T) {
          svg.append(U.svg("line", { x1: x + 22, y1: 60, x2: x + 90, y2: 60, stroke: used ? C.green : "#c9ccd3", "stroke-width": used ? 4 : 2 }));
          svg.append(U.svg("rect", { x: x + 34, y: 84, width: 46, height: 28, rx: 6, fill: used ? C.green : "#eceef1" }));
          svg.append(U.svg("text", { x: x + 57, y: 104, "text-anchor": "middle", "font-size": 15, fill: used ? "#fff" : C.muted, "font-weight": 700 }, `R${i + 1}`));
        }
        if (term) svg.append(U.svg("rect", { x: x - 16, y: 44, width: 32, height: 32, fill: n === T ? C.green : "#8a8f99" }));
        else svg.append(U.svg("circle", { cx: x, cy: 60, r: 20, fill: boot ? C.blue : i === 0 ? "#fff4d6" : "#fff", stroke: boot ? C.blue : C.ink, "stroke-width": 2.5 }));
        svg.append(U.svg("text", { x, y: 26, "text-anchor": "middle", "font-size": 15, fill: boot ? C.blue : C.ink, "font-weight": boot ? 700 : 400 }, i === T ? "end" : `S${i ? "t+" + i : "t"}`));
        if (boot) svg.append(U.svg("text", { x, y: 140, "text-anchor": "middle", "font-size": 15, fill: C.blue, "font-weight": 700 }, "V(·) bootstrap"));
      }
      const terms = []; for (let k = 1; k <= Math.min(n, 3); k++) terms.push(k === 1 ? "\\rew{R_{t+1}}" : `\\gamma${k > 2 ? "^" + (k - 1) : ""} \\rew{R_{t+${k}}}`);
      const rs = n <= 3 ? terms.join(" + ") : terms.join(" + ") + ` + \\dots + \\gamma^{${n - 1}} \\rew{R_{t+${n}}}`;
      const tex = n === T ? `\\tgt{G_t} = \\rew{R_{t+1}} + \\gamma \\rew{R_{t+2}} + \\dots + \\gamma^{T-1}\\rew{R_T} \\quad \\text{(Monte-Carlo)}` : `\\tgt{G_t^{(${n})}} = ${rs} + \\gamma${n > 1 ? "^{" + n + "}" : ""} \\nxt{V(S_{t+${n}})}${n === 1 ? "\\quad \\text{(TD(0))}" : ""}`;
      katex.render(tex, out, Object.assign({ throwOnError: false, displayMode: true }, U.KATEX));
    }
    draw();
    return {};
  });


  // ================================================================ TD control, one update at a time (SARSA vs Q-learning)
  Deck.widget("tdstep", (root) => {
    const algo = root.dataset.algo || "sarsa", SARSA = algo === "sarsa";
    const R_ = 4, C_ = 6, z = 86, start = 18, goal = 23, cliff = new Set([19, 20, 21, 22]);
    const gamma = 0.99, alpha = 0.5, eps = 0.1;
    const COL = { sa: C.orange, r: C.green, next: C.blue, tgt: C.teal };   // same colour semantics as the slides
    const cv = U.canvas(C_ * z + 2, R_ * z + 2), ctx = cv.ctx;
    const stageBox = el("ol", { class: "small", style: { margin: "0 0 6px", paddingLeft: "24px" } });
    const rule = el("div", { style: { fontSize: "18px", margin: "2px 0 6px" } });
    const nums = el("div", { class: "readout", style: { fontSize: "15px", minHeight: "118px", lineHeight: "1.6" } });
    const note = el("p", { class: "small", style: { margin: "6px 0 0", minHeight: "56px" } });
    let auto = false, forceExplore = false, rate = 10, nUpdates = 0, nEpisodes = 0;
    const spd = U.slider({ label: "auto speed", min: 1, max: 500, step: 1, value: 10, width: 110, fmt: (v) => v + " upd/s", oninput: (v) => (rate = v) });
    const ms = (normal) => (auto ? Math.min(normal, 700 / rate) : normal);   // animations shrink with the auto speed
    const autoBtn = U.button("▶ Auto", () => { auto = !auto; autoBtn.innerHTML = auto ? "❚❚ Pause" : "▶ Auto"; acc = 0; if (!auto) render(); });
    root.append(el("div", { class: "controls" }, U.button("Next step ▸", () => advance(), "primary"), autoBtn,
      U.button("Explore next choice", () => { forceExplore = true; note.innerHTML = "<b>The next ε-greedy choice will explore</b> — and we'll make it the worst one (the lowest Q), to see its effect clearly."; }, "warn"),
      U.button("Reset", () => reset()), spd),
      el("div", { class: "row" }, el("div", { class: "stack", style: { flex: "none" } }, cv, el("div", { class: "legend", style: { maxWidth: "520px" } },
        el("span", { style: "--c:" + COL.sa }, "Q(S,A) being updated"), el("span", { style: "--c:" + COL.r }, "reward R"),
        el("span", { style: "--c:" + COL.next }, SARSA ? "Q(S′,A′): next action (will take)" : "max Q(S′,·): best next action"), el("span", { style: "--c:" + COL.next }, "- - greedy route from start"))),
        el("div", { class: "col" }, rule, stageBox, nums, note)));
    const e = (t) => `<span class="c-est">${t}</span>`, r_ = (t) => `<span class="c-rew">${t}</span>`, n_ = (t) => `<span class="c-next">${t}</span>`, t_ = (t) => `<span class="c-tgt">${t}</span>`;
    const STAGES = SARSA
      ? [`In ${e("S")}, choose ${e("A")} (ε-greedy)`, `Take ${e("A")} → observe ${r_("R")} and ${n_("S′")}`, `In ${n_("S′")}, choose ${n_("A′")} (ε-greedy): the action we will really take`, `${t_("TD target")} = ${r_("R")} + γ·${n_("Q(S′, A′)")}`, `Update ${e("Q(S,A)")} towards the ${t_("target")}`, `Move on: ${e("S")} ← ${n_("S′")}, ${e("A")} ← ${n_("A′")}`]
      : [`In ${e("S")}, choose ${e("A")} (ε-greedy)`, `Take ${e("A")} → observe ${r_("R")} and ${n_("S′")}`, `In ${n_("S′")}, look at all actions: take ${n_("max Q(S′, ·)")}`, `${t_("TD target")} = ${r_("R")} + γ·${n_("max Q(S′, ·)")}`, `Update ${e("Q(S,A)")} towards the ${t_("target")}`, `Move on: ${e("S")} ← ${n_("S′")}, choose next ${e("A")} fresh (ε-greedy)`];
    katex.render(SARSA
      ? String.raw`\est{Q(S,A)} \leftarrow \est{Q(S,A)} + \alpha\big[\rew{R} + \gamma\,\nxt{Q(S',A')} - \est{Q(S,A)}\big]`
      : String.raw`\est{Q(S,A)} \leftarrow \est{Q(S,A)} + \alpha\big[\rew{R} + \gamma\,\nxt{\max_{a'} Q(S',a')} - \est{Q(S,A)}\big]`, rule, Object.assign({ throwOnError: false }, U.KATEX));
    STAGES.forEach((t) => stageBox.append(el("li", { html: t })));

    const rng = U.rng(algo === "sarsa" ? 5 : 5);
    const rc = (s) => [Math.floor(s / C_), s % C_];
    const move = (s, a) => { const [r, c] = rc(s), [dr, dc] = RL.ACTIONS[a]; return U.clamp(r + dr, 0, R_ - 1) * C_ + U.clamp(c + dc, 0, C_ - 1); };
    const argmax = (q) => q.indexOf(Math.max(...q));
    function pretrain() {   // same partially-trained Q for both slides
      const r = U.rng(42), Q = Array.from({ length: R_ * C_ }, () => [0, 0, 0, 0]);
      for (let e = 0; e < 60; e++) {
        let s = start;
        for (let t = 0; t < 60; t++) {
          const a = r() < 0.2 ? r.int(4) : argmax(Q[s]); const s2 = move(s, a);
          const rew = cliff.has(s2) ? -100 : -1, done = cliff.has(s2) || s2 === goal;
          Q[s][a] += 0.5 * (rew + (done ? 0 : gamma * Math.max(...Q[s2])) - Q[s][a]);
          if (done) break; s = s2;
        }
      }
      return Q;
    }
    let Q, S, A, R, S2, A2, target, delta, oldQ, newQ, stage, explored, explored2, agentXY, tween, done, qShown;
    function choose(s) {   // ε-greedy; returns [action, wasExploratory]
      if (forceExplore) { forceExplore = false; const q = Q[s]; return [q.indexOf(Math.min(...q)), true]; }   // worst-case exploration
      if (rng() < eps) { const g = argmax(Q[s]), others = [0, 1, 2, 3].filter((a) => a !== g); return [others[rng.int(3)], true]; }
      return [argmax(Q[s]), false];
    }
    const arrow = ["↑", "→", "↓", "←"];
    const f1 = (x) => (Math.abs(x) < 0.05 ? "0.0" : x.toFixed(1));
    // start mid-episode, right next to the cliff
    function reset() { nUpdates = 0; nEpisodes = 0; Q = pretrain(); S = 13; forceExplore = false; [A, explored] = choose(S); stage = 0; qShown = null; agentXY = rc(S); tween = null; render(); }
    function advance() {
      if (tween) return;
      if (stage === 5) {                                   // move on to the next update
        if (done) { S = start; [A, explored] = choose(S); }
        else if (SARSA) { S = S2; A = A2; explored = explored2; }
        else { S = S2; [A, explored] = choose(S); }
        agentXY = rc(S); stage = 0; qShown = null; R = null; A2 = null; render(); return;
      }
      stage++;
      if (stage === 1) {                                   // take A
        S2 = move(S, A); R = cliff.has(S2) ? -100 : -1; done = cliff.has(S2) || S2 === goal; explored2 = false;
        if (ms(800) > 20) tween = { from: rc(S), to: rc(S2), t0: performance.now(), ms: ms(800) }; else agentXY = rc(S2);   // very fast: no movement animation
      }
      if (stage === 2 && !done) {
        if (SARSA) [A2, explored2] = choose(S2);
        else A2 = argmax(Q[S2]);
      }
      if (stage === 3) { target = R + gamma * (done ? 0 : Q[S2][A2]); oldQ = Q[S][A]; delta = target - oldQ; newQ = oldQ + alpha * delta; }
      if (stage === 4) { qShown = { s: S, a: A, from: oldQ, to: newQ, t0: performance.now(), ms: ms(1200) }; Q[S][A] = newQ; nUpdates++; if (done) nEpisodes++; }
      render();
    }
    const cellXY = (s) => { const [r, c] = rc(s); return [1 + c * z, 1 + r * z]; };
    function tri(s, a, fill, stroke, lw) {
      const [x, y] = cellXY(s), cx = x + z / 2, cy = y + z / 2, P = [[x, y], [x + z, y], [x + z, y + z], [x, y + z]];
      const p1 = P[a], p2 = P[(a + 1) % 4];
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(...p1); ctx.lineTo(...p2); ctx.closePath();
      if (fill) { ctx.fillStyle = fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw || 3; ctx.stroke(); }
    }
    const numPos = (s, a) => { const [x, y] = cellXY(s), o = z * 0.3; return [[x + z / 2, y + o * 0.62], [x + z - o * 0.62, y + z / 2], [x + z / 2, y + z - o * 0.62], [x + o * 0.62, y + z / 2]][a]; };
    function draw(now) {
      ctx.clearRect(0, 0, cv.W, cv.H);
      for (let s = 0; s < R_ * C_; s++) {
        const [x, y] = cellXY(s);
        if (cliff.has(s)) { ctx.fillStyle = "#6b3a3a"; ctx.fillRect(x, y, z, z); ctx.fillStyle = "#fff"; ctx.font = `13px ${U.css("--font")}`; ctx.textAlign = "center"; ctx.fillText("cliff −100", x + z / 2, y + z / 2 + 4); continue; }
        if (s === goal) { ctx.fillStyle = "#c9ebd0"; ctx.fillRect(x, y, z, z); ctx.fillStyle = C.ink; ctx.font = `600 15px ${U.css("--font")}`; ctx.textAlign = "center"; ctx.fillText("goal", x + z / 2, y + z / 2 + 5); continue; }
        for (let a = 0; a < 4; a++) {
          let q = Q[s][a];
          if (qShown && qShown.s === s && qShown.a === a) { const k = U.clamp((now - qShown.t0) / qShown.ms, 0, 1); q = U.lerp(qShown.from, qShown.to, k); }
          tri(s, a, U.divColor(q, 25), "rgba(0,0,0,.08)", 1);
        }
        ctx.strokeStyle = "#b9bec8"; ctx.lineWidth = 1; ctx.strokeRect(x, y, z, z);
        if (s === start) { ctx.fillStyle = C.muted; ctx.font = `12px ${U.css("--font")}`; ctx.textAlign = "left"; ctx.fillText("start", x + 3, y + 12); }
      }
      // highlights
      const hl = [];
      if (stage >= 0) hl.push([S, A, COL.sa]);
      const scanning = !auto && !SARSA && stage === 2 && !done && now - stateT < 1600;   // Q-learning: look at every action in S′
      if (scanning) hl.push([S2, Math.max(0, Math.floor((now - stateT) / 400)) % 4, "#8a8f99"]);
      else if (stage >= 2 && !done && A2 != null) hl.push([S2, A2, COL.next]);
      for (const [s, a, col] of hl) tri(s, a, null, col, 4);
      // numbers
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      for (let s = 0; s < R_ * C_; s++) {
        if (cliff.has(s) || s === goal) continue;
        for (let a = 0; a < 4; a++) {
          let q = Q[s][a];
          if (qShown && qShown.s === s && qShown.a === a) { const k = U.clamp((now - qShown.t0) / qShown.ms, 0, 1); q = U.lerp(qShown.from, qShown.to, k); }
          const [px, py] = numPos(s, a), big = hl.some(([hs, ha]) => hs === s && ha === a);
          ctx.font = `${big ? "700 15px" : "12px"} ${U.css("--mono")}`; ctx.fillStyle = C.ink; ctx.fillText(f1(q), px, py);
        }
      }
      ctx.textBaseline = "alphabetic";
      // greedy route from the start: what the agent would do now without exploring
      if (nUpdates > 0) {
        const path = [start]; let st = start;
        for (let t = 0; t < 24; t++) { const nx = move(st, argmax(Q[st])); if (nx === st || path.includes(nx)) break; path.push(nx); st = nx; if (cliff.has(nx) || nx === goal) break; }
        ctx.save(); ctx.strokeStyle = COL.next; ctx.lineWidth = 5; ctx.setLineDash([10, 7]); ctx.globalAlpha = 0.75; ctx.lineCap = "round"; ctx.beginPath();
        path.forEach((c, k) => { const [x, y] = cellXY(c); k ? ctx.lineTo(x + z / 2, y + z / 2) : ctx.moveTo(x + z / 2, y + z / 2); }); ctx.stroke(); ctx.restore();
      }
      // agent
      let [ar, ac] = agentXY;
      if (tween) { const k = U.clamp((now - tween.t0) / tween.ms, 0, 1), e = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2; ar = U.lerp(tween.from[0], tween.to[0], e); ac = U.lerp(tween.from[1], tween.to[1], e); if (k >= 1) { agentXY = tween.to; tween = null; render(); } }
      ctx.beginPath(); ctx.arc(1 + ac * z + z / 2, 1 + ar * z + z / 2, 12, 0, 7); ctx.fillStyle = "rgba(29,36,51,.85)"; ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2.5; ctx.stroke();
      if (stage >= 1 && !tween && R != null) {
        const [x, y] = cellXY(S2), t = `R = ${R}`, bx = x + z / 2, by = y + z / 2 - 30;
        ctx.font = `700 17px ${U.css("--font")}`; const w = ctx.measureText(t).width + 14;
        ctx.fillStyle = COL.r; ctx.beginPath(); ctx.roundRect(bx - w / 2, by - 12, w, 24, 12); ctx.fill();
        ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.fillText(t, bx, by + 6);
      }
    }
    let stateT = 0;
    let quiet = false;
    function render() {
      if (quiet) return;
      stateT = performance.now();
      [...stageBox.children].forEach((li, i) => { li.style.fontWeight = i === stage ? 700 : 400; li.style.color = i === stage ? "var(--accent)" : i < stage ? C.ink : C.muted; });
      const span = (c, t) => `<b style="color:${c}">${t}</b>`;
      const lines = [`S = cell ${rc(S).join(",")}, A = ${span(COL.sa, arrow[A])}${explored ? " <i>(explored!)</i>" : ""} &nbsp; Q(S,A) = ${span(COL.sa, f1(stage >= 4 ? oldQ : Q[S][A]))}`];
      if (stage >= 1) lines.push(`R = ${span(COL.r, R)}, S′ = cell ${rc(S2).join(",")}${done ? " <i>(terminal: nothing to bootstrap from)</i>" : ""}`);
      if (stage >= 2 && !done) lines.push(SARSA ? `A′ = ${span(COL.next, arrow[A2])}${explored2 ? " <i>(explored!)</i>" : ""} &nbsp; Q(S′,A′) = ${span(COL.next, f1(Q[S2][A2]))}` : `max Q(S′,·) = max(${Q[S2].map(f1).join(", ")}) = ${span(COL.next, f1(Q[S2][A2]))} (${arrow[A2]})`);
      if (stage >= 3) lines.push(`target = ${span(COL.r, R)} + ${gamma}·${span(COL.next, done ? "0" : f1(Q[S2][A2]))} = ${span(COL.tgt, f1(target))}`);
      if (stage >= 4) lines.push(`δ = ${f1(target)} − ${f1(oldQ)} = ${f1(delta)} &nbsp;→&nbsp; Q(S,A) ← ${f1(oldQ)} + ${alpha}·(${f1(delta)}) = ${span(COL.sa, f1(newQ))}`);
      nums.innerHTML = lines.join("<br>") + (nUpdates ? `<br><span class="muted">updates: ${nUpdates} · episodes finished: ${nEpisodes}</span>` : "");
      const notes = SARSA ? [
        "The agent picks A with ε-greedy (usually the best-looking action).",
        "The environment answers with a reward and the next state.",
        "SARSA commits to its next action <b>now</b> — including any exploration. Its value is what goes into the target.",
        "Target = what just happened + the discounted value of <b>what we will actually do next</b>.",
        "Move Q(S,A) a fraction α of the way towards the target.",
        "A′ is carried over: SARSA really takes the action it used in the target → it learns the value of its own (exploring) behaviour.",
      ] : [
        "The agent picks A with ε-greedy (usually the best-looking action).",
        "The environment answers with a reward and the next state.",
        "Q-learning does not care what it will do next: it assumes the <b>best</b> action from S′ (greedy target policy).",
        "Target = what just happened + the discounted value of <b>the best thing we could do next</b>.",
        "Move Q(S,A) a fraction α of the way towards the target.",
        "The next action is chosen fresh with ε-greedy and may explore — but that never enters the target → learns the optimal Q (off-policy).",
      ];
      note.innerHTML = notes[stage];
    }
    let last = 0, acc = 0;
    const loop = U.loop((dt, now) => {
      if (auto) {
        acc += dt * rate * 6;                      // 6 stages per update
        let n = 0;
        while (acc >= 1 && n < 600) {
          if (tween) { if (now - tween.t0 >= tween.ms) { agentXY = tween.to; tween = null; } else break; }
          acc -= 1; n++; quiet = true; advance(); quiet = false;
        }
        if (n) render();
        if (acc > 600) acc = 600;
      }
      draw(now);
    });
    reset();
    return { start: loop.start, stop() { loop.stop(); auto = false; autoBtn.innerHTML = "▶ Auto"; } };
  });

  // ================================================================ Cliff walking: SARSA vs Q-learning
  Deck.widget("cliff", (root) => {
    const mk = () => { const g = new RL.Grid(4, 12, { step: -1, gamma: 1 }); g.start = 36; for (let c = 1; c < 11; c++) { g.type[36 + c] = "C"; g.reward[36 + c] = -99; } g.type[47] = "G"; g.reward[47] = 0; return g; };
    const G = mk();
    const vS = new RL.GridView(el("div"), G, 38, { fontSize: 11, showValues: true, showArrows: false, labels: { 36: "S", 47: "G" } });
    const vQ = new RL.GridView(el("div"), G, 38, { fontSize: 11, showValues: true, showArrows: false, labels: { 36: "S", 47: "G" } });
    const pp = el("div");
    const plot = new U.Plot(pp, 470, 300, { xlim: [0, 500], ylim: [-100, 0], xlabel: "episodes", ylabel: "sum of rewards / episode (smoothed)", yticks: 5 });
    const epsSl = U.slider({ label: "ε", min: 0, max: 0.3, step: 0.01, value: 0.1, fmt: (v) => v.toFixed(2) });
    let run = false;
    const runBtn = U.button("▶ Train", () => { run = !run; runBtn.innerHTML = run ? "❚❚ Pause" : "▶ Train"; }, "primary");
    const info = el("div", { class: "small muted" });
    root.append(el("div", { class: "controls" }, runBtn, U.button("Reset", reset), epsSl, info),
      el("div", { class: "row" }, el("div", { class: "stack", style: { gap: "4px" } },
        el("b", { style: { color: C.blue } }, "SARSA (on-policy)"), vS.cv, el("b", { style: { color: C.orange } }, "Q-learning (off-policy)"), vQ.cv), el("div", {}, pp,
          el("div", { class: "legend" }, el("span", { style: "--c:" + C.blue }, "SARSA"), el("span", { style: "--c:" + C.orange }, "Q-learning")))));
    const rng = U.rng(3);
    let QS, QQ, rS, rQ, ep;
    function reset() { QS = Array.from({ length: 48 }, () => [0, 0, 0, 0]); QQ = Array.from({ length: 48 }, () => [0, 0, 0, 0]); rS = []; rQ = []; ep = 0; draw(); }
    const pol = (Q, s, eps) => (rng() < eps ? rng.int(4) : argmaxRand(Q[s], rng));
    function episode(Q, sarsa, eps, alpha = 0.5) {
      let s = G.start, a = pol(Q, s, eps), R = 0;
      for (let t = 0; t < 500; t++) {
        const [s2, r, done] = G.sample(s, a, rng); R += r;
        const a2 = pol(Q, s2, eps);
        const target = done ? r : r + (sarsa ? Q[s2][a2] : Math.max(...Q[s2]));
        Q[s][a] += alpha * (target - Q[s][a]);
        if (done) break;
        s = s2; a = a2;
      }
      return R;
    }
    function greedyPath(Q) { let s = G.start; const p = [s]; for (let t = 0; t < 40; t++) { const a = argmax(Q[s]); const t2 = G.move(s, a); if (G.type[t2] === "C") { p.push(t2); break; } s = t2; p.push(s); if (s === 47) break; } return p; }
    function draw() {
      for (const [v, Q, col] of [[vS, QS, C.blue], [vQ, QQ, C.orange]]) { v.V = Q.map((q) => Math.max(...q)); v.o.vmax = 20; v.paths = [{ cells: greedyPath(Q), color: col, width: 5 }]; v.draw(); }
      const xmax = Math.max(500, ep);
      plot.o.xlim = [0, xmax]; plot.axes(); plot.hline(-13, C.muted, [4, 4], "−13 optimal (cliff edge) · −17 safe path"); plot.hline(-17, C.muted, [2, 4]);
      plot.line(U.smooth(rS, 20).map((y, i) => [i + 1, y]), C.blue, 2.2); plot.line(U.smooth(rQ, 20).map((y, i) => [i + 1, y]), C.orange, 2.2);
      info.textContent = `episode ${ep}`;
    }
    let acc = 0;
    const loop = U.loop((dt) => { if (!run) return; acc += dt; if (acc < 0.03) return; acc = 0; const eps = epsSl.get(); for (let k = 0; k < 2; k++) { rS.push(episode(QS, true, eps)); rQ.push(episode(QQ, false, eps)); ep++; } draw(); if (ep >= 1000) { run = false; runBtn.innerHTML = "▶ Train"; } });
    reset();
    return { start: loop.start, stop() { loop.stop(); run = false; runBtn.innerHTML = "▶ Train"; } };
  });

  // ================================================================ curse of dimensionality
  Deck.widget("curse", (root) => {
    const out = el("div", { style: { fontSize: "26px", margin: "10px 0" } });
    const d = U.slider({ label: "state dimensions", min: 1, max: 12, step: 1, value: 4, oninput: draw });
    const b = U.slider({ label: "bins per dimension", min: 2, max: 100, step: 1, value: 10, oninput: draw });
    const a = U.slider({ label: "actions", min: 2, max: 20, step: 1, value: 2, oninput: draw });
    const bar = el("div", { style: { height: "26px", background: C.orange, borderRadius: "6px", transition: "width .2s" } });
    root.append(el("div", { class: "controls" }, d, b, a), out, el("div", { style: { background: C.panel, borderRadius: "6px", width: "100%" } }, bar), el("p", { class: "hint" }, "bar: log scale, up to 10⁸⁰ (≈ atoms in the observable universe)"));
    function draw() {
      const lg = d.get() * Math.log10(b.get()) + Math.log10(a.get());
      const bytes = lg + Math.log10(4);
      const pretty = (l) => (l < 9 ? Math.round(Math.pow(10, l)).toLocaleString() : `10^${l.toFixed(1)}`);
      out.innerHTML = `Q-table entries: <b>${pretty(lg)}</b> &nbsp;·&nbsp; memory: <b>${bytes < 3 ? Math.round(Math.pow(10, bytes)) + " B" : bytes < 6 ? (Math.pow(10, bytes - 3)).toFixed(1) + " kB" : bytes < 9 ? (Math.pow(10, bytes - 6)).toFixed(1) + " MB" : bytes < 12 ? (Math.pow(10, bytes - 9)).toFixed(1) + " GB" : "10^" + (bytes - 9).toFixed(0) + " GB"}</b>`;
      bar.style.width = U.clamp(lg / 80, 0.01, 1) * 100 + "%";
    }
    draw();
    return {};
  });

  // ================================================================ DQN architecture (illustrative values)
  Deck.widget("dqnarch", (root) => {
    const W = 700, H = 360, svg = U.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const names = ["x", "ẋ", "θ", "θ̇"], s = [0, 0, 0.05, 0];
    const sls = names.map((n, i) => U.slider({ label: n, min: [-2.4, -2, -0.2, -2][i], max: [2.4, 2, 0.2, 2][i], step: 0.01, value: s[i], width: 110, fmt: (v) => v.toFixed(2), oninput: (v) => { s[i] = v; draw(); } }));
    root.append(el("div", { class: "row" }, svg, el("div", { class: "stack" }, el("b", {}, "Input state"), ...sls, el("p", { class: "hint" }, "Q-values illustrative (a sensible controller), not a trained net."))));
    const layers = [4, 8, 8, 2], xs = [70, 250, 430, 610];
    const rng = U.rng(12), Wts = [0, 1, 2].map((l) => Array.from({ length: layers[l + 1] }, () => Array.from({ length: layers[l] }, () => rng.normal() * 0.8)));
    function draw() {
      svg.innerHTML = "";
      const acts = [s.map((v, i) => v / [2.4, 2, 0.2, 2][i])];
      for (let l = 0; l < 2; l++) acts.push(Wts[l].map((row) => Math.tanh(row.reduce((a, w, j) => a + w * acts[l][j], 0))));
      const pref = 30 * (s[2] + 0.4 * s[3] + 0.02 * s[0] + 0.1 * s[1]);
      const base = 60 - 150 * Math.abs(s[2]) - 8 * Math.abs(s[0]);
      const q = [base - pref, base + pref];
      acts.push(q);
      const pos = (l, i) => [xs[l], H / 2 + (i - (layers[l] - 1) / 2) * (l === 3 ? 110 : 38)];
      for (let l = 0; l < 3; l++) for (let i = 0; i < layers[l]; i++) for (let j = 0; j < layers[l + 1]; j++) {
        const [x1, y1] = pos(l, i), [x2, y2] = pos(l + 1, j), w = l < 2 ? Wts[l][j][i] : 0.5;
        svg.append(U.svg("line", { x1, y1, x2, y2, stroke: w > 0 ? C.blue : C.red, "stroke-opacity": Math.min(0.5, Math.abs(w) * 0.3), "stroke-width": 1.2 }));
      }
      const best = q[1] > q[0] ? 1 : 0;
      layers.forEach((n, l) => { for (let i = 0; i < n; i++) {
        const [x, y] = pos(l, i), a = acts[l][i];
        if (l === 3) {
          svg.append(U.svg("rect", { x: x - 52, y: y - 28, width: 104, height: 56, rx: 10, fill: i === best ? C.green : "#fff", stroke: C.ink, "stroke-width": 2 }));
          svg.append(U.svg("text", { x, y: y - 6, "text-anchor": "middle", "font-size": 15, fill: i === best ? "#fff" : C.ink }, `Q(s, ${i ? "→" : "←"})`));
          svg.append(U.svg("text", { x, y: y + 16, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: i === best ? "#fff" : C.ink }, a.toFixed(1)));
        } else {
          svg.append(U.svg("circle", { cx: x, cy: y, r: l === 0 ? 20 : 13, fill: U.divColor(a, 1), stroke: C.ink, "stroke-width": 1.5 }));
          if (l === 0) svg.append(U.svg("text", { x: x - 30, y: y + 5, "text-anchor": "end", "font-size": 17 }, names[i]));
        }
      } });
      [["state s", 70], ["hidden", 250], ["hidden", 430], ["one Q per action", 610]].forEach(([t, x]) => svg.append(U.svg("text", { x, y: 20, "text-anchor": "middle", "font-size": 14, fill: C.muted }, t)));
      svg.append(U.svg("text", { x: 610, y: H - 12, "text-anchor": "middle", "font-size": 15, fill: C.green, "font-weight": 700 }, `act: argmax → ${best ? "push right" : "push left"}`));
    }
    draw();
    return {};
  });

  // ================================================================ replay buffer + target network animation
  Deck.widget("replay", (root) => {
    const W = 1100, H = 380, cv = U.canvas(W, H, root), ctx = cv.ctx;
    const cap = 48, buf = Array(cap).fill(null);
    let ptr = 0, envSteps = 0, updates = 0, syncs = 0, batch = [], flash = 0, acc = 0, epi = 0, tInEp = 0;
    const C_sl = U.slider({ label: "target_update_interval (updates)", min: 5, max: 60, step: 1, value: 20 });
    const ctr = el("div", { class: "controls" }, C_sl);
    root.prepend(ctr);
    const rng = U.rng(8);
    const hue = (e) => `hsl(${(e * 47) % 360} 65% 55%)`;
    function box(x, y, w, h, fill, title, sub) {
      ctx.fillStyle = fill; ctx.beginPath(); ctx.roundRect(x, y, w, h, 12); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = "700 18px " + U.css("--font"); ctx.textAlign = "center"; ctx.fillText(title, x + w / 2, y + h / 2 - (sub ? 6 : -6));
      if (sub) { ctx.font = "14px " + U.css("--font"); ctx.fillText(sub, x + w / 2, y + h / 2 + 16); }
    }
    function arrow(x1, y1, x2, y2, col = C.ink) { ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke(); const a = Math.atan2(y2 - y1, x2 - x1); ctx.beginPath(); ctx.moveTo(x2, y2); ctx.lineTo(x2 - 12 * Math.cos(a - 0.4), y2 - 12 * Math.sin(a - 0.4)); ctx.lineTo(x2 - 12 * Math.cos(a + 0.4), y2 - 12 * Math.sin(a + 0.4)); ctx.fill(); }
    const slot = (i) => [330 + (i % 12) * 32, 90 + Math.floor(i / 12) * 32];
    const loop = U.loop((dt) => {
      acc += dt;
      if (acc > 0.09) {
        acc = 0; envSteps++; tInEp++;
        buf[ptr] = { ep: epi, t: tInEp }; ptr = (ptr + 1) % cap;
        if (rng() < 0.05) { epi++; tInEp = 0; }
        if (envSteps % 4 === 0 && envSteps > 16) {
          const filled = buf.map((b, i) => (b ? i : -1)).filter((i) => i >= 0);
          batch = []; while (batch.length < 8) { const k = filled[rng.int(filled.length)]; if (!batch.includes(k)) batch.push(k); }
          updates++;
          if (updates % C_sl.get() === 0) { syncs++; flash = 1; }
        }
      }
      flash = Math.max(0, flash - dt * 1.5);
      ctx.clearRect(0, 0, W, H);
      box(20, 120, 200, 90, C.ink, "Environment", "acting with ε-greedy");
      arrow(220, 165, 322, 165); ctx.fillStyle = C.muted; ctx.font = "13px " + U.css("--font"); ctx.textAlign = "center"; ctx.fillText("(s, a, r, s′, done)", 270, 150);
      // buffer
      ctx.fillStyle = C.ink; ctx.font = "600 16px " + U.css("--font"); ctx.textAlign = "left"; ctx.fillText(`Replay buffer (capacity ${cap}, FIFO)`, 330, 76);
      for (let i = 0; i < cap; i++) {
        const [x, y] = slot(i), b = buf[i];
        ctx.fillStyle = b ? hue(b.ep) : "#eef0f3"; ctx.fillRect(x, y, 28, 28);
        if (batch.includes(i)) { ctx.strokeStyle = C.ink; ctx.lineWidth = 3; ctx.strokeRect(x - 1, y - 1, 30, 30); }
        if (i === ptr) { ctx.fillStyle = C.red; ctx.beginPath(); ctx.moveTo(x + 14, y - 3); ctx.lineTo(x + 8, y - 12); ctx.lineTo(x + 20, y - 12); ctx.fill(); }
      }
      ctx.fillStyle = C.muted; ctx.font = "13px " + U.css("--font"); ctx.fillText("colour = episode · ▼ = next write (overwrites oldest) · outlined = random minibatch", 330, 228);
      // networks
      arrow(720, 150, 800, 110); ctx.fillStyle = C.muted; ctx.fillText("minibatch", 735, 118);
      box(800, 60, 260, 80, C.blue, "Online Q-network  θ", `gradient updates: ${updates}`);
      box(800, 250, 260, 80, flash > 0 ? `rgb(${31 + 200 * flash},${95 + 60 * flash},${174 - 100 * flash})` : "#5a6b85", "Target network  θ⁻", `copied from θ ${syncs}×`);
      arrow(930, 245, 930, 146, C.green);
      ctx.fillStyle = C.green; ctx.textAlign = "right"; ctx.font = "600 14px " + U.css("--font"); ctx.fillText("target y = r + γ max Q(s′, a′; θ⁻)", 922, 200);
      ctx.fillStyle = C.orange; ctx.fillText(flash > 0 ? "θ⁻ ← θ" : "", 1000, 350);
      ctx.fillStyle = C.ink; ctx.font = "15px " + U.css("--mono"); ctx.textAlign = "left"; ctx.fillText(`env steps: ${envSteps}   (train_freq = 4, learning_starts = 16)`, 20, 300);
    });
    return { start: loop.start, stop: loop.stop };
  });

  // ================================================================ CartPole: human / random / random-search trained linear policy
  Deck.widget("cartpole", (root) => {
    const W = 640, H = 300, cv = U.canvas(W, H), ctx = cv.ctx;
    const env = new RL.CartPole({ seed: 3 });
    let mode = 0, act = 0, keyL = false, keyR = false, best = null, bestR = -1, iters = 0, hist = [], training = false, speed = 0.5, acc = 0;
    const pp = el("div");
    const plot = new U.Plot(pp, 380, 230, { xlim: [0, 60], ylim: [0, 520], xlabel: "search iteration", ylabel: "episode length (= return)", yticks: 5 });
    const tog = U.toggle(["You (A / D keys)", "Random", "Linear policy"], (i) => { mode = i; env.reset(); }, 0);
    const setTrainLabel = () => (trainBtn.innerHTML = training ? "❚❚ Pause search" : iters ? "▶ Continue search" : "▶ Train by random search");
    const trainBtn = U.button("", () => { training = !training; if (training) { mode = 2; tog.set(2); env.reset(); } setTrainLabel(); }, "warn");
    const resetBtn = U.button("↺ Reset policy", () => { training = false; best = null; bestR = -1; iters = 0; hist = []; env.reset(); setTrainLabel(); });
    setTrainLabel();
    const spd = U.slider({ label: "speed", min: 0.25, max: 2, step: 0.25, value: 0.5, fmt: (v) => v + "×", oninput: (v) => (speed = v) });
    const info = el("div", { class: "readout", style: { fontSize: "15px" } });
    const wInfo = el("div", { class: "readout", style: { fontSize: "15px" } });
    root.append(el("div", { class: "controls" }, tog, spd, U.button("Reset episode", () => env.reset())),
      el("div", { class: "row" }, el("div", { class: "stack" }, cv, info), el("div", { class: "stack" }, el("div", { class: "controls" }, trainBtn, resetBtn), pp, wInfo)));
    const rng = U.rng(4);
    // linear policy: push right iff w · s > 0
    const score = (w, s) => w[0] * s[0] + w[1] * s[1] + w[2] * s[2] + w[3] * s[3];
    const policy = (w, s) => (score(w, s) > 0 ? 1 : 0);
    function evalW(w, n = 3) { const e = new RL.CartPole({ seed: rng.int(1e6) }); let tot = 0; for (let k = 0; k < n; k++) { let s = e.reset(), done = false; while (!done) { const [s2, , te, tr] = e.step(policy(w, s)); s = s2; done = te || tr; } tot += e.t; } return tot / n; }
    function trainStep() {
      // hill climbing: perturb the best weights, keep the candidate if it does at least as well
      const cand = best ? best.map((x) => x + 0.5 * rng.normal()) : [0, 1, 2, 3].map(() => rng.normal());
      const R = evalW(cand);
      if (R >= bestR) { best = cand; bestR = R; }
      iters++; hist.push([iters, R, bestR]);
    }
    const onKey = (e, down) => { if (!root.isConnected || !root.closest(".slide.active")) return; const k = e.key.toLowerCase(); if (k === "a") keyL = down; if (k === "d") keyR = down; };
    document.addEventListener("keydown", (e) => onKey(e, true)); document.addEventListener("keyup", (e) => onKey(e, false));
    const f2 = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
    const loop = U.loop((dt) => {
      if (training) { for (let k = 0; k < 2; k++) trainStep(); if (bestR >= 500 && iters > 5) { training = false; setTrainLabel(); } }
      acc += dt * speed * 50;
      while (acc >= 1) {
        acc -= 1;
        if (env.done) env.reset();
        act = mode === 0 ? (keyR ? 1 : keyL ? 0 : env.t % 2) : mode === 1 ? rng.int(2) : best ? policy(best, env.s) : rng.int(2);
        env.step(act);
      }
      env.draw(ctx, W, H, { action: act, label: ["human", "random", best ? "linear policy" : "linear policy (untrained → random)"][mode] });
      plot.o.xlim = [0, Math.max(60, iters)]; plot.axes(); plot.hline(500, C.green, [5, 4], "max (500)");
      plot.points(hist.map((h) => [h[0], h[1]]), "rgba(31,95,174,.45)", 3); plot.line(hist.map((h) => [h[0], h[2]]), C.orange, 2.5);
      const s = env.s;
      info.innerHTML = `s = (x, ẋ, θ, θ̇) = (${s.map((v) => v.toFixed(2)).join(", ")})<br>${mode === 0 ? "hold <b>A</b> / <b>D</b> to push left / right (no key: alternate)" : mode === 2 && best ? `w·s = ${score(best, s).toFixed(2)} → push <b>${act ? "right" : "left"}</b>` : "&nbsp;"}`;
      wInfo.innerHTML = best
        ? `push right ⇔ ${f2(best[0])}·x ${f2(best[1])}·ẋ ${f2(best[2])}·θ ${f2(best[3])}·θ̇ &gt; 0<br>iterations: ${iters} · best avg length: ${bestR.toFixed(0)}<br><span style="color:${C.blue}">dots</span> = candidates, <span style="color:${C.orange}">line</span> = best so far`
        : "no policy yet — press Train (4 random weights to start)";
    });
    return { start: loop.start, stop() { loop.stop(); keyL = keyR = false; } };
  });

  // ================================================================ epsilon schedule (SB3 DQN)
  Deck.widget("epsched", (root) => {
    const T = 100000;
    const plot = new U.Plot(root, 760, 260, { xlim: [0, T], ylim: [0, 1.05], xlabel: "timestep", ylabel: "ε (prob. random action)", xfmt: (v) => v / 1000 + "k", yticks: 5 });
    const f = U.slider({ label: "exploration_fraction", min: 0.01, max: 1, step: 0.01, value: 0.1, fmt: (v) => v.toFixed(2), oninput: draw });
    const e1 = U.slider({ label: "exploration_final_eps", min: 0, max: 0.3, step: 0.01, value: 0.05, fmt: (v) => v.toFixed(2), oninput: draw });
    const ls = U.slider({ label: "learning_starts", min: 0, max: 20000, step: 500, value: 1000, oninput: draw });
    root.prepend(el("div", { class: "controls" }, f, e1, ls));
    function draw() {
      plot.axes();
      plot.clip((c) => { c.fillStyle = "rgba(232,161,0,.18)"; c.fillRect(plot.X(0), plot.Y(1.05), plot.X(ls.get()) - plot.X(0), plot.Y(0) - plot.Y(1.05)); });
      const pts = []; for (let t = 0; t <= T; t += 500) { const p = Math.min(1, t / (f.get() * T)); pts.push([t, 1 + p * (e1.get() - 1)]); }
      plot.line(pts, C.blue, 3);
      plot.text(ls.get() + 1500, 0.95, "no gradient updates yet (buffer filling)", C.gold);
    }
    draw();
    return {};
  });

  // ================================================================ REINFORCE on a 3-armed bandit
  Deck.widget("pg", (root) => {
    const means = [0.2, 0.5, 0.8], K = 3;
    let h, t, baseline, hist, run = false, useB = true;
    const pp = el("div");
    const plot = new U.Plot(pp, 520, 280, { xlim: [0, 600], ylim: [0, 1], xlabel: "step", ylabel: "π(a)", yticks: 5 });
    const bars = el("div", { style: { display: "flex", gap: "18px", alignItems: "flex-end", height: "200px", padding: "0 10px" } });
    const barEls = means.map((m, i) => { const b = el("div", { style: { width: "70px", background: [C.red, C.blue, C.green][i], borderRadius: "6px 6px 0 0", transition: "height .1s" } }); const lab = el("div", { class: "tiny", style: { textAlign: "center" } }); bars.append(el("div", { style: { display: "flex", flexDirection: "column", justifyContent: "flex-end", height: "100%", alignItems: "center", gap: "4px" } }, lab, b, el("div", { class: "tiny" }, `arm ${i + 1}: mean ${m}`))); return [b, lab]; });
    const aSl = U.slider({ label: "α", min: 0.01, max: 0.5, step: 0.01, value: 0.1, fmt: (v) => v.toFixed(2) });
    const runBtn = U.button("▶ Run", () => { run = !run; runBtn.innerHTML = run ? "❚❚" : "▶ Run"; }, "primary");
    root.append(el("div", { class: "controls" }, runBtn, U.button("Reset", reset), aSl, U.toggle(["with baseline", "no baseline"], (i) => { useB = !i; reset(); })),
      el("div", { class: "row" }, el("div", { class: "stack" }, bars, el("p", { class: "small", html: "$h_a \\leftarrow h_a + \\alpha\\,(R - b)\\,\\big(\\mathbb{1}[a{=}A] - \\pi(a)\\big)$" })), el("div", {}, pp, el("div", { class: "legend" }, ...[C.red, C.blue, C.green].map((c, i) => el("span", { style: "--c:" + c }, `π(arm ${i + 1})`))))));
    renderMathInElement(root, { delimiters: [{ left: "$", right: "$", display: false }] });
    const rng = U.rng(6);
    const pis = () => { const m = Math.max(...h), e = h.map((x) => Math.exp(x - m)), z = e.reduce((a, b) => a + b); return e.map((x) => x / z); };
    function reset() { h = [0, 0, 0]; t = 0; baseline = 0; hist = [[0, ...pis()]]; draw(); }
    function step() {
      const p = pis(), A = rng.categorical(p), R = means[A] + 0.5 * rng.normal();
      const b = useB ? baseline : 0; t++; baseline += (R - baseline) / t;
      for (let a = 0; a < K; a++) h[a] += aSl.get() * (R - b) * ((a === A ? 1 : 0) - p[a]);
      hist.push([t, ...pis()]);
    }
    function draw() {
      const p = pis();
      barEls.forEach(([b, lab], i) => { b.style.height = p[i] * 160 + "px"; lab.textContent = p[i].toFixed(2); });
      plot.o.xlim = [0, Math.max(600, t)]; plot.axes();
      [C.red, C.blue, C.green].forEach((c, i) => plot.line(hist.map((x) => [x[0], x[i + 1]]), c, 2.2));
    }
    let acc = 0;
    const loop = U.loop((dt) => { if (!run) return; acc += dt; if (acc < 0.02) return; acc = 0; for (let k = 0; k < 3; k++) step(); draw(); });
    reset();
    return { start: loop.start, stop() { loop.stop(); run = false; runBtn.innerHTML = "▶ Run"; } };
  });

  // ================================================================ PPO clipped objective
  Deck.widget("ppoclip", (root) => {
    const mkPlot = (title) => { const d = el("div"); const p = new U.Plot(d, 470, 280, { xlim: [0, 2], ylim: [-2, 2], xlabel: "probability ratio r = π_new(a|s) / π_old(a|s)", ylabel: "objective", yticks: 4, title }); return [d, p]; };
    const [d1, p1] = mkPlot("advantage A > 0 (good action)"), [d2, p2] = mkPlot("advantage A < 0 (bad action)");
    const eps = U.slider({ label: "clip ε", min: 0.05, max: 0.5, step: 0.01, value: 0.2, fmt: (v) => v.toFixed(2), oninput: draw });
    root.append(el("div", { class: "controls" }, eps, el("span", { class: "legend" }, el("span", { style: "--c:#9aa1ad" }, "unclipped r·A"), el("span", { style: "--c:" + C.orange }, "PPO: min(r·A, clip(r)·A)"))), el("div", { class: "row" }, d1, d2));
    function draw() {
      const e = eps.get();
      for (const [p, A] of [[p1, 1], [p2, -1]]) {
        p.axes(); p.hline(0, C.muted, [2, 3]);
        p.clip((c) => { c.fillStyle = "rgba(43,138,62,.1)"; c.fillRect(p.X(1 - e), p.Y(2), p.X(1 + e) - p.X(1 - e), p.Y(-2) - p.Y(2)); });
        const xs = []; for (let r = 0; r <= 2.0001; r += 0.01) xs.push(r);
        p.line(xs.map((r) => [r, r * A]), "#9aa1ad", 2, [6, 4]);
        p.line(xs.map((r) => [r, Math.min(r * A, U.clamp(r, 1 - e, 1 + e) * A)]), C.orange, 3.5);
        p.vline(1, C.green, [3, 3]);
        if (A > 0) p.text(1.55, 1.75, "flat: no gain beyond 1+ε", C.muted, "center");
        else p.text(0.45, -1.2, "flat: no gain below 1−ε", C.muted, "center");
      }
    }
    draw();
    return {};
  });

  // ================================================================ Reward hacking gridworld
  Deck.widget("hack", (root) => {
    // 5x8 grid, start (2,0), finish (2,7) +10 terminal, bonus tile (1,1)
    const R = 5, Cc = 8, start = 16, finish = 23, bonus = 9;
    let design = 0, Q, trained = 0, roll = null, rollIdx = 0, rollR = 0;
    const z = 62, cv = U.canvas(Cc * z, R * z), ctx = cv.ctx;
    const info = el("div", { class: "readout", style: { fontSize: "16px" } });
    const tog = U.toggle(["+1 every time on ★", "+1 only the first time on ★"], (i) => { design = i; reset(); });
    const trainBtn = U.button("Train (Q-learning, 3000 episodes)", () => { train(3000); rollout(); }, "primary");
    root.append(el("div", { class: "controls" }, tog, trainBtn, U.button("Watch greedy policy", rollout)), el("div", { class: "row" }, cv, info));
    const nS = R * Cc * 2; // (cell, collected flag)
    const rng = U.rng(10);
    function reset() { Q = Array.from({ length: nS }, () => [0, 0, 0, 0]); trained = 0; roll = null; draw(); }
    function envStep(s, a) {
      const cell = s % (R * Cc), got = s >= R * Cc ? 1 : 0;
      const r0 = Math.floor(cell / Cc), c0 = cell % Cc, [dr, dc] = RL.ACTIONS[a];
      const nc = U.clamp(r0 + dr, 0, R - 1) * Cc + U.clamp(c0 + dc, 0, Cc - 1);
      let rew = -0.1, ngot = got, done = false;
      if (nc === bonus && (design === 0 || !got)) { rew += 1; ngot = 1; }
      if (nc === finish) { rew += 10; done = true; }
      return [nc + (design === 1 && ngot ? R * Cc : 0), rew, done];
    }
    function train(n) {
      for (let e = 0; e < n; e++) {
        let s = start;
        for (let t = 0; t < 100; t++) {
          const a = rng() < 0.2 ? rng.int(4) : argmaxRand(Q[s], rng);
          const [s2, r, done] = envStep(s, a);
          Q[s][a] += 0.2 * (r + (done ? 0 : 0.99 * Math.max(...Q[s2])) - Q[s][a]);
          s = s2; if (done) break;
        }
      }
      trained += n;
    }
    function rollout() { roll = [start]; let s = start; rollR = 0; for (let t = 0; t < 100; t++) { const [s2, r, d] = envStep(s, argmax(Q[s])); rollR += r; roll.push(s2); s = s2; if (d) break; } rollIdx = 0; }
    function draw() {
      ctx.clearRect(0, 0, cv.W, cv.H);
      for (let i = 0; i < R * Cc; i++) {
        const r = Math.floor(i / Cc), c = i % Cc;
        ctx.fillStyle = i === finish ? "#c9ebd0" : i === bonus ? "#fff1c2" : "#fff"; ctx.fillRect(c * z, r * z, z, z);
        ctx.strokeStyle = "#d0d3d9"; ctx.strokeRect(c * z, r * z, z, z);
      }
      ctx.font = "30px " + U.css("--font"); ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText("🏁", (finish % Cc) * z + z / 2, Math.floor(finish / Cc) * z + z / 2); ctx.fillStyle = C.gold; ctx.fillText("★", (bonus % Cc) * z + z / 2, Math.floor(bonus / Cc) * z + z / 2);
      ctx.fillStyle = C.ink; ctx.font = "600 14px " + U.css("--font"); ctx.fillText("start", (start % Cc) * z + z / 2, Math.floor(start / Cc) * z + 12);
      if (roll) {
        const k = Math.min(rollIdx, roll.length - 1), cells = roll.slice(0, k + 1).map((s) => s % (R * Cc));
        ctx.strokeStyle = C.orange; ctx.globalAlpha = 0.5; ctx.lineWidth = 4; ctx.beginPath(); cells.forEach((s, i) => { const x = (s % Cc) * z + z / 2, y = Math.floor(s / Cc) * z + z / 2; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.globalAlpha = 1;
        const s = cells[cells.length - 1]; ctx.beginPath(); ctx.arc((s % Cc) * z + z / 2, Math.floor(s / Cc) * z + z / 2, 14, 0, 7); ctx.fillStyle = C.orange; ctx.fill();
      }
      const fin = roll && roll[roll.length - 1] % (R * Cc) === finish;
      info.innerHTML = `Reward: −0.1 per step, +10 at 🏁 (ends episode), bonus at ★<br>Episodes cap at 100 steps · γ = 0.99<br><br>trained episodes: ${trained}<br>${roll ? `greedy rollout: return <b>${rollR.toFixed(1)}</b> in ${roll.length - 1} steps<br>reached the finish? ${fin ? "<span class='good'>yes</span>" : "<span class='bad'>no — it farms the bonus</span>"}` : ""}`;
    }
    let acc = 0;
    const loop = U.loop((dt) => { if (!roll) return; acc += dt; if (acc > 0.12 && rollIdx < roll.length) { acc = 0; rollIdx++; draw(); } });
    reset();
    return { start: loop.start, stop: loop.stop };
  });
})();

/* Lecture 1 widgets — Intro to RL & MDPs */
(function () {
  "use strict";
  const { el, C } = U;

  // ================================================================ paradigms: SL vs UL vs RL
  Deck.widget("paradigms", (root) => {
    const W = 560, H = 360, rng = U.rng(3);
    const desc = el("div", { class: "box", style: { flex: 1, minHeight: "300px" } });
    const cv = U.canvas(W, H);
    const tog = U.toggle(["Supervised", "Unsupervised", "Reinforcement"], (i) => setMode(i), 0);
    root.append(el("div", { class: "controls" }, tog, U.button("↺ Reset", () => setMode(mode))), el("div", { class: "row" }, el("div", { class: "box", style: { padding: "6px" } }, cv), desc));
    const ctx = cv.ctx;
    // data: three blobs
    const centers = [[160, 120, 0], [380, 110, 1], [290, 270, 1]];
    const pts = [];
    for (let i = 0; i < 75; i++) { const [cx, cy, y] = centers[i % 3]; pts.push({ x: cx + rng.normal() * 38, y: cy + rng.normal() * 34, lab: y, blob: i % 3, k: 0 }); }
    let mode = 0, w = [0.3, -0.8, 0.1], cents = [], iter = 0, acc = 0, acc2 = 0;
    // RL mini grid
    const G = { rows: 4, cols: 5, goal: 4, pit: 9, start: 15 };   // small grid → learns in ~15 s
    const NC = G.cols, NS = G.rows * G.cols;
    let Q, s, ep, stepsThis, hist, pops, epRet, retHist;
    const score = el("div", { class: "readout", style: { fontSize: "16px", margin: "6px 0" } });
    const plotBox = el("div");
    const rlPlot = new U.Plot(plotBox, 500, 190, { xlim: [0, 100], ylim: [-50, 10], xlabel: "episode", ylabel: "score", yticks: 4, pad: [10, 12, 36, 52] });
    const texts = [
      `<h3>Supervised learning</h3><p>Data: pairs $(x, y)$ — every example comes with the <b>correct answer</b>.</p><p>Goal: learn $f: x \\mapsto y$ (classification, regression).</p><p class="muted">The line is fitted by gradient descent on labelled points.</p>`,
      `<h3>Unsupervised learning</h3><p>Data: only $x$ — <b>no labels</b>.</p><p>Goal: discover hidden structure (clusters, low-dim manifolds, anomalies).</p><p class="muted">Here: k-means finds 3 groups on its own.</p>`,
      `<h3>Reinforcement learning</h3><p class="small" style="margin:2px 0">No dataset: the agent <b>generates its own data</b> by acting. Feedback is a <b>scalar reward</b>: −1 per step, +10 at the goal, −10 in the pit.</p>`,
    ];
    function setMode(i) {
      mode = i; desc.innerHTML = texts[i]; if (window.renderMathInElement) renderMathInElement(desc, { delimiters: [{ left: "$", right: "$", display: false }] });
      if (i === 0) w = [0.3, -0.8, 0.1];
      if (i === 1) { cents = [[100, 300], [120, 60], [500, 330]]; iter = 0; }
      if (i === 2) {
        Q = Array.from({ length: NS }, () => [0, 0, 0, 0]); s = G.start; ep = 0; stepsThis = 0; hist = []; pops = []; epRet = 0; retHist = [];
        desc.append(score, plotBox, el("div", { class: "legend" }, el("span", { style: "--c:rgba(31,95,174,.45)" }, "score of each episode"), el("span", { style: "--c:" + C.orange }, "average of last 10"), el("span", { style: "--c:" + C.green }, "best possible (+4)")));
      }
    }
    setMode(0);
    const nx = (p) => [(p.x - 280) / 150, (p.y - 180) / 150];
    function drawPts(colorFn) { for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.fillStyle = colorFn(p); ctx.fill(); ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke(); } }
    const loop = U.loop((dt) => {
      acc += dt;
      ctx.clearRect(0, 0, W, H);
      if (mode === 0) {
        for (let k = 0; k < 3; k++) for (const p of pts) { const [a, b] = nx(p), z = w[0] * a + w[1] * b + w[2], pr = 1 / (1 + Math.exp(-z)), g = pr - p.lab; w[0] -= 0.02 * g * a; w[1] -= 0.02 * g * b; w[2] -= 0.02 * g; }
        drawPts((p) => (p.lab ? C.blue : C.orange));
        // boundary w0 a + w1 b + w2 = 0
        ctx.strokeStyle = C.ink; ctx.lineWidth = 3; ctx.setLineDash([8, 6]); ctx.beginPath();
        const ys = [-2, 2].map((a) => [(a * 150 + 280), ((-(w[0] * a + w[2]) / w[1]) * 150 + 180)]);
        ctx.moveTo(...ys[0]); ctx.lineTo(...ys[1]); ctx.stroke(); ctx.setLineDash([]);
        ctx.font = "15px " + U.css("--font"); ctx.fillStyle = C.orange; ctx.fillText("label: normal", 20, 24); ctx.fillStyle = C.blue; ctx.fillText("label: abnormal", 20, 44);
      } else if (mode === 1) {
        if (acc > 0.8 && iter < 8) {
          acc = 0; iter++;
          for (const p of pts) { let b = 0, bd = 1e9; cents.forEach((c, k) => { const d = (p.x - c[0]) ** 2 + (p.y - c[1]) ** 2; if (d < bd) { bd = d; b = k; } }); p.k = b; }
          cents = cents.map((c, k) => { const m = pts.filter((p) => p.k === k); return m.length ? [m.reduce((a, p) => a + p.x, 0) / m.length, m.reduce((a, p) => a + p.y, 0) / m.length] : c; });
        }
        const cols = [C.green, C.purple, C.gold];
        drawPts((p) => (iter ? cols[p.k] : "#9aa1ad"));
        cents.forEach((c, k) => { ctx.strokeStyle = cols[k]; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(c[0] - 10, c[1] - 10); ctx.lineTo(c[0] + 10, c[1] + 10); ctx.moveTo(c[0] + 10, c[1] - 10); ctx.lineTo(c[0] - 10, c[1] + 10); ctx.stroke(); });
        ctx.fillStyle = C.ink; ctx.font = "15px " + U.css("--font"); ctx.fillText(`k-means iteration ${iter}`, 20, 24);
        if (iter >= 8 && acc > 2.5) setMode(1);
      } else {
        // Q-learning on a 4x5 grid (a few steps per frame so the improvement is visible)
        const z = 70, ox = (W - NC * z) / 2, oy = 24;
        acc2 += dt;
        while (acc2 > 1 / 45) {
          acc2 -= 1 / 45;
          const a = rng() < 0.1 ? rng.int(4) : Q[s].indexOf(Math.max(...Q[s]));
          const [r, c] = [Math.floor(s / NC), s % NC], [dr, dc] = RL.ACTIONS[a];
          const nr = U.clamp(r + dr, 0, G.rows - 1), nc = U.clamp(c + dc, 0, NC - 1), t = nr * NC + nc;
          const rew = t === G.goal ? 10 : t === G.pit ? -10 : -1, done = t === G.goal || t === G.pit;
          Q[s][a] += 0.5 * (rew + (done ? 0 : 0.95 * Math.max(...Q[t])) - Q[s][a]);
          stepsThis++; epRet += rew;
          if (done || stepsThis > 40) {
            pops.push({ s: t, txt: rew > 0 ? "+10" : rew < -1 ? "−10" : "", life: 1 });
            hist.push(stepsThis); retHist.push(epRet); ep++; s = G.start; stepsThis = 0; epRet = 0;
            break;
          }
          s = t;
        }
        for (let i = 0; i < NS; i++) {
          const r = Math.floor(i / NC), c = i % NC, x = ox + c * z, y = oy + r * z;
          ctx.fillStyle = i === G.goal ? "#c9ebd0" : i === G.pit ? "#f6c9c9" : U.divColor(Math.max(...Q[i]), 10);
          ctx.fillRect(x, y, z, z); ctx.strokeStyle = "#c8ccd4"; ctx.strokeRect(x, y, z, z);
          if (i === G.goal || i === G.pit) { ctx.fillStyle = C.ink; ctx.font = "600 16px " + U.css("--font"); ctx.textAlign = "center"; ctx.fillText(i === G.goal ? "+10" : "−10", x + z / 2, y + z / 2 + 5); ctx.textAlign = "left"; }
        }
        const [r, c] = [Math.floor(s / NC), s % NC];
        ctx.beginPath(); ctx.arc(ox + c * z + z / 2, oy + r * z + z / 2, 14, 0, 7); ctx.fillStyle = C.orange; ctx.fill();
        pops = pops.filter((p) => (p.life -= dt) > 0);
        for (const p of pops) { const [pr, pc] = [Math.floor(p.s / NC), p.s % NC]; ctx.globalAlpha = p.life; ctx.fillStyle = p.txt[0] === "+" ? C.green : C.red; ctx.font = "bold 22px " + U.css("--font"); ctx.fillText(p.txt, ox + pc * z + 10, oy + pr * z - 8 + p.life * 10); ctx.globalAlpha = 1; }
        ctx.fillStyle = C.ink; ctx.font = "600 17px " + U.css("--font");
        ctx.fillText(`episode ${ep + 1}   ·   score so far: ${epRet > 0 ? "+" : ""}${epRet}`, ox, H - 12);
        const last = retHist.slice(-10), avg = last.length ? last.reduce((a, b) => a + b) / last.length : 0, prevRet = retHist[retHist.length - 1];
        score.innerHTML = `episodes played: <b>${ep}</b> · last score: <b>${prevRet == null ? "–" : (prevRet > 0 ? "+" : "") + prevRet}</b><br>average of last 10: <b style="color:${avg > 0 ? C.green : C.red}">${last.length ? avg.toFixed(1) : "–"}</b> &nbsp;(best possible: +4 = six −1 steps, then +10)`;
        const n = retHist.length, xmax = Math.max(30, n);
        rlPlot.o.xlim = [0, xmax]; rlPlot.axes(); rlPlot.hline(4, C.green, [5, 4]);
        rlPlot.points(retHist.map((v, i) => [i + 1, v]), "rgba(31,95,174,.45)", 3);
        rlPlot.line(U.smooth(retHist, 10).map((v, i) => [i + 1, v]), C.orange, 3);
        if (ep >= 120) setMode(2);
      }
    });
    return { start: loop.start, stop: loop.stop };
  });


  // ================================================================ RL in the real world: media cards
  Deck.widget("realworld", (root) => {
    const vids = [...root.querySelectorAll("video")];
    // --- Atari-style Breakout played by a (scripted) agent
    const bc = root.querySelector("[data-anim=breakout]"), BW = 360, BH = 150;
    const bcv = U.canvas(BW, BH, bc), b = bcv.ctx;
    let ball, pad, bricks, score, lives;
    const resetBO = () => { ball = { x: 180, y: 100, vx: 150, vy: -150 }; pad = 180; score = 0; bricks = []; for (let r = 0; r < 5; r++) for (let c = 0; c < 12; c++) bricks.push({ x: 6 + c * 29, y: 22 + r * 9, r, alive: true }); };
    resetBO();
    const bcol = ["#c92a2a", "#d9480f", "#e8a100", "#2b8a3e", "#1f5fae"];
    function stepBO(dt) {
      ball.x += ball.vx * dt; ball.y += ball.vy * dt;
      if (ball.x < 4 || ball.x > BW - 4) ball.vx *= -1;
      if (ball.y < 16) ball.vy = Math.abs(ball.vy);
      pad += U.clamp(ball.x + (ball.vx > 0 ? 6 : -6) - pad, -260 * dt, 260 * dt);   // the "agent": track the ball
      if (ball.y > BH - 14 && ball.vy > 0 && Math.abs(ball.x - pad) < 26) { ball.vy = -Math.abs(ball.vy); ball.vx = (ball.x - pad) * 7 + (Math.random() - 0.5) * 30; }
      if (ball.y > BH) { ball.x = 180; ball.y = 100; ball.vy = -150; }
      for (const k of bricks) if (k.alive && ball.x > k.x && ball.x < k.x + 27 && ball.y > k.y && ball.y < k.y + 7) { k.alive = false; ball.vy *= -1; score += (5 - k.r) * 10; break; }
      if (!bricks.some((k) => k.alive)) resetBO();
    }
    function drawBO() {
      b.fillStyle = "#0b0e16"; b.fillRect(0, 0, BW, BH);
      for (const k of bricks) if (k.alive) { b.fillStyle = bcol[k.r]; b.fillRect(k.x, k.y, 27, 7); }
      b.fillStyle = "#e6e9f0"; b.fillRect(pad - 24, BH - 12, 48, 5);
      b.beginPath(); b.arc(ball.x, ball.y, 3.5, 0, 7); b.fill();
      b.font = "12px " + U.css("--mono"); b.fillStyle = "#9fc3f0"; b.textAlign = "left"; b.fillText(`SCORE ${score}`, 6, 13);
      b.textAlign = "right"; b.fillText("illustration of an Atari agent", BW - 6, 13);
    }
    // --- LLM chat with human feedback
    const chat = root.querySelector("[data-anim=chat]");
    const Q = "Explain RL in one sentence.", A = "An agent learns by trial and error to choose actions that maximise its long-term reward.";
    let ct = 0;
    function drawChat() {
      const t = ct % 9, typed = A.slice(0, Math.max(0, Math.floor((t - 1.2) * 28)));
      const fb = t > 5.2;
      chat.innerHTML = `<div class="bub u">${Q}</div><div class="bub a">${typed}${typed.length < A.length && t > 1.2 ? '<span class="cur">▍</span>' : ""}</div>` +
        `<div class="fb ${fb ? "on" : ""}">human feedback: <span>👍 reward +1</span> <span class="dim">👎</span></div>`;
    }
    const loop = U.loop((dt) => { for (let k = 0; k < 3; k++) stepBO(dt / 3); drawBO(); ct += dt; drawChat(); });
    return {
      start() { loop.start(); vids.forEach((v) => { v.currentTime = 0; const p = v.play(); if (p) p.catch(() => {}); }); },
      stop() { loop.stop(); vids.forEach((v) => v.pause()); },
    };
  });

  // ================================================================ agent–environment loop
  Deck.widget("aeloop", (root) => {
    const W = 620, H = 440;
    const svg = U.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    svg.style.width = "560px"; svg.style.height = "auto";
    const logBody = el("tbody");
    const log = el("table", { class: "t", style: { fontSize: "17px", width: "100%" } }, el("thead", {}, el("tr", {}, ...["t", "state S<sub>t</sub>", "action A<sub>t</sub>", "reward R<sub>t+1</sub>", "return so far"].map((h) => el("th", { html: h })))), logBody);
    const logWrap = el("div", { style: { height: "330px", overflow: "hidden" } }, log);
    let auto = false;
    const autoBtn = U.button("▶ Auto", () => { auto = !auto; autoBtn.innerHTML = auto ? "❚❚ Pause" : "▶ Auto"; if (auto) run(); });
    root.append(el("div", { class: "row" }, el("div", {}, svg), el("div", { class: "col" }, el("div", { class: "controls" }, U.button("Step", () => stepOnce(), "primary"), autoBtn, U.button("Reset", reset)), logWrap, el("p", { class: "hint" }, "Environment: 3×4 grid, −1 per move, +10 at ★. Agent: a (not yet good) random policy."))));
    const S = (t, a, ...k) => { const e = U.svg(t, a, ...k); svg.append(e); return e; };
    // boxes
    S("rect", { x: 170, y: 16, width: 280, height: 110, rx: 16, fill: C.blue });
    S("text", { x: 310, y: 56, "text-anchor": "middle", fill: "#fff", "font-size": 26, "font-weight": 700 }, "Agent");
    const agentTxt = S("text", { x: 310, y: 94, "text-anchor": "middle", fill: "#dbe8fa", "font-size": 18 }, "policy π(a | s)");
    S("rect", { x: 120, y: 250, width: 380, height: 176, rx: 16, fill: "#eef0f3", stroke: C.ink, "stroke-width": 2 });
    S("text", { x: 140, y: 278, fill: C.ink, "font-size": 22, "font-weight": 700 }, "Environment");
    // grid inside environment
    const gx = 290, gy = 268, z = 48, cells = [];
    for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) cells.push(S("rect", { x: gx + c * z, y: gy + r * z, width: z, height: z, fill: "#fff", stroke: "#aab0bb" }));
    S("text", { x: gx + 3 * z + z / 2, y: gy + z / 2 + 8, "text-anchor": "middle", "font-size": 24, fill: C.gold }, "★");
    const dot = S("circle", { r: 14, fill: C.orange, stroke: "#fff", "stroke-width": 3 });
    // arrows
    const mk = U.svg("marker", { id: "ae-arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 8, markerHeight: 8, orient: "auto" }, U.svg("path", { d: "M0,0 L10,5 L0,10 z", fill: C.ink }));
    svg.prepend(U.svg("defs", {}, mk));
    const pA = S("path", { d: "M450,70 C580,70 580,330 505,330", fill: "none", stroke: C.ink, "stroke-width": 3, "marker-end": "url(#ae-arr)" });
    const pS = S("path", { d: "M120,320 C40,320 40,90 165,90", fill: "none", stroke: C.ink, "stroke-width": 3, "marker-end": "url(#ae-arr)" });
    const pR = S("path", { d: "M120,380 C10,380 10,50 165,50", fill: "none", stroke: C.ink, "stroke-width": 3, "marker-end": "url(#ae-arr)" });
    S("text", { x: 588, y: 200, "font-size": 20, fill: C.orange, "font-weight": 700, "text-anchor": "middle" }, "A");
    S("text", { x: 588, y: 222, "font-size": 15, fill: C.orange, "text-anchor": "middle" }, "action");
    S("text", { x: 70, y: 214, "font-size": 20, fill: C.blue, "font-weight": 700 }, "S");
    S("text", { x: 22, y: 244, "font-size": 20, fill: C.green, "font-weight": 700 }, "R");
    const pkt = S("g", { style: "display:none" });
    const pktR = U.svg("rect", { x: -34, y: -16, width: 68, height: 32, rx: 8 }), pktT = U.svg("text", { "text-anchor": "middle", "dominant-baseline": "central", fill: "#fff", "font-size": 17, "font-weight": 700 });
    pkt.append(pktR, pktT);
    let s, t, G, busy = false;
    const pos = (s) => [gx + (s % 4) * z + z / 2, gy + Math.floor(s / 4) * z + z / 2];
    const name = (s) => `(${Math.floor(s / 4)},${s % 4})`;
    function place() { const [x, y] = pos(s); dot.setAttribute("cx", x); dot.setAttribute("cy", y); }
    function reset() { s = 8; t = 0; G = 0; logBody.innerHTML = ""; place(); agentTxt.textContent = "policy π(a | s)"; }
    function fly(path, text, color, ms = 650) {
      pkt.style.display = ""; pktR.setAttribute("fill", color); pktT.textContent = text;
      const L = path.getTotalLength(), t0 = performance.now();
      return new Promise((res) => { const f = (tt) => { const k = Math.min(1, (tt - t0) / ms), p = path.getPointAtLength(k * L); pkt.setAttribute("transform", `translate(${p.x},${p.y})`); k < 1 ? requestAnimationFrame(f) : (pkt.style.display = "none", res()); }; requestAnimationFrame(f); });
    }
    const rng = U.rng(11);
    async function stepOnce() {
      if (busy) return; busy = true;
      const a = rng.int(4), arrows = ["↑", "→", "↓", "←"];
      agentTxt.textContent = `in ${name(s)} → choose ${arrows[a]}`;
      await fly(pA, arrows[a], C.orange);
      const r = Math.floor(s / 4), c = s % 4, [dr, dc] = RL.ACTIONS[a];
      const ns = U.clamp(r + dr, 0, 2) * 4 + U.clamp(c + dc, 0, 3), rew = ns === 3 ? 10 : -1;
      const prev = s; s = ns; place(); G += rew;
      await Promise.all([fly(pS, name(ns), C.blue), fly(pR, (rew > 0 ? "+" : "") + rew, C.green, 750)]);
      logBody.prepend(el("tr", {}, ...[t, name(prev), arrows[a], rew, G].map((x) => el("td", {}, String(x)))));
      while (logBody.children.length > 9) logBody.lastChild.remove();
      t++;
      if (s === 3) { logBody.prepend(el("tr", {}, el("td", { colspan: 5, style: { color: C.green, fontWeight: 700 } }, `episode over — total reward ${G}`))); s = 8; t = 0; G = 0; place(); }
      busy = false;
    }
    async function run() { while (auto && root.isConnected) { await stepOnce(); await U.sleep(200); } }
    reset();
    return { stop() { auto = false; autoBtn.innerHTML = "▶ Auto"; } };
  });

  // ================================================================ delayed reward vs gamma
  Deck.widget("delayed", (root) => {
    const plot = new U.Plot(root, 760, 330, { xlim: [0, 30], ylim: [-6, 40], xlabel: "time step k", ylabel: "discounted return so far", yticks: 6 });
    const out = el("div", { class: "readout", style: { marginTop: "6px" } });
    const sl = U.slider({ label: "discount γ", min: 0, max: 1, step: 0.01, value: 0.9, fmt: (v) => v.toFixed(2), oninput: draw });
    root.prepend(el("div", { class: "controls" }, sl, el("span", { class: "legend" }, el("span", { style: "--c:" + C.orange }, "“Spend”: +1 every step"), el("span", { style: "--c:" + C.blue }, "“Invest”: −1 ×5 steps, then +3 every step"))));
    root.append(out);
    const rA = (k) => 1, rB = (k) => (k < 5 ? -1 : 3);
    function draw() {
      const g = sl.get(), cum = (rf) => { let s = 0; const pts = []; for (let k = 0; k <= 30; k++) { s += Math.pow(g, k) * rf(k); pts.push([k, s]); } return pts; };
      const a = cum(rA), b = cum(rB);
      const ymax = Math.max(8, a[30][1], b[30][1]) * 1.1, ymin = Math.min(-6, ...b.map((p) => p[1])) * 1.1;
      plot.o.ylim = [ymin, ymax]; plot.axes(); plot.hline(0, C.muted, [2, 3]);
      plot.line(a, C.orange, 3); plot.line(b, C.blue, 3); plot.points(a, C.orange, 3); plot.points(b, C.blue, 3);
      const GA = a[30][1], GB = b[30][1], win = GA > GB ? "Spend" : "Invest";
      out.innerHTML = `G(spend) = ${GA.toFixed(2)} &nbsp;&nbsp; G(invest) = ${GB.toFixed(2)} &nbsp;&nbsp;→ better: <b style="color:${GA > GB ? C.orange : C.blue}">${win}</b>`;
    }
    draw();
    return {};
  });

  // ================================================================ observability: full vs partial
  Deck.widget("observe", (root) => {
    const maze = [
      "#########",
      "#...#...#",
      "#.#...#.#",
      "#.#.#.#.#",
      "#...#...#",
      "#.#...#.#",
      "#########",
    ];
    const R = maze.length, Cc = maze[0].length, z = 56;
    const cv = U.canvas(Cc * z, R * z), ctx = cv.ctx;
    const obsCv = U.canvas(3 * 50, 3 * 50), octx = obsCv.ctx;
    let partial = false, pos = [1, 1], alias = [];
    const tog = U.toggle(["Fully observable", "Partially observable"], (i) => { partial = !!i; alias = []; draw(); });
    const info = el("div", { class: "small" });
    root.append(el("div", { class: "controls" }, tog, U.button("Random step", () => { rand(); draw(); }), U.button("Which cells look the same?", () => { findAlias(); draw(); }, "warn")),
      el("div", { class: "row" }, cv, el("div", { class: "stack" }, el("b", {}, "What the agent receives:"), obsCv, info)));
    const free = (r, c) => maze[r] && maze[r][c] === ".";
    cv.addEventListener("click", (e) => { const [x, y] = cv.pos(e), c = Math.floor(x / z), r = Math.floor(y / z); if (free(r, c) && Math.abs(r - pos[0]) + Math.abs(c - pos[1]) === 1) { pos = [r, c]; alias = []; draw(); } });
    const rng = U.rng(5);
    function rand() { const opts = RL.ACTIONS.map(([dr, dc]) => [pos[0] + dr, pos[1] + dc]).filter(([r, c]) => free(r, c)); pos = rng.choice(opts); alias = []; }
    const patch = (r, c) => { let s = ""; for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) s += maze[r + dr]?.[c + dc] || "#"; return s; };
    function findAlias() { const p = patch(...pos); alias = []; for (let r = 1; r < R - 1; r++) for (let c = 1; c < Cc - 1; c++) if (free(r, c) && patch(r, c) === p) alias.push([r, c]); }
    function draw() {
      ctx.clearRect(0, 0, cv.W, cv.H);
      for (let r = 0; r < R; r++) for (let c = 0; c < Cc; c++) {
        const vis = !partial || (Math.abs(r - pos[0]) <= 1 && Math.abs(c - pos[1]) <= 1);
        ctx.fillStyle = maze[r][c] === "#" ? "#3b4252" : "#fff";
        ctx.fillRect(c * z, r * z, z, z);
        if (!vis) { ctx.fillStyle = "rgba(90,96,110,.82)"; ctx.fillRect(c * z, r * z, z, z); }
        ctx.strokeStyle = "#d5d8de"; ctx.strokeRect(c * z, r * z, z, z);
      }
      for (const [r, c] of alias) { ctx.strokeStyle = C.orange; ctx.lineWidth = 4; ctx.strokeRect(c * z + 4, r * z + 4, z - 8, z - 8); ctx.lineWidth = 1; }
      ctx.beginPath(); ctx.arc(pos[1] * z + z / 2, pos[0] * z + z / 2, z * 0.28, 0, 7); ctx.fillStyle = C.orange; ctx.fill();
      // observation panel
      octx.clearRect(0, 0, 150, 150);
      if (partial) {
        for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) { const ch = maze[pos[0] + dr][pos[1] + dc]; octx.fillStyle = ch === "#" ? "#3b4252" : "#fff"; octx.fillRect((dc + 1) * 50, (dr + 1) * 50, 50, 50); octx.strokeStyle = "#ccc"; octx.strokeRect((dc + 1) * 50, (dr + 1) * 50, 50, 50); }
        octx.beginPath(); octx.arc(75, 75, 13, 0, 7); octx.fillStyle = C.orange; octx.fill();
        info.innerHTML = `Observation $o_t$ = local 3×3 view.<br>${alias.length > 1 ? `<span class="hl">${alias.length} different positions give this exact observation!</span><br>→ $o_t$ is <b>not</b> Markov.` : "Location is <b>not</b> directly observed."}`;
      } else {
        octx.fillStyle = C.ink; octx.font = "bold 26px " + U.css("--mono"); octx.textAlign = "center"; octx.fillText(`(${pos[0]}, ${pos[1]})`, 75, 82);
        info.innerHTML = "Observation = exact position $(row, col)$<br>$o_t = s_t$ : fully observable.";
      }
      renderMathInElement(info, { delimiters: [{ left: "$", right: "$", display: false }] });
    }
    draw();
    return {};
  });

  // ================================================================ Markov property: pendulum
  Deck.widget("markovprop", (root) => {
    const W = 560, H = 320, cv = U.canvas(W, H), ctx = cv.ctx;
    let rep = 0, frozen = false, th = 1.0, thd = 0, prevTh = 1.0;
    const tog = U.toggle(["state = θ", "state = (θ, θ̇)", "state = (θₜ, θₜ₋₁)"], (i) => { rep = i; draw(); });
    const fz = U.button("Freeze &amp; predict next", () => { frozen = !frozen; fz.innerHTML = frozen ? "▶ Resume" : "Freeze &amp; predict next"; draw(); }, "primary");
    const txt = el("div", { class: "box", style: { flex: 1 } });
    root.append(el("div", { class: "controls" }, tog, fz), el("div", { class: "row" }, el("div", { class: "box", style: { padding: "4px" } }, cv), txt));
    const px = 280, py = 45, L = 215;
    const deg = (a) => `${a >= 0 ? "+" : "−"}${Math.abs(a * 180 / Math.PI).toFixed(0)}°`;
    function arc(a, r, color, label, dy = 0) {                   // angle from the vertical, with its value
      ctx.strokeStyle = color; ctx.lineWidth = 2.5; ctx.beginPath();
      const a0 = Math.PI / 2, a1 = Math.PI / 2 - a;                // canvas angles: straight down = π/2
      ctx.arc(px, py, r, Math.min(a0, a1), Math.max(a0, a1)); ctx.stroke();
      // label: inside the wedge between vertical and rod when it's wide enough, otherwise on the far side of the vertical
      ctx.font = "700 17px " + U.css("--font"); ctx.fillStyle = color;
      const side = a >= 0 ? 1 : -1;
      if (Math.abs(a) > 0.6) {
        const mid = Math.PI / 2 - a / 2; ctx.textAlign = "center";
        ctx.fillText(label, px + (r + 34) * Math.cos(mid), py + (r + 34) * Math.sin(mid) + 6 + dy);
      } else {
        ctx.textAlign = side > 0 ? "right" : "left";
        ctx.fillText(label, px - side * 10, py + r + 6 + dy);
      }
    }
    function annotate() {
      // vertical reference
      ctx.setLineDash([5, 5]); ctx.strokeStyle = "#9aa1ad"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, py + L + 25); ctx.stroke(); ctx.setLineDash([]);
      arc(th, 78, C.blue, `θ = ${deg(th)}`);
      if (rep === 2) arc(prevTh, 120, "#8a8f99", `θₜ₋₁ = ${deg(prevTh)}`, 14);
      if (rep === 1) {                                              // angular velocity: tangent arrow at the bob
        const bx = px + L * Math.sin(th), by = py + L * Math.cos(th), k = U.clamp(thd * 28, -90, 90);
        const tx = Math.cos(th), ty = -Math.sin(th), ex = bx + tx * k, ey = by + ty * k;
        if (Math.abs(k) > 4) {
          ctx.strokeStyle = C.green; ctx.fillStyle = C.green; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(ex, ey); ctx.stroke();
          const ang = Math.atan2(ey - by, ex - bx); ctx.beginPath(); ctx.moveTo(ex + 4 * Math.cos(ang), ey + 4 * Math.sin(ang));
          ctx.lineTo(ex - 13 * Math.cos(ang - 0.45), ey - 13 * Math.sin(ang - 0.45)); ctx.lineTo(ex - 13 * Math.cos(ang + 0.45), ey - 13 * Math.sin(ang + 0.45)); ctx.fill();
        }
        ctx.font = "700 16px " + U.css("--font"); ctx.fillStyle = C.green; ctx.textAlign = "center"; ctx.fillText(`θ̇ = ${thd >= 0 ? "+" : "−"}${Math.abs(thd).toFixed(1)} rad/s`, bx, by + 42);
      }
      // what the agent receives
      const rows = [["θ = " + deg(th), C.blue], ...[[["θ̇ = ?", C.red]], [[`θ̇ = ${thd >= 0 ? "+" : "−"}${Math.abs(thd).toFixed(1)} rad/s`, C.green]], [["θₜ₋₁ = " + deg(prevTh), "#6b7280"]]][rep]];
      if (rep === 2) rows[0][0] = "θₜ = " + deg(th);
      ctx.fillStyle = "rgba(255,255,255,.92)"; ctx.strokeStyle = frozen ? C.orange : "#c8ccd4"; ctx.lineWidth = frozen ? 3 : 1.5;
      ctx.beginPath(); ctx.roundRect(8, 8, 205, 30 + 24 * rows.length, 10); ctx.fill(); ctx.stroke();
      ctx.textAlign = "left"; ctx.font = "600 13px " + U.css("--font"); ctx.fillStyle = C.muted; ctx.fillText(frozen ? "AGENT OBSERVES (frozen):" : "AGENT OBSERVES:", 18, 28);
      rows.forEach(([t, col], i) => { ctx.font = "700 18px " + U.css("--font"); ctx.fillStyle = col; ctx.fillText(t, 18, 52 + 24 * i); });
    }
    function pend(a, color, alpha = 1, w = 8) {
      ctx.globalAlpha = alpha; ctx.strokeStyle = color; ctx.lineWidth = w; ctx.lineCap = "round";
      const x = px + L * Math.sin(a), y = py + L * Math.cos(a);
      ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(x, y); ctx.stroke();
      ctx.beginPath(); ctx.arc(x, y, 20, 0, 7); ctx.fillStyle = color; ctx.fill(); ctx.globalAlpha = 1;
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = C.ink; ctx.fillRect(px - 60, py - 8, 120, 8);
      if (frozen) {
        const dtp = 0.35;
        if (rep === 0) { pend(th + 0.35, C.orange, 0.35); pend(th - 0.35, C.orange, 0.35); ctx.fillStyle = C.orange; ctx.font = "bold 40px " + U.css("--font"); ctx.fillText("?", px + L * Math.sin(th) + 40, py + L * Math.cos(th) + 20); }
        else if (rep === 1) { pend(th + thd * dtp, C.green, 0.4); }
        else { pend(th + (th - prevTh) * 3, C.green, 0.4); }
      }
      if (rep === 2) pend(prevTh, "#9aa1ad", frozen ? 0.45 : 0.3);          // previous frame is part of the observation
      pend(th, C.blue);
      annotate();
      txt.innerHTML = [
        `<h3>State = angle only</h3><p>The agent observes just one number, θ (in the box). Is the pendulum going <b>left or right</b>? The same θ occurs on the way out and on the way back.</p><p>The future depends on <b>how we got here</b> → <span class="bad">not Markov</span>.</p>`,
        `<h3>State = angle + angular velocity</h3><p>(θ, θ̇) determines the next state exactly (given physics).</p><p>History adds nothing → <span class="good">Markov</span>.</p>`,
        `<h3>State = last two frames</h3><p>Velocity can be <b>inferred</b> from θₜ − θₜ₋₁.</p><p>This is exactly why DQN on Atari stacks the last <b>4 frames</b> as its state.</p>`,
      ][rep];
    }
    const loop = U.loop((dt) => {
      if (frozen) return;
      for (let k = 0; k < 4; k++) { prevTh = th; const h = dt / 4; thd += -9.8 / 2.3 * Math.sin(th) * h; th += thd * h; }
      prevTh = th - thd * 0.08;
      draw();
    });
    draw();
    return { start: loop.start, stop: loop.stop };
  });

  // ================================================================ Student Markov chain (+ rewards / matrix / MC returns)
  Deck.widget("chain", (root) => {
    const withR = root.dataset.rewards === "1", withM = root.dataset.matrix === "1", withG = root.dataset.returns === "1";
    const M = RL.studentMRP, rng = U.rng(+root.dataset.seed || 2);
    const gw = el("div");
    const g = RL.buildChainGraph(gw, { rewards: withR, onclick: (id) => hilite(id) });
    g.svg.style.width = root.dataset.width ? root.dataset.width + "px" : withM || withG ? "620px" : "800px"; g.svg.style.height = "auto";
    const epBox = el("div", { class: "readout", style: { minHeight: "64px", fontSize: "17px" } }, "Press “Sample episode”.");
    const ctr = el("div", { class: "controls" });
    let busy = false, stopFlag = false, gamma = 0.5, returns = [];
    ctr.append(U.button("Sample episode", () => sample(true), "primary"));
    if (withG) ctr.append(U.button("×50 fast", () => { for (let i = 0; i < 50; i++) sample(false); }), U.button("Reset", () => { returns = []; drawPlot(); epBox.textContent = ""; }));
    let side = null, plot = null, tbl = null;
    if (withM) {
      tbl = el("table", { class: "t", style: { fontSize: "15px" } });
      const S = M.states;
      tbl.append(el("tr", {}, el("th", { html: "from \\ to" }), ...S.map((s) => el("th", {}, s))));
      S.forEach((s) => tbl.append(el("tr", { "data-s": s }, el("th", {}, s), ...S.map((t) => el("td", { "data-t": t }, M.P[s][t] ? String(M.P[s][t]) : "")))));
      side = el("div", { class: "stack" }, tbl, el("p", { class: "hint" }, "Click a state: its outgoing arrows = its row. Each row sums to 1. Sleep is absorbing (terminal)."));
    }
    if (withG) {
      const sl = U.slider({ label: "γ", min: 0, max: 1, step: 0.05, value: 0.5, fmt: (v) => v.toFixed(2), oninput: (v) => { gamma = v; returns = []; drawPlot(); } });
      ctr.append(sl);
      const pw = el("div");
      plot = new U.Plot(pw, 460, 300, { xlim: [0, 50], ylim: [-12, 4], xlabel: "episodes sampled from C1", ylabel: "average return", yticks: 6 });
      side = el("div", { class: "stack" }, pw, el("div", { class: "legend" }, el("span", { style: "--c:" + C.blue }, "running mean of G"), el("span", { style: "--c:" + C.orange }, "true v(C1)")));
    }
    if (root.dataset.compact) epBox.style.display = "none";
    root.append(ctr, el("div", { class: "row" }, el("div", { class: "stack" }, gw, epBox), side));
    function hilite(id) {
      g.resetStyles(); g.styleNode(id, "#fff4d6", C.orange);
      Object.values(g.edges).forEach((e) => g.styleEdge(e.id, e.from === id ? "hl" : "dim"));
      if (tbl) U.$$("tr", tbl).forEach((tr) => (tr.style.background = tr.dataset.s === id ? "#fff4d6" : ""));
    }
    function retStr(ep) {
      let G = 0; const terms = [];
      ep.slice(0, -1).forEach((s, k) => { const r = M.R[s], w = Math.pow(gamma, k); G += w * r; if (k < 6) terms.push(`${k ? (r >= 0 ? " + " : " − ") : r < 0 ? "−" : ""}${Math.abs(r)}${k ? "·" + (gamma === 0.5 ? `(½)${k > 1 ? "^" + k : ""}` : `${gamma}^${k}`) : ""}`); });
      return [G, terms.join("") + (ep.length > 7 ? " + …" : "")];
    }
    async function sample(anim) {
      if (busy) return;
      const ep = M.sample(rng);
      if (withG) { const [G, str] = retStr(ep); returns.push(G); epBox.innerHTML = `${ep.join(" → ")}<br><b>G₁ = ${str} = ${G.toFixed(2)}</b>`; drawPlot(); if (!anim) return; }
      else epBox.textContent = ep.join(" → ");
      busy = true; stopFlag = false;
      g.resetStyles();
      for (let i = 0; i < ep.length - 1 && !stopFlag; i++) {
        const e = g.edgeBetween(ep[i], ep[i + 1]);
        g.styleNode(ep[i], "#fff4d6", C.orange);
        if (tbl) U.$$("tr", tbl).forEach((tr) => (tr.style.background = tr.dataset.s === ep[i] ? "#fff4d6" : ""));
        await g.token(e.id, 520);
        g.styleEdge(e.id, "hl");
      }
      g.styleNode(ep[ep.length - 1], "#fff4d6", C.orange);
      await U.sleep(600); g.hideToken(); busy = false;
    }
    function drawPlot() {
      if (!plot) return;
      const v = M.values(gamma).C1, n = returns.length, xmax = Math.max(50, n);
      const run = []; let s = 0; returns.forEach((G, i) => { s += G; run.push([i + 1, s / (i + 1)]); });
      const lo = Math.min(v - 4, ...returns.slice(0, 200)), hi = Math.max(v + 4, ...returns.slice(0, 200));
      plot.o.xlim = [0, xmax]; plot.o.ylim = [Math.floor(lo), Math.ceil(hi)];
      plot.axes(); plot.points(returns.map((G, i) => [i + 1, G]), "rgba(31,95,174,.25)", 3);
      plot.hline(v, C.orange, [6, 4], `v(C1) = ${v.toFixed(2)}`); plot.line(run, C.blue, 3);
    }
    drawPlot();
    return { stop() { stopFlag = true; } };
  });


  // ================================================================ Monte-Carlo estimate of v(C1): tape → histogram → running mean
  Deck.widget("mcvalue", (root) => {
    const M = RL.studentMRP, rng = U.rng(+root.dataset.seed || 5);
    let gamma = 0.9, returns = [], revealed = false, busy = false, stopFlag = false, token = 0;
    // left: compact graph + tape
    const gw = el("div");
    const g = RL.buildChainGraph(gw, { rewards: true });
    g.svg.style.width = "520px"; g.svg.style.height = "auto";
    const tape = el("div", { class: "mc-tape" });
    const total = el("div", { class: "mc-total" });
    // right: histogram + running mean
    const hw = el("div"), rw = el("div");
    const hist = new U.Plot(hw, 540, 220, { xlim: [-20, 10], ylim: [0, 5], xlabel: "return G of one episode", ylabel: "episodes", yticks: 4, pad: [14, 12, 40, 50] });
    const run = new U.Plot(rw, 540, 150, { xlim: [0, 20], ylim: [-8, 2], xlabel: "number of episodes", ylabel: "average G", yticks: 3, pad: [10, 12, 40, 50] });
    const info = el("div", { class: "readout", style: { fontSize: "15px" } });
    const gSl = U.slider({ label: "γ", min: 0, max: 1, step: 0.05, value: 0.9, width: 120, fmt: (v) => v.toFixed(2), oninput: (v) => { gamma = v; clearAll(); } });
    const revealBtn = U.button("Reveal exact v(C1)", () => { revealed = !revealed; revealBtn.innerHTML = revealed ? "Hide exact value" : "Reveal exact v(C1)"; drawPlots(); }, "warn");
    root.append(el("div", { class: "controls" },
      U.button("Sample 1 episode", () => sample(), "primary"), U.button("+10", () => many(10)), U.button("+100", () => many(100)),
      U.button("Reset", () => clearAll()), gSl, U.button("γ=½", () => { gSl.set(0.5); gamma = 0.5; clearAll(); }), U.button("γ=0.9", () => { gSl.set(0.9); gamma = 0.9; clearAll(); }), revealBtn),
      el("div", { class: "row", style: { gap: "20px" } },
        el("div", { class: "stack", style: { flex: "none", gap: "6px" } }, gw, tape, total),
        el("div", { class: "stack", style: { gap: "4px" } }, hw, rw, info)));
    const FR = ["1", "½", "¼", "⅛", "1/16", "1/32", "1/64", "1/128", "1/256", "1/512"];
    const gpow = (k) => (gamma === 0.5 ? FR[k] || "≈0" : Math.pow(gamma, k).toFixed(2));
    const f = (x) => (Math.abs(x) < 0.005 ? "0" : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(2));
    const G = (ep) => ep.slice(0, -1).reduce((a, st, k) => a + Math.pow(gamma, k) * M.R[st], 0);
    function tapeHTML(ep, upto) {   // one column per step: state, reward, weight, contribution
      const n = ep.length - 1, show = Math.min(n, 10), cols = [];
      for (let k = 0; k < show; k++) {
        const st = ep[k], on = k < upto, c = Math.pow(gamma, k) * M.R[st];
        cols.push(`<div class="col ${on ? "on" : ""}"><div class="st">${st}</div><div class="r">${M.R[st] > 0 ? "+" : ""}${M.R[st]}</div><div class="w">×${gpow(k)}</div><div class="c">${on ? f(c) : ""}</div></div>`);
      }
      if (n > show) { const rest = ep.slice(show, -1).reduce((a, st, j) => a + Math.pow(gamma, show + j) * M.R[st], 0); cols.push(`<div class="col ${upto >= n ? "on" : ""}"><div class="st">+${n - show} more</div><div class="r">…</div><div class="w">…</div><div class="c">${upto >= n ? f(rest) : ""}</div></div>`); }
      return `<div class="lab"><div>state</div><div>reward R</div><div>weight γᵏ</div><div>γᵏ·R</div></div>` + cols.join("");
    }
    function showEpisode(ep, upto) {
      tape.innerHTML = tapeHTML(ep, upto);
      const done = upto >= ep.length - 1;
      total.innerHTML = done ? `This episode's return: <b>G = ${f(G(ep))}</b> &nbsp;<span class="muted">(sum of the bottom row)</span>` : `<span class="muted">adding up the discounted rewards…</span>`;
    }
    const sampleNow = () => { const ep = M.sample(rng); returns.push(G(ep)); return ep; };
    async function sample() {
      if (busy) return;
      const ep = M.sample(rng);
      busy = true; stopFlag = false; const my = ++token;
      g.resetStyles(); showEpisode(ep, 0);
      for (let i = 0; i < ep.length - 1 && !stopFlag; i++) {
        g.styleNode(ep[i], "#fff4d6", C.orange);
        await g.token(g.edgeBetween(ep[i], ep[i + 1]).id, 420);
        g.styleEdge(g.edgeBetween(ep[i], ep[i + 1]).id, "hl");
        if (token !== my) return;
        showEpisode(ep, i + 1);
      }
      g.hideToken(); busy = false;
      if (!stopFlag) { returns.push(G(ep)); drawPlots(); }
    }
    function many(n) {
      stopFlag = true; busy = false; token++; g.hideToken(); g.resetStyles();
      let ep; for (let i = 0; i < n; i++) ep = sampleNow();
      showEpisode(ep, ep.length - 1); drawPlots();
    }
    function clearAll() { stopFlag = true; busy = false; token++; g.hideToken(); g.resetStyles(); returns = []; tape.innerHTML = tapeHTML(["C1", "…", "Sleep"].slice(0, 1).concat(["Sleep"]), 0); total.innerHTML = `<span class="muted">Press “Sample 1 episode”: each episode starts in Class 1.</span>`; drawPlots(); }
    function drawPlots() {
      const v = M.values(gamma).C1, n = returns.length;
      const mean = n ? returns.reduce((a, b) => a + b, 0) / n : null;
      const sd = n > 1 ? Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1)) : 0;
      // histogram
      const lo = Math.floor(Math.min(-6, ...returns, v) - 1), hi = Math.ceil(Math.max(2, ...returns, v) + 1);
      const bw = (hi - lo) / 24, counts = new Array(24).fill(0);
      returns.forEach((x) => counts[U.clamp(Math.floor((x - lo) / bw), 0, 23)]++);
      hist.o.xlim = [lo, hi]; hist.o.ylim = [0, Math.max(4, ...counts) * 1.15]; hist.axes();
      hist.bars(counts.map((_, i) => lo + (i + 0.5) * bw), counts, "rgba(31,95,174,.55)", bw * 0.92);
      if (mean != null) { hist.vline(mean, C.blue, []); hist.text(mean, hist.o.ylim[1] * 0.93, ` average = ${mean.toFixed(2)}`, C.blue, "left", "600 14px"); }
      if (revealed) { hist.vline(v, C.orange, [6, 4]); hist.text(v, hist.o.ylim[1] * 0.78, ` exact v(C1) = ${v.toFixed(2)}`, C.orange, "left", "600 14px"); }
      // running mean with ±2 standard errors
      const xs = [], m = [], loB = [], hiB = []; let s = 0, s2 = 0;
      returns.forEach((x, i) => { s += x; s2 += x * x; const k = i + 1, mu = s / k, se = k > 1 ? Math.sqrt(Math.max(0, (s2 - k * mu * mu) / (k - 1)) / k) : 0; xs.push(k); m.push(mu); loB.push(mu - 2 * se); hiB.push(mu + 2 * se); });
      run.o.xlim = [0, Math.max(20, n)]; run.o.ylim = [Math.min(lo, ...loB.slice(2)), Math.max(hi * 0.5, ...hiB.slice(2), 1)]; run.axes();
      if (n > 2) run.band(xs.slice(1), loB.slice(1), hiB.slice(1), C.blue, 0.15);
      run.line(xs.map((x, i) => [x, m[i]]), C.blue, 2.5);
      if (revealed) run.hline(v, C.orange, [6, 4]);
      info.innerHTML = n ? `${n} episodes · average return <b style="color:${C.blue}">${mean.toFixed(2)}</b> ± ${(2 * sd / Math.sqrt(n)).toFixed(2)} (shaded band)<br><span class="muted">each episode's G is different — v(C1) is their <b>average</b></span>`
        : `<span class="muted">The value v(C1) is the average return over (infinitely) many episodes from Class 1.</span>`;
    }
    clearAll();
    return { stop() { stopFlag = true; busy = false; token++; g.hideToken(); } };
  });

  // ================================================================ discount factor
  Deck.widget("discount", (root) => {
    const plot = new U.Plot(root, 760, 250, { xlim: [-0.5, 40.5], ylim: [0, 1.05], xlabel: "steps into the future k", ylabel: "weight γᵏ", yticks: 5 });
    const out = el("div", { class: "readout" });
    const sl = U.slider({ label: "γ", min: 0, max: 1, step: 0.01, value: 0.9, fmt: (v) => v.toFixed(2), oninput: draw });
    root.prepend(el("div", { class: "controls" }, sl, ...[0, 0.5, 0.9, 0.99, 1].map((v) => U.button("γ=" + v, () => { sl.set(v); draw(); }))));
    root.append(out);
    function draw() {
      const g = sl.get(), ks = [...Array(41).keys()];
      plot.axes(); plot.bars(ks, ks.map((k) => Math.pow(g, k)), C.blue, 0.75);
      const hz = g < 1 ? 1 / (1 - g) : Infinity;
      if (hz <= 40) plot.vline(hz, C.orange, [6, 4]);
      out.innerHTML = `effective horizon ≈ 1/(1−γ) = <b>${hz === Infinity ? "∞" : hz.toFixed(1)}</b> steps &nbsp;·&nbsp; +10 reward in 10 steps is worth <b>${(10 * Math.pow(g, 10)).toFixed(2)}</b> now &nbsp;·&nbsp; max return with |R| ≤ 1: <b>${hz === Infinity ? "unbounded" : hz.toFixed(1)}</b>`;
    }
    draw();
    return {};
  });

  // ================================================================ Student MRP values with Bellman check
  Deck.widget("mrpvalues", (root) => {
    const M = RL.studentMRP, gw = el("div");
    const g = RL.buildChainGraph(gw, { rewards: true, onclick: (id) => explain(id) });
    g.svg.style.width = "800px"; g.svg.style.height = "auto";
    const out = el("div", { class: "readout", style: { fontSize: "17px", minHeight: "90px" } }, "Click a state to check its Bellman equation.");
    const sl = U.slider({ label: "γ", min: 0, max: 1, step: 0.01, value: +(root.dataset.gamma || 1), fmt: (v) => v.toFixed(2), oninput: () => { update(); if (sel) explain(sel); } });
    root.append(el("div", { class: "controls" }, sl, ...[0, 0.9, 1].map((v) => U.button("γ = " + v, () => { sl.set(v); update(); if (sel) explain(sel); }))), gw, out);
    let v, sel = null;
    function update() {
      v = M.values(sl.get());
      for (const s of M.states) g.setSub(s, s === "Sleep" ? "" : U.fmt(v[s], 1), v[s] >= 0 ? C.blue : C.red);
    }
    function explain(id) {
      sel = id; g.resetStyles(); g.styleNode(id, "#fff4d6", C.orange);
      Object.values(g.edges).forEach((e) => g.styleEdge(e.id, e.from === id ? "hl" : "dim"));
      if (id === "Sleep") { out.innerHTML = "Sleep is terminal: v(Sleep) = 0"; return; }
      const gm = sl.get(), terms = Object.entries(M.P[id]).map(([t, p]) => `${p}·${U.fmt(v[t], 1)}`);
      const rhs = M.R[id] + gm * Object.entries(M.P[id]).reduce((a, [t, p]) => a + p * v[t], 0);
      out.innerHTML = `<span class="c-est">v(${id})</span> = <span class="c-rew">R(${id})</span> + γ Σ P·<span class="c-next">v(s′)</span> = <span class="c-rew">${M.R[id]}</span> + ${gm.toFixed(2)}·(<span class="c-next">${terms.join(" + ")}</span>) = <b class="c-est">${rhs.toFixed(2)}</b> ✓ &nbsp;(matches ${v[id].toFixed(2)})`;
    }
    update();
    if (root.dataset.select) explain(root.dataset.select);
    return {};
  });

  // ================================================================ iterative evaluation vs direct solve
  Deck.widget("iterate", (root) => {
    const M = RL.studentMRP, S = M.states.filter((s) => s !== "Sleep");
    const plot = new U.Plot(root, 560, 330, { xlim: [-0.5, S.length - 0.5], ylim: [-25, 12], ylabel: "v(s)", xticks: 6, xfmt: (i) => S[i] || "", yticks: 7 });
    const ep = new U.Plot(root, 440, 330, { xlim: [0, 60], ylim: [-3, 2], xlabel: "sweep k", ylabel: "log₁₀ max error", yticks: 5 });
    plot.cv.style.display = ep.cv.style.display = "inline-block";
    let run = false;
    const sl = U.slider({ label: "γ", min: 0.5, max: 1, step: 0.01, value: 0.9, fmt: (v) => v.toFixed(2), oninput: reset });
    const info = el("div", { class: "readout" });
    const runBtn = U.button("▶ Run", () => { run = !run; runBtn.innerHTML = run ? "❚❚" : "▶ Run"; });
    root.prepend(el("div", { class: "controls" }, U.button("One sweep", () => { sweep(); draw(); }, "primary"), runBtn, U.button("Reset", reset), sl));
    root.append(info);
    let V, k, errs, truth;
    function reset() { V = Object.fromEntries(M.states.map((s) => [s, 0])); k = 0; errs = []; truth = M.values(sl.get()); draw(); }
    function sweep() {
      const g = sl.get(), nV = { Sleep: 0 };
      for (const s of S) nV[s] = M.R[s] + g * Object.entries(M.P[s]).reduce((a, [t, p]) => a + p * V[t], 0);
      V = nV; k++;
      errs.push([k, Math.log10(Math.max(1e-3, ...S.map((s) => Math.abs(V[s] - truth[s]))))]);
    }
    function draw() {
      plot.axes(); plot.hline(0, C.muted, [2, 3]);
      plot.bars(S.map((_, i) => i), S.map((s) => V[s]), C.blue, 0.6, 0.85);
      S.forEach((s, i) => { plot.line([[i - 0.4, truth[s]], [i + 0.4, truth[s]]], C.orange, 3, [5, 3]); });
      ep.o.xlim = [0, Math.max(60, k)]; ep.axes(); ep.line(errs, C.purple, 2.5);
      info.innerHTML = `k = ${k} sweeps &nbsp;·&nbsp; v<sub>k+1</sub> = R + γ P v<sub>k</sub> &nbsp;·&nbsp; dashed orange = direct solve (I − γP)<sup>−1</sup>R &nbsp;·&nbsp; each sweep costs O(n²), direct solve O(n³)`;
    }
    let acc = 0;
    const loop = U.loop((dt) => { if (!run) return; acc += dt; if (acc > 0.12) { acc = 0; sweep(); draw(); } });
    reset();
    return { start: loop.start, stop() { loop.stop(); run = false; runBtn.innerHTML = "▶ Run"; } };
  });

  // ================================================================ Student MDP (policy evaluation / optimal)
  Deck.widget("mdp", (root) => {
    const mode = root.dataset.mode || "policy";
    const M = RL.studentMDP, gw = el("div");
    const g = RL.buildMDPGraph(gw, { onclick: (id) => explain(id) });
    g.svg.style.width = "760px"; g.svg.style.height = "auto";
    const out = el("div", { class: "readout", style: { fontSize: "16px", minHeight: "70px" } }, "Click a state to see its Bellman equation.");
    const pairs = { FB: ["Quit", "Facebook"], C1: ["Study", "Facebook"], C2: ["Study", "Sleep"], C3: ["Study", "Pub"] };
    const names = { FB: "Facebook", C1: "Class 1", C2: "Class 2", C3: "Class 3" };
    const pi = { FB: { Quit: 0.5, Facebook: 0.5 }, C1: { Study: 0.5, Facebook: 0.5 }, C2: { Study: 0.5, Sleep: 0.5 }, C3: { Study: 0.5, Pub: 0.5 } };
    const side = el("div", { class: "stack", style: { width: "330px" } });
    let v, sel = null, showQ = mode === "q";
    if (mode === "plain") {
      side.append(el("div", { class: "small", html: "<p>○ states &nbsp; ● chance node</p><p>Arrows out of a state are <b>actions</b> (with their reward).</p><p>Only <b>Pub</b> is stochastic: the <i>environment</i> decides where you end up.</p><p>Sleep is terminal.</p>" }));
    } else if (mode === "policy") {
      side.append(el("b", {}, "Policy π(a|s)"));
      const sls = [];
      for (const s of ["FB", "C1", "C2", "C3"]) {
        const [a, b] = pairs[s];
        const sl = U.slider({ label: `${names[s]}: P(${a})`, min: 0, max: 1, step: 0.05, value: 0.5, width: 120, fmt: (x) => x.toFixed(2), oninput: (x) => { pi[s][a] = x; pi[s][b] = 1 - x; update(); } });
        sl.style.display = "flex"; sl.style.justifyContent = "space-between"; sl.style.fontSize = "16px";
        side.append(sl); sls.push([sl, s, a, b]);
      }
      // student "personas": each is a policy (P(first action) in FB, C1, C2, C3)
      const PERSONAS = [
        ["🎲", "Random", [0.5, 0.5, 0.5, 0.5], "Flips a coin in every state."],
        ["📚", "Always studies", [1, 1, 1, 1], "Quits Facebook, studies in every class."],
        ["⚖️", "Moderate", [0.8, 0.8, 0.8, 0.7], "Mostly studies, but sometimes scrolls or goes to the pub."],
        ["🎉", "Party-goer", [0.5, 0.6, 0.6, 0], "Gets to Class 3… and always heads to the pub instead of the exam."],
        ["📱", "Procrastinator", [0.1, 0.2, 0.5, 0.5], "Hard to get off Facebook; rarely starts studying."],
      ];
      const personaTxt = el("div", { style: { minHeight: "38px", margin: "2px 0", fontSize: "15px", lineHeight: "1.3" } });
      const board = el("div", { style: { fontSize: "15px" } }), tried = {};
      function drawBoard() {
        const rows = PERSONAS.filter((p) => tried[p[1]] != null);
        if (!rows.length) { board.innerHTML = ""; return; }
        const vals = rows.map((p) => tried[p[1]]), top = Math.max(1, ...vals.map(Math.abs));
        board.innerHTML = `<b>v(Class 1) — students tried so far</b>` + rows.sort((a, b) => tried[b[1]] - tried[a[1]]).map((p) => {
          const v = tried[p[1]], w = Math.max(3, (Math.abs(v) / top) * 100);
          return `<div style="display:flex;align-items:center;gap:6px;margin-top:2px;line-height:1.25"><span style="width:150px;white-space:nowrap">${p[0]} ${p[1]}</span>` +
            `<span style="flex:1;height:12px;background:var(--panel);border-radius:6px;position:relative"><span style="position:absolute;left:0;top:0;bottom:0;width:${w}%;border-radius:6px;background:${v >= 0 ? C.blue : C.red}"></span></span>` +
            `<b style="width:44px;text-align:right;color:${v >= 0 ? C.blue : C.red}">${v.toFixed(1)}</b></div>`;
        }).join("");
      }
      const pbtns = el("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px" } });
      PERSONAS.forEach(([ico, name, ps, desc]) => {
        const bt = U.button(`${ico} ${name}`, () => {
          sls.forEach(([sl, s, a, b], i) => { sl.set(ps[i]); pi[s][a] = ps[i]; pi[s][b] = 1 - ps[i]; });
          update(); tried[name] = v.C1; drawBoard();
          personaTxt.innerHTML = `<b>${ico} ${name}:</b> ${desc}`;
        });
        bt.style.fontSize = "15px"; bt.style.padding = "5px 8px";
        pbtns.append(bt);
      });
      side.append(el("b", { style: { marginTop: "4px" } }, "Try a type of student:"), pbtns, personaTxt, board);
    } else {
      side.append(el("div", { class: "controls" }, U.toggle(["show v*", "show q*"], (i) => { showQ = !!i; update(); }, showQ ? 1 : 0)),
        el("p", { class: "small" }, "Green arrows = optimal policy π*(s) = argmaxₐ q*(s,a)."));
    }
    if (mode === "policy") {        // policy mode: readout under the graph, side panel for policy + personas
      g.svg.style.width = "700px"; out.style.fontSize = "14px"; out.style.minHeight = "0";
      root.append(el("div", { class: "row" }, el("div", { class: "stack", style: { flex: "none", gap: "4px" } }, gw, out), side));
    } else {
      if (mode !== "plain") side.append(out);
      root.append(el("div", { class: "row" }, gw, side));
    }
    const actEdge = (s, a) => (s === "C3" && a === "Pub" ? "C3:Pub" : `${s}:${a}`);
    function update() {
      g.resetStyles();
      if (mode === "plain") { v = M.evaluate(pi, 1); return; }
      if (mode === "policy") {
        v = M.evaluate(pi, 1);
        for (const s of ["FB", "C1", "C2", "C3"]) for (const a of Object.keys(M.A[s])) {
          const e = g.edges[actEdge(s, a)]; e.path.setAttribute("stroke-width", 1 + 6 * pi[s][a]); e.path.setAttribute("stroke-opacity", 0.25 + 0.75 * pi[s][a]);
        }
        for (const s of M.states) g.setSub(s, s === "Sleep" ? "" : "v=" + U.fmt(v[s], 1), v[s] >= 0 ? C.blue : C.red);
      } else {
        v = M.optimal(1);
        for (const s of ["FB", "C1", "C2", "C3"]) {
          const qs = Object.keys(M.A[s]).map((a) => [a, M.q(v, s, a)]), best = Math.max(...qs.map((x) => x[1]));
          for (const [a, q] of qs) {
            const id = actEdge(s, a); g.styleEdge(id, q >= best - 1e-9 ? "green" : "dim");
            const e = g.edges[id], base = e.lab.dataset.base || e.lab.textContent; e.lab.dataset.base = base;
            e.lab.textContent = showQ ? `${base}  (q*=${U.fmt(q, 1)})` : base;
          }
        }
        for (const s of M.states) g.setSub(s, s === "Sleep" ? "" : (showQ ? "" : "v*=" + U.fmt(v[s], 0)), C.green);
      }
      if (sel) explain(sel, true);
    }
    function explain(s, silent) {
      if (mode === "plain") return;
      sel = s;
      if (!M.A[s] || !Object.keys(M.A[s]).length) { out.innerHTML = "Sleep is terminal: v = 0"; return; }
      const acts = Object.keys(M.A[s]);
      const qstr = (a) => { const d = M.A[s][a]; return `${d.r} + ${Object.entries(d.P).map(([t, p]) => (p === 1 ? "" : p + "·") + U.fmt(v[t], 1)).join(" + ")}`; };
      if (mode === "policy") {
        const total = acts.reduce((acc, a) => acc + pi[s][a] * M.q(v, s, a), 0);
        out.innerHTML = `v<sub>π</sub>(${names[s]}) = Σ<sub>a</sub> π(a|s)·[R + Σ P·v<sub>π</sub>(s′)]<br>= ${acts.map((a) => `${pi[s][a].toFixed(2)}·(${qstr(a)})`).join("<br>&nbsp;+ ")}<br>= <b>${total.toFixed(2)}</b> ✓`;
      } else {
        out.innerHTML = `v*(${names[s]}) = max<sub>a</sub> [R + Σ P·v*(s′)]<br>= max( ${acts.map((a) => `${a}: ${qstr(a)} = ${U.fmt(M.q(v, s, a), 1)}`).join(",<br>&nbsp;&nbsp;") } )<br>= <b>${U.fmt(Math.max(...acts.map((a) => M.q(v, s, a))), 1)}</b>`;
      }
    }
    update();
    if (root.dataset.select) explain(root.dataset.select);
    return {};
  });

  // ================================================================ 4x4 grid: random vs optimal policy
  Deck.widget("gridwalk", (root) => {
    const G = new RL.Grid(4, 4, { step: -1, gamma: 1 });
    G.type[0] = "T"; G.type[15] = "T";
    const view = new RL.GridView(el("div"), G, 96, { fontSize: 18 });
    let pol = 0, showV = false, running = false, s = null, steps = 0, lens = [[], []];
    const rnd = G.randomPolicy();
    let V = Array(16).fill(0); for (let k = 0; k < 400; k++) V = G.evalSweep(V, rnd);
    let Vs = Array(16).fill(0); for (let k = 0; k < 20; k++) Vs = G.viSweep(Vs);
    const opt = G.greedy(Vs);
    const info = el("div", { class: "readout", style: { fontSize: "17px" } });
    const rng = U.rng(9);
    root.append(el("div", { class: "controls" },
      U.toggle(["Random policy", "Optimal policy"], (i) => { pol = i; draw(); }),
      U.button("Run episode", () => { s = rng.choice(G.states()); steps = 0; view.trail = [s]; running = true; }, "primary"),
      U.button("Show v<sub>π</sub>", () => { showV = !showV; draw(); })),
      el("div", { class: "row" }, view.cv, info));
    function draw() {
      view.pi = pol ? opt : rnd; view.V = showV ? (pol ? Vs : V) : null; view.o.showValues = showV; view.agent = s; view.draw();
      const avg = (a) => (a.length ? (a.reduce((x, y) => x + y) / a.length).toFixed(1) : "–");
      info.innerHTML = `Reward −1 per step, γ = 1<br>Grey = terminal<br><br>steps this episode: <b>${steps}</b><br>avg steps (random): ${avg(lens[0])} over ${lens[0].length}<br>avg steps (optimal): ${avg(lens[1])} over ${lens[1].length}<br><br>${showV ? "v<sub>π</sub>(s) = −(expected steps to finish)" : ""}`;
    }
    let acc = 0;
    const loop = U.loop((dt) => {
      if (!running) return;
      acc += dt; if (acc < (pol ? 0.25 : 0.08)) return; acc = 0;
      const p = (pol ? opt : rnd)[s], a = rng.categorical(p);
      s = G.move(s, a); steps++; view.trail.push(s);
      if (G.isTerminal(s) || steps > 400) { running = false; lens[pol].push(steps); setTimeout(() => { if (!running) { view.trail = []; s = null; draw(); } }, 900); }
      draw();
    });
    draw();
    return { start: loop.start, stop() { loop.stop(); running = false; } };
  });

  // ================================================================ backup diagram: expectation vs optimality
  Deck.widget("backup", (root) => {
    const W = 700, H = 380, svg = U.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
    const out = el("div", { class: "readout", style: { fontSize: "17px", minHeight: "120px" } });
    let opt = root.dataset.mode === "max", gamma = 1, stage = 0;
    const rng = U.rng(4);
    let pa = 0.5, r = [-2, 1], P = [[1, 0], [0.4, 0.6]], vs = [[3, 0], [2, 6]];
    const tog = U.toggle(["Bellman expectation (average)", "Bellman optimality (max)"], (i) => { opt = !!i; stage = 0; draw(); }, opt ? 1 : 0);
    const sl = U.slider({ label: "π(a₁|s)", min: 0, max: 1, step: 0.05, value: 0.5, fmt: (x) => x.toFixed(2), oninput: (x) => { pa = x; draw(); } });
    root.append(el("div", { class: "controls" }, tog, sl,
      U.button("Back up one level ↑", () => { stage = Math.min(2, stage + 1); draw(); }, "primary"),
      U.button("New numbers", () => { r = [rng.int(7) - 3, rng.int(7) - 3]; const p = +(0.1 + 0.8 * rng()).toFixed(1); P = [[1, 0], [p, +(1 - p).toFixed(1)]]; vs = [[rng.int(11) - 3, rng.int(11) - 3], [rng.int(11) - 3, rng.int(11) - 3]]; stage = 0; draw(); })),
      root.dataset.layout === "wide" ? el("div", { class: "row center" }, svg, out) : el("div", { class: "row" }, svg, out));
    if (root.dataset.small) { svg.style.width = root.dataset.small + "px"; svg.style.height = "auto"; svg.style.flex = "none"; }
    function draw() {
      svg.innerHTML = "";
      const S = (t, a, ...k) => { const e = U.svg(t, a, ...k); svg.append(e); return e; };
      const root_ = [350, 50], acts = [[190, 175], [510, 175]], leaves = [[[110, 320], [270, 320]], [[430, 320], [590, 320]]];
      const q = [0, 1].map((a) => r[a] + gamma * (P[a][0] * vs[a][0] + P[a][1] * vs[a][1]));
      const vroot = opt ? Math.max(...q) : pa * q[0] + (1 - pa) * q[1], best = q[0] >= q[1] ? 0 : 1;
      const line = (a, b, col, w) => S("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: col, "stroke-width": w });
      [0, 1].forEach((a) => {
        const hot = stage >= 2 && (!opt || a === best);
        line(root_, acts[a], hot ? C.orange : "#9aa1ad", hot ? 4 : 2);
        S("text", { x: (root_[0] + acts[a][0]) / 2 + (a ? 30 : -30), y: 112, "text-anchor": "middle", "font-size": 16, fill: C.muted }, opt ? `a${a + 1}` : `π=${(a ? 1 - pa : pa).toFixed(2)}`);
        [0, 1].forEach((j) => {
          if (P[a][j] === 0) return;
          line(acts[a], leaves[a][j], stage >= 1 ? C.blue : "#9aa1ad", stage >= 1 ? 3.5 : 2);
          S("text", { x: (acts[a][0] + leaves[a][j][0]) / 2 + (j ? 16 : -16), y: 255, "text-anchor": "middle", "font-size": 15, fill: C.muted }, `P=${P[a][j]}`);
          S("circle", { cx: leaves[a][j][0], cy: leaves[a][j][1], r: 24, fill: "#fff", stroke: C.ink, "stroke-width": 2.5 });
          S("text", { x: leaves[a][j][0], y: leaves[a][j][1] + 6, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: C.blue }, vs[a][j]);
        });
        S("circle", { cx: acts[a][0], cy: acts[a][1], r: 12, fill: C.ink });
        S("text", { x: acts[a][0] + (a ? 22 : -22), y: acts[a][1] + 6, "text-anchor": a ? "start" : "end", "font-size": 16, "font-weight": 700, fill: C.green }, `r=${r[a]}`);
        if (stage >= 1) S("text", { x: acts[a][0], y: acts[a][1] - 22, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: opt && a === best && stage >= 2 ? C.green : C.purple }, `q=${U.fmt(q[a], 1)}`);
      });
      S("circle", { cx: root_[0], cy: root_[1], r: 28, fill: stage >= 2 ? "#fff4d6" : "#fff", stroke: stage >= 2 ? C.orange : C.ink, "stroke-width": 3 });
      S("text", { x: root_[0], y: root_[1] + 6, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: C.orange }, stage >= 2 ? U.fmt(vroot, 1) : "s");
      S("text", { x: 20, y: 30, "font-size": 15, fill: C.muted }, "○ state   ● action   (γ = 1)");
      const qline = [0, 1].map((a) => `q(s,a${a + 1}) = ${r[a]} + (${P[a].map((p, j) => (p ? `${p}·${vs[a][j]}` : null)).filter(Boolean).join(" + ")}) = ${U.fmt(q[a], 1)}`).join("<br>");
      out.innerHTML = stage === 0 ? "Leaves hold v(s′). Press “Back up one level”." :
        stage === 1 ? `<b>Step 1 — average over what the environment does:</b><br>${qline}` :
        `${qline}<br><b>Step 2 — ${opt ? "agent picks the best action" : "average over what the agent does"}:</b><br>` +
        (opt ? `v*(s) = max(${U.fmt(q[0], 1)}, ${U.fmt(q[1], 1)}) = <b>${U.fmt(vroot, 1)}</b>` : `v<sub>π</sub>(s) = ${pa.toFixed(2)}·${U.fmt(q[0], 1)} + ${(1 - pa).toFixed(2)}·${U.fmt(q[1], 1)} = <b>${U.fmt(vroot, 1)}</b>`);
      sl.style.opacity = opt ? 0.35 : 1;
    }
    draw();
    return {};
  });
})();

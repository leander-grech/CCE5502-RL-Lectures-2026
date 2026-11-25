/* Assignment-briefing widgets */
(function () {
  "use strict";
  const { el, C } = U;

  // ---------------------------------------------------------------- SHA-256 (pure JS: works from file://)
  function sha256(bytes) {
    const K = new Uint32Array([0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);
    const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    const l = bytes.length, n = (((l + 9) >> 6) + 1) << 6, m = new Uint8Array(n);
    m.set(bytes); m[l] = 0x80;
    const dv = new DataView(m.buffer); dv.setUint32(n - 4, l * 8);
    const w = new Uint32Array(64), r = (x, k) => (x >>> k) | (x << (32 - k));
    for (let o = 0; o < n; o += 64) {
      for (let i = 0; i < 16; i++) w[i] = dv.getUint32(o + 4 * i);
      for (let i = 16; i < 64; i++) { const s0 = r(w[i - 15], 7) ^ r(w[i - 15], 18) ^ (w[i - 15] >>> 3), s1 = r(w[i - 2], 17) ^ r(w[i - 2], 19) ^ (w[i - 2] >>> 10); w[i] = (w[i - 16] + s0 + w[i - 7] + s1) | 0; }
      let [a, b, c, d, e, f, g, h] = H;
      for (let i = 0; i < 64; i++) {
        const t1 = (h + (r(e, 6) ^ r(e, 11) ^ r(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i]) | 0;
        const t2 = ((r(a, 2) ^ r(a, 13) ^ r(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) | 0;
        h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
      }
      H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }
    const out = new Uint8Array(32), ov = new DataView(out.buffer);
    H.forEach((x, i) => ov.setUint32(4 * i, x));
    return out;
  }
  const enc = (s) => new TextEncoder().encode(s);
  const hex = (b) => Array.from(b, (x) => x.toString(16).padStart(2, "0")).join("");

  // mirror of rl_assignment/variant.py
  function makeVariant(raw) {
    const sid = raw.trim().replace(/\s*-\s*[A-Za-z]+\d+\s*$/, "").replace(/[^0-9A-Za-z]/g, "").toUpperCase();
    if (sid.length < 4) return null;
    const d = sha256(enc(`CCE5502-2026:${sid}`));
    const u = (k, lo, hi) => lo + (hi - lo) * (((d[2 * k] << 8) | d[2 * k + 1]) / 65535);
    const rnd = (x, n) => Number(x.toFixed(n));
    return {
      sid, fp: hex(sha256(enc(`CCE5502-2026:${sid}:fp`))).slice(0, 10),
      L: rnd(u(0, 0.35, 0.8), 3), mp: rnd(u(1, 0.05, 0.25), 3), mc: rnd(u(2, 0.7, 1.6), 3), F: rnd(u(3, 8, 13), 2),
      noise: rnd(u(4, 0, 0.01), 4), delay: 3 + (d[20] % 2), side: d[10] % 2 === 0 ? "left" : "right",
      strength: rnd(u(6, 0.6, 0.85), 2), x: rnd((d[14] % 2 ? 1 : -1) * u(8, 0.6, 1.4), 2),
    };
  }
  window.makeVariant = makeVariant;

  Deck.widget("variantcard", (root) => {
    const inp = el("input", { type: "text", value: "123456M", style: { font: "22px " + U.css("--mono"), padding: "6px 12px", width: "220px", borderRadius: "8px", border: "2px solid " + C.blue } });
    const card = el("pre", { style: { fontSize: "19px", minHeight: "250px" } });
    const W = 360, H = 250, cv = U.canvas(W, H), ctx = cv.ctx;
    root.append(el("div", { class: "controls" }, el("b", {}, "Student number:"), inp, el("span", { class: "hint" }, "type any ID — same result as setup_student.py")), el("div", { class: "row center" }, card, el("div", { class: "box", style: { padding: "6px" } }, cv)));
    const f = (x, n) => x.toFixed(n);
    function draw() {
      const v = makeVariant(inp.value);
      if (!v) { card.textContent = "(type at least 4 characters)"; return; }
      card.textContent = [
        `variant for ${v.sid}  (fingerprint ${v.fp})`,
        `  pole half-length   ${f(v.L, 3)} m    (std 0.500)`,
        `  pole mass          ${f(v.mp, 3)} kg   (std 0.100)`,
        `  cart mass          ${f(v.mc, 3)} kg   (std 1.000)`,
        `  actuator force     ${f(v.F, 2)} N    (std 10.00)`,
        `  observation noise  σ = ${f(v.noise, 4)}`,
        `  faulty actuator    ${v.delay} steps (${20 * v.delay} ms) late,`,
        `                     ${v.side.toUpperCase()} push only ${Math.round(100 * v.strength)}%`,
        `  Part D target      x* = ${v.x >= 0 ? "+" : ""}${f(v.x, 2)} m`,
      ].join("\n");
      // sketch: standard (grey) vs yours (colour), to scale
      ctx.clearRect(0, 0, W, H);
      const ground = 200, sc = 150;
      ctx.strokeStyle = "#aab0bb"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(10, ground + 14); ctx.lineTo(W - 10, ground + 14); ctx.stroke();
      const cart = (cx, mass, len, pm, col, alpha) => {
        ctx.globalAlpha = alpha; const w = 50 * Math.sqrt(mass), h = 24 * Math.sqrt(mass);
        ctx.fillStyle = col; ctx.fillRect(cx - w / 2, ground - h + 12, w, h);
        ctx.strokeStyle = col === C.ink ? "#c98a3a" : "#aab0bb"; ctx.lineWidth = 4 + 30 * pm; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(cx, ground - h + 12); ctx.lineTo(cx, ground - h + 12 - 2 * len * sc * 0.9); ctx.stroke(); ctx.globalAlpha = 1;
      };
      cart(95, 1, 0.5, 0.1, "#aab0bb", 0.9); cart(260, v.mc, v.L, v.mp, C.ink, 1);
      ctx.fillStyle = C.muted; ctx.font = "14px " + U.css("--font"); ctx.textAlign = "center";
      ctx.fillText("standard", 95, H - 8); ctx.fillText("yours (to scale)", 260, H - 8);
    }
    inp.addEventListener("input", draw);
    inp.addEventListener("keydown", (e) => e.stopPropagation());
    draw();
    return {};
  });

  // ---------------------------------------------------------------- delayed actuator timeline
  Deck.widget("delayline", (root) => {
    const W = 1060, H = 190, cv = U.canvas(W, H), ctx = cv.ctx;
    const sl = U.slider({ label: "delay d (steps)", min: 0, max: 5, step: 1, value: 3, oninput: draw });
    root.append(el("div", { class: "controls" }, sl, el("span", { class: "hint" }, "each step = 20 ms")), cv);
    const cmds = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1];
    function draw() {
      const d = sl.get(), z = 66, x0 = 150;
      ctx.clearRect(0, 0, W, H);
      ctx.font = "600 16px " + U.css("--font"); ctx.fillStyle = C.ink; ctx.textAlign = "left";
      ctx.fillText("agent commands", 8, 52); ctx.fillText("motor executes", 8, 132);
      for (let t = 0; t < cmds.length; t++) {
        const x = x0 + t * z;
        ctx.fillStyle = C.muted; ctx.font = "13px " + U.css("--font"); ctx.textAlign = "center"; ctx.fillText(`t=${t}`, x + 24, 18);
        const box = (y, a, col, faded) => { ctx.fillStyle = faded ? "#eceef1" : col; ctx.fillRect(x, y, 50, 36); ctx.fillStyle = faded ? C.muted : "#fff"; ctx.font = "bold 20px " + U.css("--font"); ctx.fillText(a === 1 ? "→" : "←", x + 25, y + 25); };
        box(32, cmds[t], cmds[t] ? C.blue : C.orange);
        const e = t - d, a = e >= 0 ? cmds[e] : t % 2;
        box(112, a, a ? C.blue : C.orange, e < 0);
        if (e >= 0 && d > 0 && t < cmds.length) { ctx.strokeStyle = "rgba(29,36,51,.25)"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.moveTo(x0 + e * z + 25, 68); ctx.lineTo(x + 25, 112); ctx.stroke(); }
      }
      ctx.fillStyle = C.muted; ctx.font = "14px " + U.css("--font"); ctx.textAlign = "left";
      ctx.fillText(d ? `The command chosen at step t is executed at step t+${d} (${20 * d} ms later). Grey = still executing the start-up pattern.` : "No delay: every command is executed immediately.", 8, H - 6);
    }
    draw();
    return {};
  });

  // ---------------------------------------------------------------- why several seeds
  Deck.widget("seeds", (root) => {
    const pl = new U.Plot(el("div"), 500, 280, { xlim: [0, 100], ylim: [0, 520], xlabel: "training steps (×1000) — synthetic curves", ylabel: "episode length", yticks: 5 });
    const n = U.slider({ label: "number of seeds", min: 1, max: 8, step: 1, value: 1, oninput: draw });
    let mode = 0;
    root.append(el("div", { class: "controls" }, n, U.toggle(["show every seed", "mean + min–max band"], (i) => { mode = i; draw(); }), U.button("new random seeds", () => { base += 17; draw(); })), pl.cv);
    let base = 1;
    const cols = [C.blue, C.orange, C.green, C.purple, C.teal, C.pink, C.gold, C.red];
    function curve(seed) {
      const r = U.rng(seed), up = 20 + r() * 40, ys = [];
      let v = 20;
      for (let t = 0; t <= 100; t++) { const target = t < up ? 20 + (t / up) * 480 * (0.5 + 0.5 * r()) : 500 - (r() < 0.04 ? 300 * r() : 0); v += (target - v) * 0.25 + r.normal() * 25; ys.push([t, U.clamp(v, 9, 500)]); }
      return ys;
    }
    function draw() {
      const cs = Array.from({ length: n.get() }, (_, i) => curve(base * 100 + i));
      pl.axes();
      if (mode === 0 || cs.length === 1) cs.forEach((c, i) => pl.line(c, cols[i], 2));
      else {
        const xs = cs[0].map((p) => p[0]), lo = xs.map((_, t) => Math.min(...cs.map((c) => c[t][1]))), hi = xs.map((_, t) => Math.max(...cs.map((c) => c[t][1])));
        pl.band(xs, lo, hi, C.blue, 0.2); pl.line(xs.map((x, t) => [x, cs.reduce((a, c) => a + c[t][1], 0) / cs.length]), C.blue, 3);
      }
    }
    draw();
    return {};
  });
})();

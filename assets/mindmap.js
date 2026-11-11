/* Mind-map widget: a lecture overview at a glance — and the colour scheme of the whole deck.
 *
 *   <div data-widget="mindmap">
 *     <script type="application/json">
 *       { "center": ["Lecture 1", "Intro to RL"],
 *         "branches": [ { "label": "What is RL?", "dark": false,
 *                         "leaves": [ ["Agent–environment loop", "agent–environment loop"],       // → that slide
 *                                     ["Quick check", "Quick check", 3] ] }, ... ] }                // → slide, 3 build steps shown
 *     </script>
 *   </div>
 *
 * On the slide: the centre appears first, then the branches grow one after another automatically
 * (speed set by TS in the widget). Arriving from a later slide (e.g. with M) shows the finished map immediately.
 * Hover a concept: its whole path (centre → branch → concept) glows. Click a concept → jump to its slide
 * (and build step). Click a branch pill → jump to the first slide of that part. Coming back to the map
 * lights up the concept you just left ("you are here").
 *
 * Deck colours: every slide gets the colour of the part it belongs to (heading bar, bullets, boxes,
 * buttons, highlighted text, section dividers, progress bar). Parts with "dark": true use an inverted theme
 * on their text slides.
 */
(function () {
  "use strict";
  const { el, C } = U;
  const PALETTE = ["#1f5fae", "#0b7285", "#2b8a3e", "#d9480f", "#c2255c", "#6741d9", "#b7791f", "#4c6ef5"];
  Deck.PALETTE = PALETTE;

  const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
  const mix = (h, t, w) => { const a = hex2rgb(h), b = hex2rgb(t); return "#" + a.map((x, i) => Math.round(x * (1 - w) + b[i] * w).toString(16).padStart(2, "0")).join(""); };

  // ------------------------------------------------------------------ colour every slide by its part
  Deck.onBoot((slides) => {
    const holder = document.querySelector('[data-widget="mindmap"] script');
    if (!holder) return;
    const data = JSON.parse(holder.textContent);
    // part of each slide: the branch whose concept targets it; untargeted slides inherit from the slide before;
    // section dividers take the part of the slide after them
    const own = new Map();
    data.branches.forEach((b, k) => b.leaves.forEach((l) => { const i = Deck.findSlide(Array.isArray(l) ? l[1] : l); if (i >= 0 && !own.has(i)) own.set(i, k); }));
    if (!own.size) return;
    const part = new Array(slides.length).fill(null);
    let curPart = null;
    for (let i = Math.min(...own.keys()); i < slides.length; i++) { if (own.has(i)) curPart = own.get(i); part[i] = curPart; }
    for (let i = slides.length - 2; i >= 0; i--) if (slides[i].classList.contains("section") && part[i + 1] != null) part[i] = part[i + 1];
    slides.forEach((s, i) => {
      const k = part[i];
      if (k == null) return;
      const b = data.branches[k], col = b.color || PALETTE[k % PALETTE.length];
      s.dataset.part = k + 1;
      s.style.setProperty("--accent", col);
      s.style.setProperty("--accent-soft", mix(col, "#ffffff", 0.88));
      s.style.setProperty("--accent-light", mix(col, "#ffffff", 0.45));
      s.style.setProperty("--hl", col);
      if (s.classList.contains("section")) {
        s.style.background = `linear-gradient(135deg, ${mix(col, "#000000", 0.5)} 0%, ${col} 100%)`;
        const kick = s.querySelector(".kicker");
        if (kick) kick.textContent = `Part ${k + 1} · ${b.label}`;
      } else if (b.dark && !s.classList.contains("brk") && !s.querySelector("[data-widget]")) {
        s.classList.add("dark");
      }
    });
  });

  // ------------------------------------------------------------------ the widget
  Deck.widget("mindmap", (root) => {
    const TS = 0.15;  // animation time scale: 1 = original slow build-up, smaller = faster
    const data = JSON.parse(root.querySelector("script").textContent);
    const W = 1232, H = 606, cx = W / 2, cy = H / 2;
    const svg = U.svg("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, class: "mindmap" });
    root.style.margin = "0 -40px";
    root.append(svg);
    const uid = "mm" + Math.random().toString(36).slice(2, 7);
    const defs = U.svg("defs");
    svg.append(defs);
    const font = U.css("--font");
    const measure = (() => { const c = document.createElement("canvas").getContext("2d"); return (t, px, w = 400) => { c.font = `${w} ${px}px ${font}`; return c.measureText(t).width; }; })();
    let x0 = cx, x1 = cx, yT = cy, yB = cy;
    const grow = (xa, xb, ya, yb) => { x0 = Math.min(x0, xa); x1 = Math.max(x1, xb); yT = Math.min(yT, ya); yB = Math.max(yB, yb); };

    const B = data.branches, nL = Math.ceil(B.length / 2);
    const sides = [B.slice(0, nL), B.slice(nL)];
    const leafGap = 29, groupGap = 30;
    const gLinks = U.svg("g"), gNodes = U.svg("g");
    svg.append(gLinks, gNodes);


    // timeline items: {el, at (s after reveal), dur, len?}
    const branches = [];   // per branch: {col, items, parts:{trunk, pill, leaves:[{link, leaf}]}, revealedAt}
    const centreItems = [];

    sides.forEach((list, side) => {
      const dir = side === 0 ? -1 : 1;
      const bx = cx + dir * 245, lx = cx + dir * 372;
      const heights = list.map((b) => Math.max(1, b.leaves.length) * leafGap);
      const total = heights.reduce((a, b) => a + b, 0) + groupGap * (list.length - 1);
      let y0 = cy - total / 2;
      list.forEach((b, j) => {
        const idx = side === 0 ? j : nL + j, col = b.color || PALETTE[idx % PALETTE.length];
        const f = U.svg("filter", { id: `${uid}g${idx}`, x: "-30%", y: "-60%", width: "160%", height: "220%" },
          U.svg("feDropShadow", { dx: 0, dy: 0, stdDeviation: 7, "flood-color": col, "flood-opacity": 1 }));
        defs.append(f);
        const by = y0 + heights[j] / 2;
        const br = { col, items: [], trunk: null, pill: null, leaves: [], revealedAt: null, glowId: `url(#${uid}g${idx})` };
        br.trunk = U.svg("path", { d: `M${cx},${cy} C${cx + dir * 120},${cy} ${bx - dir * 120},${by} ${bx},${by}`, fill: "none", stroke: col, "stroke-width": 7, "stroke-linecap": "round", opacity: 0.85 });
        gLinks.append(br.trunk); br.items.push({ el: br.trunk, at: 0, dur: 1.0 * TS, len: true });
        const label = `${idx + 1}  ${b.label}`, bw = measure(label, 19, 700) + 30;
        grow(bx - bw / 2, bx + bw / 2, by - 19, by + 19);
        br.pill = U.svg("g", { class: "mm-pill", style: "cursor:pointer" },
          U.svg("rect", { x: bx - bw / 2, y: by - 19, width: bw, height: 38, rx: 19, fill: col }),
          U.svg("text", { x: bx, y: by + 6.5, "text-anchor": "middle", "font-size": 19, "font-weight": 700, fill: "#fff" }, label),
          U.svg("title", {}, `Part ${idx + 1}: go to its first slide`));
        gNodes.append(br.pill); br.items.push({ el: br.pill, at: 0.8 * TS, dur: 0.6 * TS });
        br.pill.addEventListener("click", (e) => { e.stopPropagation(); const i = Math.min(...b.leaves.map((l) => Deck.findSlide(Array.isArray(l) ? l[1] : l)).filter((x) => x >= 0)); if (isFinite(i)) Deck.go(i); });
        br.pill.addEventListener("mouseenter", () => highlight(idx, -1));
        br.pill.addEventListener("mouseleave", () => highlight(-1));
        b.leaves.forEach((lf, k) => {
          const [text, target, step] = Array.isArray(lf) ? lf : [lf, null, null];
          const ly = y0 + leafGap * (k + 0.5), ex = bx + dir * (bw / 2);
          const link = U.svg("path", { d: `M${ex},${by} C${ex + dir * 40},${by} ${lx - dir * 50},${ly} ${lx - dir * 6},${ly}`, fill: "none", stroke: col, "stroke-width": 2.2, opacity: 0.75 });
          gLinks.append(link);
          const tw = measure(text, 17) + 16, tx = side === 0 ? lx - tw : lx;
          grow(tx, tx + tw, ly - 13, ly + 13);
          const leaf = U.svg("g", { class: "mm-leaf", style: target ? "cursor:pointer" : "" },
            U.svg("rect", { x: tx, y: ly - 13, width: tw, height: 26, rx: 13, fill: "#fff", stroke: col, "stroke-width": 1.5 }),
            U.svg("circle", { cx: lx - dir * 6, cy: ly, r: 3.5, fill: col }),
            U.svg("text", { x: tx + tw / 2, y: ly + 5.5, "text-anchor": "middle", "font-size": 17, fill: C.ink }, text));
          if (target) {
            leaf.append(U.svg("title", {}, `Go to: ${target}${step ? ` (step ${step})` : ""}`));
            leaf.addEventListener("click", (e) => { e.stopPropagation(); const i = Deck.findSlide(target); if (i >= 0) Deck.go(i, step || undefined); else console.warn("mindmap: no slide titled", target); });
          }
          leaf.addEventListener("mouseenter", () => highlight(idx, k));
          leaf.addEventListener("mouseleave", () => highlight(-1));
          gNodes.append(leaf);
          const at = (1.5 + k * 0.5) * TS;
          br.items.push({ el: link, at, dur: 0.6 * TS, len: true }, { el: leaf, at: at + 0.3 * TS, dur: 0.6 * TS });
          br.leaves.push({ link, leaf, target });
        });
        branches[idx] = br;
        y0 += heights[j] + groupGap;
      });
    });

    // centre node (on top)
    const [t1, t2] = data.center, cw = Math.max(measure(t1, 17, 600), measure(t2, 25, 700)) + 56;
    const centre = U.svg("g", {},
      U.svg("rect", { x: cx - cw / 2, y: cy - 44, width: cw, height: 88, rx: 44, fill: C.ink }),
      U.svg("text", { x: cx, y: cy - 8, "text-anchor": "middle", "font-size": 17, "font-weight": 600, fill: "#9fc3f0", "letter-spacing": "0.08em" }, t1.toUpperCase()),
      U.svg("text", { x: cx, y: cy + 22, "text-anchor": "middle", "font-size": 25, "font-weight": 700, fill: "#fff" }, t2));
    gNodes.append(centre);
    centreItems.push({ el: centre, at: 0, dur: 0.8 * TS });
    let centreAt = null;

    // fit to the box
    const half = Math.max(cx - x0, x1 - cx) + 10, halfH = Math.max(cy - yT, yB - cy) + 10;
    const sc = Math.max(half / (W / 2), halfH / (H / 2)), vw = W * sc, vh = H * sc;
    svg.setAttribute("viewBox", `${cx - vw / 2} ${cy - vh / 2} ${vw} ${vh}`);

    // ---------------------------------------------------------------- highlight / glow
    let pinned = null;  // {b, k} kept lit after returning from a slide
    function highlight(bi, k = -1) {
      const target = bi < 0 ? pinned : { b: bi, k };
      branches.forEach((br, j) => {
        const on = !target || target.b === j;
        const glowBranch = target && target.b === j;
        br.trunk.style.opacity = on ? "" : 0.12;
        br.trunk.style.filter = glowBranch ? br.glowId : "";
        br.trunk.style.strokeWidth = glowBranch ? 10 : "";
        br.pill.style.filter = glowBranch ? br.glowId : "";
        br.pill.style.opacity = br.revealedAt == null ? 0 : on ? 1 : 0.25;
        br.leaves.forEach((l, i) => {
          const lit = glowBranch && (target.k < 0 || target.k === i);
          const dim = target && !lit;
          if (br.revealedAt != null) l.leaf.style.opacity = dim ? 0.22 : 1;
          l.link.style.opacity = dim ? 0.1 : "";
          l.leaf.style.filter = lit && target.k >= 0 ? br.glowId : "";
          l.link.style.filter = lit && target.k >= 0 ? br.glowId : "";
          l.link.style.strokeWidth = lit && target.k >= 0 ? 4 : "";
          const r = l.leaf.querySelector("rect");
          r.setAttribute("fill", lit && target.k >= 0 ? "#fff8e1" : "#fff"); r.setAttribute("stroke-width", lit && target.k >= 0 ? 3 : 1.5);
        });
      });
    }

    // ---------------------------------------------------------------- animation
    const allItems = () => [...centreItems.map((it) => [it, centreAt]), ...branches.flatMap((br) => br.items.map((it) => [it, br.revealedAt]))];
    allItems().forEach(([it]) => { if (it.len) it.L = it.el.getTotalLength(); });
    function paint(now) {
      let pending = false;
      for (const [it, at0] of allItems()) {
        let k = 0;
        if (at0 != null) { k = U.clamp((now - at0 - it.at * 1000) / (it.dur * 1000), 0, 1); if (k < 1) pending = true; }
        const e = 1 - Math.pow(1 - k, 3);
        if (it.len) { it.el.style.strokeDasharray = it.L; it.el.style.strokeDashoffset = it.L * (1 - e); }
        else it.el.style.opacity = e;
      }
      return pending;
    }
    let raf = null;
    const run = () => { if (raf) return; const f = (t) => { raf = paint(t) ? requestAnimationFrame(f) : null; if (!raf) highlight(-1); }; raf = requestAnimationFrame(f); };
    return {
      start() {
        const now = performance.now();
        const here = Deck.current(), from = Deck.cameFrom;
        const replay = from == null || from < here;          // coming forwards → play the build-up
        centreAt = replay ? now : now - 1e6;
        let t = now + 900 * TS;                                     // after the centre has faded in
        branches.forEach((br) => {
          br.revealedAt = replay ? t : now - 1e6;
          const last = br.items.reduce((m, it) => Math.max(m, it.at + it.dur), 0);
          t += last * 1000 + 400 * TS;                              // next branch starts when this one is done
        });
        run();
        // "you are here": light up the concept of the slide we just came from
        pinned = null;
        if (from != null) branches.forEach((br, b) => br.leaves.forEach((l, k) => { if (l.target && Deck.findSlide(l.target) === from) pinned = { b, k }; }));
        highlight(-1);
      },
      stop() { if (raf) cancelAnimationFrame(raf); raf = null; paint(performance.now() + 1e7); },
    };
  });
})();

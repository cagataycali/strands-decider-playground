// viz.js -- LANE VIZ. Animates the SVG diagrams space/viz.py renders, after they enter the
// DOM (gr.HTML swaps innerHTML on every yield). Same pattern as static/app.js: the server
// draws the 0-state, this file flips each element to its value. Numbers count up through
// app.js's own [data-sdp="count"] handler -- NOT duplicated here; if app.js is absent the
// numbers are already printed as static text, so nothing is lost.
(function () {
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function flip(n) {
    const kind = n.dataset.viz;
    if (kind === 'bubble' || kind === 'point' || kind === 'mark' || kind === 'cell') n.classList.add('in');
    else if (kind === 'bar') n.setAttribute('width', n.dataset.w);
    else if (kind === 'rect') { n.setAttribute('height', n.dataset.h); n.setAttribute('y', n.dataset.y); }
  }

  function reveal(root) {
    if (!root || !root.querySelectorAll) return;
    const nodes = root.querySelectorAll('[data-viz]:not([data-viz-done])');
    if (!nodes.length) return;
    nodes.forEach(n => n.setAttribute('data-viz-done', '1'));
    // two frames: paint the 0-state, then transition (bubbles 600ms, staggered by data-delay)
    requestAnimationFrame(() => requestAnimationFrame(() => {
      nodes.forEach(n => {
        const d = REDUCED ? 0 : parseInt(n.dataset.delay || '0', 10);
        if (d) setTimeout(() => flip(n), d); else flip(n);
      });
    }));
  }

  const mo = new MutationObserver(muts => {
    for (const m of muts) m.addedNodes.forEach(a => { if (a.nodeType === 1) reveal(a); });
  });
  function start() {
    reveal(document);
    mo.observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();

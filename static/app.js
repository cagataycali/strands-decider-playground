// app.js -- Strands Decider Playground motion + keyboard.
// Loaded via launch(js=...). Server renders static HTML with data-* targets; this file
// only animates them after they enter the DOM (gr.HTML swaps innerHTML on every yield).
// Everything here degrades to "values already visible" when JS is off or motion is reduced.
(function () {
  const REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const DUR = REDUCED ? 0 : 600;

  function fmt(v, dec) {
    if (!isFinite(v)) return '';
    return v.toLocaleString(undefined, { minimumFractionDigits: dec, maximumFractionDigits: dec });
  }

  // numbers count up from 0 to data-count over DUR ms (ease-out cubic)
  function countUp(el) {
    const target = parseFloat(el.dataset.count);
    const dec = parseInt(el.dataset.dec || '0', 10);
    const unit = el.dataset.unit || '';
    if (!isFinite(target) || DUR === 0) { el.textContent = fmt(target, dec) + unit; return; }
    const t0 = performance.now();
    function step(now) {
      const p = Math.min(1, (now - t0) / DUR);
      const e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(target * e, dec) + unit;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  // one pass over a freshly inserted subtree
  function reveal(root) {
    if (!root || !root.querySelectorAll) return;
    const nodes = root.querySelectorAll('[data-sdp]:not([data-sdp-done])');
    if (!nodes.length) return;
    nodes.forEach(n => n.setAttribute('data-sdp-done', '1'));
    // two frames so the browser paints the 0-state first, then transitions to the value
    requestAnimationFrame(() => requestAnimationFrame(() => {
      nodes.forEach((n, i) => {
        const kind = n.dataset.sdp;
        const delay = REDUCED ? 0 : Math.min(i * 18, 240);
        setTimeout(() => {
          if (kind === 'card') n.classList.add('in');
          else if (kind === 'bar') n.style.width = n.dataset.w + '%';
          else if (kind === 'ring') n.style.strokeDashoffset = n.dataset.off;
          else if (kind === 'count') countUp(n);
          else if (kind === 'rect') { if (n.dataset.w) n.setAttribute('width', n.dataset.w); if (n.dataset.h) { n.setAttribute('height', n.dataset.h); n.setAttribute('y', n.dataset.y); } }
          else if (kind === 'cell') n.classList.add('in');
          else if (kind === 'progress') n.style.width = n.dataset.w + '%';
        }, delay);
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

  // keyboard: Enter in the ask bar runs, Esc clears (Shift+Enter keeps a newline in textareas)
  document.addEventListener('keydown', e => {
    const t = e.target;
    const inAsk = t && t.closest && t.closest('#sdp-ask');
    if (e.key === 'Enter' && !e.shiftKey && inAsk && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT') && !t.closest('.wrap-inner')) {
      e.preventDefault();
      const b = document.getElementById('sdp-run');
      if (b) b.click();
    } else if (e.key === 'Escape') {
      const c = document.getElementById('sdp-clear');
      if (c) { c.click(); if (t && t.blur) t.blur(); }
    }
  });
})();

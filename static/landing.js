// landing.js -- landing sections: scroll-to-ask, copy buttons, install tabs, proof count-up.
// Rides launch(js=...) next to app.js (gr.HTML never runs <script>). Everything is delegated on
// `document` in the capture phase, so it works no matter when Gradio inserts the markup and no
// matter which section's HTML the button sits in. Without JS the page still reads: numbers are
// printed, the first tab is open, the links are real anchors.
(function () {
  if (window.__ldLandingLoaded) return;
  window.__ldLandingLoaded = true;
  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var DUR = REDUCED ? 0 : 600;

  function findFirst(selectors) {
    var parts = String(selectors || '').split(',');
    for (var i = 0; i < parts.length; i++) {
      var s = parts[i].trim();
      if (!s) continue;
      var el = null;
      try { el = document.querySelector(s); } catch (e) { el = null; }
      if (el) return el;
    }
    return null;
  }

  // "Ask a question now": scroll the ask bar into view, then focus its textarea.
  function scrollToAsk(selectors) {
    var el = findFirst(selectors);
    if (!el) return false;
    var block = el.closest && (el.closest('#sdp-ask') || el.closest('.block')) || el;
    block.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'start' });
    var focusable = el.matches && el.matches('textarea, input') ? el : el.querySelector && el.querySelector('textarea, input');
    if (focusable) {
      setTimeout(function () { focusable.focus({ preventScroll: true }); }, REDUCED ? 0 : 420);
    }
    return true;
  }

  function copyText(text, btn) {
    function done(ok) {
      if (!btn) return;
      var label = btn.textContent;
      btn.textContent = ok ? 'copied' : 'failed';
      btn.classList.toggle('is-copied', ok);
      setTimeout(function () { btn.textContent = label; btn.classList.remove('is-copied'); }, 1400);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(fallback(text)); });
    } else {
      done(fallback(text));
    }
    function fallback(t) {
      try {
        var ta = document.createElement('textarea');
        ta.value = t; ta.setAttribute('readonly', ''); ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        var ok = document.execCommand('copy');
        document.body.removeChild(ta);
        return ok;
      } catch (e) { return false; }
    }
  }

  function switchTab(btn) {
    var root = btn.closest('[data-ld-tabs]');
    if (!root) return;
    var key = btn.getAttribute('data-ld-tab');
    root.querySelectorAll('[data-ld-tab]').forEach(function (b) {
      var on = b === btn;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    root.querySelectorAll('[data-ld-pane]').forEach(function (p) {
      p.classList.toggle('is-on', p.getAttribute('data-ld-pane') === key);
    });
  }

  document.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    var scroll = t.closest('[data-ld-scroll]');
    if (scroll) {
      if (scrollToAsk(scroll.getAttribute('data-ld-scroll'))) { e.preventDefault(); e.stopPropagation(); }
      return;
    }
    var copy = t.closest('[data-ld-copy]');
    if (copy) { e.preventDefault(); e.stopPropagation(); copyText(copy.getAttribute('data-ld-copy'), copy); return; }
    var tab = t.closest('[data-ld-tab]');
    if (tab) { e.preventDefault(); e.stopPropagation(); switchTab(tab); }
  }, true);

  // proof numbers count up once, when they scroll into view (600 ms ease-out; instant under reduced motion)
  function countUp(el) {
    var target = parseFloat(el.getAttribute('data-ld-count'));
    var suffix = el.getAttribute('data-ld-suffix') || '';
    if (!isFinite(target)) return;
    if (DUR === 0) { el.textContent = String(target) + suffix; return; }
    var t0 = performance.now();
    function step(now) {
      var p = Math.min(1, (now - t0) / DUR);
      var eased = 1 - Math.pow(1 - p, 3);
      el.textContent = String(Math.round(target * eased)) + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }
  var io = ('IntersectionObserver' in window) ? new IntersectionObserver(function (entries) {
    entries.forEach(function (en) {
      if (en.isIntersecting) { io.unobserve(en.target); countUp(en.target); }
    });
  }, { threshold: 0.4 }) : null;
  function arm(root) {
    if (!root || !root.querySelectorAll) return;
    root.querySelectorAll('[data-ld-count]:not([data-ld-armed])').forEach(function (el) {
      el.setAttribute('data-ld-armed', '1');
      if (io) { el.textContent = '0' + (el.getAttribute('data-ld-suffix') || ''); io.observe(el); } else { countUp(el); }
    });
  }
  function start() {
    arm(document);
    new MutationObserver(function (muts) {
      muts.forEach(function (m) { m.addedNodes.forEach(function (n) { if (n.nodeType === 1) arm(n); }); });
    }).observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();

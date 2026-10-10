// bench.js -- client side of the JevBench section: (1) the task picker (search / tier /
// category filter over the JSON the server embedded; a click writes "<task-id>#<nonce>" into the hidden
// Gradio textbox #sdb-pick, same mechanism EXAMPLES measured to work on Gradio 6.30), (2) board tooltips
// + column click -> #sdb-detail textbox, (3) marks newly filled cells so they animate in on a live run.
(function () {
  var REDUCED = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function box(id) { var r = document.querySelector(id); return r ? (r.querySelector('textarea, input') || null) : null; }
  function send(id, value) {
    var b = box(id); if (!b) return false;
    b.value = value + '#' + Date.now().toString(36);
    b.focus({ preventScroll: true });
    b.dispatchEvent(new Event('input', { bubbles: true }));
    ['keydown', 'keypress', 'keyup'].forEach(function (k) { b.dispatchEvent(new KeyboardEvent(k, { bubbles: true, key: 'Unidentified' })); });
    b.blur();
    return true;
  }

  // ---- picker ---------------------------------------------------------------------
  function picker(root) {
    if (!root || root.__sdb) return;
    root.__sdb = true;
    var data = [];
    try { data = JSON.parse(root.querySelector('#sdb-tasks').textContent); } catch (e) { return; }
    var list = root.querySelector('.sdb-list'), search = root.querySelector('.sdb-search'),
        fam = root.querySelector('.sdb-fam'), count = root.querySelector('.sdb-count');
    var tier = '';
    function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]; }); }
    // The same words the question-type control uses (yes / no · pick one · rate); the API names stay in data-task.
    var KIND = { noul: 'yes / no', choice: 'pick one', score: 'rate' };
    function kind(t) { return esc(KIND[t.type] || t.type) + (t.type !== 'noul' ? ' ×' + t.n : ''); }
    function render() {
      var q = (search.value || '').toLowerCase().trim(), f = fam.value;
      var rows = data.filter(function (t) {
        if (tier && t.tier !== tier) return false;
        if (f && t.family !== f) return false;
        if (q && (t.id + ' ' + t.q + ' ' + t.state).toLowerCase().indexOf(q) < 0) return false;
        return true;
      });
      count.textContent = rows.length + ' of ' + data.length;
      list.innerHTML = rows.slice(0, 120).map(function (t) {
        return '<div class="sdb-item" role="option" tabindex="0" data-task="' + esc(t.id) + '">' +
          '<div class="id"><b>' + esc(t.id) + '</b><span>' + esc(t.tier) + ' · ' + esc(t.family) + ' · ' + kind(t) + '</span></div>' +
          '<div class="q">' + esc(t.q) + '</div><div class="st">' + esc(t.state) + '</div></div>';
      }).join('') + (rows.length > 120 ? '<div class="sdb-dim">… ' + (rows.length - 120) + ' more — narrow the search</div>' : '');
    }
    root.querySelectorAll('.sdb-f').forEach(function (b) {
      b.addEventListener('click', function () {
        root.querySelectorAll('.sdb-f').forEach(function (x) { x.classList.remove('on'); });
        b.classList.add('on'); tier = b.getAttribute('data-tier') || ''; render();
      });
    });
    search.addEventListener('input', render); fam.addEventListener('change', render);
    function choose(el) {
      var id = el.getAttribute('data-task'); if (!id) return;
      list.querySelectorAll('.sdb-item.loading').forEach(function (x) { x.classList.remove('loading'); });
      el.classList.add('loading');
      if (send('#sdb-pick', id)) {
        var ask = document.querySelector('#sdp-ask'); if (ask && ask.scrollIntoView) ask.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'start' });
      }
      setTimeout(function () { el.classList.remove('loading'); }, 1500);
    }
    list.addEventListener('click', function (ev) { var el = ev.target.closest('.sdb-item'); if (el) choose(el); });
    list.addEventListener('keydown', function (ev) { if (ev.key === 'Enter' || ev.key === ' ') { var el = ev.target.closest('.sdb-item'); if (el) { ev.preventDefault(); choose(el); } } });
    render();
  }

  // ---- board ----------------------------------------------------------------------
  var seen = {};
  function board(wrap) {
    if (!wrap) return;
    var tip = wrap.querySelector('.sdb-tip'), svg = wrap.querySelector('svg.sdb-board');
    if (!svg) return;
    // animate newly filled cells on a live board (the server re-renders the whole SVG on each poll)
    if (wrap.getAttribute('data-live') === '1') {
      svg.querySelectorAll('.sdb-live .sdb-c:not(.sdb-none)').forEach(function (c) {
        var k = c.getAttribute('data-row') + '|' + c.getAttribute('data-task');
        if (!seen[k]) { seen[k] = 1; if (!REDUCED) c.classList.add('sdb-new'); }
      });
    }
    if (svg.__sdb) return; svg.__sdb = true;
    function show(ev, el) {
      var t = el.getAttribute('data-tip'); if (!t || !tip) return;
      var p = t.split('|');
      if (p.length >= 5) {
        var ok = el.classList.contains('sdb-ok');
        tip.innerHTML = '<b>' + p[0] + '</b> · ' + p[1] + '<br>answered <b class="' + (ok ? 'ok' : '') + '">' + p[2] + '</b> · gold <b>' + p[3] + '</b>' +
          '<br>p(gold) ' + p[4] + (p[5] ? ' · ' + p[5] + ' ms' : '') + (p[6] ? '<br>' + p[6] : '');
      } else tip.innerHTML = '<b>' + p[0] + '</b>' + (p[1] ? ' · ' + p[1] : '');
      tip.hidden = false;
      var x = ev.clientX + 12, y = ev.clientY + 12;
      if (x + 340 > window.innerWidth) x = ev.clientX - 350;
      if (y + 90 > window.innerHeight) y = ev.clientY - 90;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    }
    svg.addEventListener('mousemove', function (ev) { var el = ev.target.closest('[data-tip]'); if (el) show(ev, el); else if (tip) tip.hidden = true; });
    svg.addEventListener('mouseleave', function () { if (tip) tip.hidden = true; });
    svg.addEventListener('click', function (ev) {
      var el = ev.target.closest('[data-task]'); if (!el) return;
      var id = el.getAttribute('data-task'); if (id) send('#sdb-detail', id);
      var d = document.querySelector('#sdb-detail-out'); if (d && d.scrollIntoView) d.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'nearest' });
    });
    svg.addEventListener('keydown', function (ev) { if (ev.key === 'Enter') { var el = ev.target.closest('[data-task]'); if (el) send('#sdb-detail', el.getAttribute('data-task')); } });
    // count-up for the right-margin totals (fresh render only)
    if (!REDUCED) svg.querySelectorAll('.sdb-num[data-count]').forEach(function (n) {
      var target = parseInt(n.getAttribute('data-count'), 10) || 0, t0 = performance.now();
      function step(now) { var k = Math.min(1, (now - t0) / 600); n.textContent = Math.round(target * (1 - Math.pow(1 - k, 3))); if (k < 1) requestAnimationFrame(step); }
      requestAnimationFrame(step);
    });
  }
  // task links inside the error panels
  document.addEventListener('click', function (ev) { var el = ev.target.closest('.sdb-tline[data-task]'); if (el) send('#sdb-detail', el.getAttribute('data-task')); });

  function scan() { picker(document.querySelector('#sdb-picker')); document.querySelectorAll('.sdb-wrap').forEach(board); }
  var mo = new MutationObserver(function () { scan(); });
  function start() { scan(); mo.observe(document.body, { childList: true, subtree: true }); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start); else start();
})();

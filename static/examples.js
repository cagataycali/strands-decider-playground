// examples.js -- LANE EXAMPLES. Makes the server-rendered gallery clickable + keyboard reachable.
// A click (or Enter/Space) on a card writes its example id into the hidden Gradio textbox
// #sdx-pick and fires an input event; app.py's `# --- LANE EXAMPLES ---` region listens on that
// textbox, fills the ask bar (state, question type, question, options, image, models) and runs.
// Standalone (--demo) there is no textbox: the click just marks the card.
(function () {
  function pickBox() {
    var root = document.querySelector('#sdx-pick');
    return root ? (root.querySelector('textarea, input') || null) : null;
  }
  function choose(card) {
    var id = card.getAttribute('data-example');
    if (!id) return;
    document.querySelectorAll('.sdx-card.loading').forEach(function (c) { c.classList.remove('loading'); });
    card.classList.add('loading');
    var box = pickBox();
    if (box) {
      // Gradio reads the component value from the DOM on 'input'; a nonce makes re-clicking the
      // same example fire again.
      box.value = id + '#' + Date.now().toString(36);
      // MEASURED (Gradio 6.30, Chromium): a bare 'input' Event does NOT reach the Python .input()
      // handler; real typing does. Gradio's Textbox reads the value on key events, so a synthetic
      // keydown/keypress/keyup after the input event is what makes the server see the new value.
      box.focus({ preventScroll: true });
      box.dispatchEvent(new Event('input', { bubbles: true }));
      ['keydown', 'keypress', 'keyup'].forEach(function (k) {
        box.dispatchEvent(new KeyboardEvent(k, { bubbles: true, key: 'Unidentified' }));
      });
      box.blur();
      var ask = document.querySelector('#sdp-ask');
      if (ask && ask.scrollIntoView) ask.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    setTimeout(function () { card.classList.remove('loading'); }, 1500);
  }
  function wire(root) {
    root.querySelectorAll('.sdx-card').forEach(function (card) {
      if (card.__sdx) return;
      card.__sdx = true;
      card.addEventListener('click', function () { choose(card); });
      card.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); choose(card); }
      });
    });
    reveal(root);
  }
  function reveal(root) {
    var cards = root.querySelectorAll('.sdx-card:not(.in)');
    if (!('IntersectionObserver' in window)) { cards.forEach(function (c) { c.classList.add('in'); }); return; }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
      });
    }, { rootMargin: '0px 0px -5% 0px' });
    cards.forEach(function (c, i) { c.style.transitionDelay = (Math.min(i, 8) * 40) + 'ms'; io.observe(c); });
  }
  function boot() {
    wire(document);
    // Gradio re-renders gr.HTML on update; keep new cards wired.
    var mo = new MutationObserver(function () { wire(document); });
    mo.observe(document.body, { childList: true, subtree: true });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();

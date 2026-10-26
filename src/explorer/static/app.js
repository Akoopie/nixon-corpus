// Nixon Tapes Explorer: a hash-routed single page over the local server's JSON.
const view = document.getElementById("view");
const audio = document.getElementById("audio");
const player = document.getElementById("player");
const follow = document.getElementById("follow");
const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const PLACES = { OVAL: "Oval Office", EOB: "EOB office", WHT: "Telephone", CAB: "Cabinet Room", CDHW: "Camp David (Aspen)", CDST: "Camp David (study)", CDSD: "Camp David (desk)" };
const SPEAKER_COLOR = { Nixon: "--s1", Haldeman: "--s2", Kissinger: "--s3", Ehrlichman: "--s4", Haig: "--s5", Colson: "--s6", Ziegler: "--s7", Connally: "--s8" };

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmtDate = (d) => { const [y, m, dd] = d.split("-").map(Number); const wd = new Date(Date.UTC(y, m - 1, dd)).toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" }); return `${wd}, ${MONTHS[m - 1]} ${dd}, ${y}`; };
const shortDate = (d) => { const [y, m, dd] = d.split("-").map(Number); return `${MONTHS[m - 1].slice(0, 3)} ${dd}, ${y}`; };
const mmss = (s) => { s = Math.max(0, Math.floor(s)); const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), x = s % 60; return (h ? h + ":" + String(m).padStart(2, "0") : m) + ":" + String(x).padStart(2, "0"); };
const hrs = (min) => min >= 60 ? (min / 60).toFixed(1) + " h" : Math.round(min) + " min";
const place = (loc) => `<span class="place" title="${esc(PLACES[loc] || loc)}">${esc(loc)}</span>`;
async function get(url) { const r = await fetch(url); if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText); return r.json(); }

function speakerName(sp) {
  if (!sp || sp === "other") return { label: "?", base: "" };
  const m = sp.match(/^Unknown \(possibly (.+)\)$/);
  if (m) return { label: m[1] + "?", base: m[1], unsure: true };
  if (sp.startsWith("Unknown")) return { label: "?", base: "" };
  return { label: sp, base: sp };
}
const speakerColor = (base) => (SPEAKER_COLOR[base] ? `var(${SPEAKER_COLOR[base]})` : "var(--ink-3)");

// --- routing ---------------------------------------------------------------------------------------------------
async function route() {
  const [path, query] = location.hash.slice(1).split("?");
  const params = new URLSearchParams(query || "");
  const parts = (path || "/").split("/").filter(Boolean);
  window.scrollTo(0, 0);
  view.innerHTML = `<p class="loading">Loading…</p>`;
  try {
    if (!parts.length) await home();
    else if (parts[0] === "day") await day(parts[1]);
    else if (parts[0] === "conv") await conv(parts[1], params);
    else if (parts[0] === "person") await person(parts[1]);
    else if (parts[0] === "people") await people(params.get("q") || "");
    else if (parts[0] === "search") await search(params.get("q") || "");
    else await home();
  } catch (e) { view.innerHTML = `<p>Could not load this page: ${esc(e.message)}</p>`; }
}
window.addEventListener("hashchange", route);
document.getElementById("search").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = document.getElementById("q").value.trim();
  if (q) location.hash = `#/search?q=${encodeURIComponent(q)}`;
});

// --- home: the calendar -----------------------------------------------------------------------------------------
async function home() {
  const days = await get("/api/calendar");
  const by = Object.fromEntries(days.map((d) => [d.date, d]));
  const tot = days.reduce((a, d) => ({ n: a.n + d.n, m: a.m + d.minutes, w: a.w + d.words }), { n: 0, m: 0, w: 0 });
  const level = (m) => (m <= 0 ? "" : m < 60 ? "l1" : m < 180 ? "l2" : m < 360 ? "l3" : m < 600 ? "l4" : "l5");
  let html = `<section class="stack"><div class="eyebrow">White House tapes, February 16, 1971 – July 12, 1973</div>
    <h1>Pick a day</h1>
    <div class="stats"><div><b class="num">${days.length}</b><span>days with tape</span></div><div><b class="num">${tot.n.toLocaleString()}</b><span>conversations</span></div>
    <div><b class="num">${Math.round(tot.m / 60).toLocaleString()}</b><span>hours of audio</span></div><div><b class="num">${(tot.w / 1e6).toFixed(1)}M</b><span>transcribed words</span></div></div>
    <div class="legend">hours of tape per day: <span class="cell l1"></span>&lt;1 <span class="cell l2"></span>1–3 <span class="cell l3"></span>3–6 <span class="cell l4"></span>6–10 <span class="cell l5"></span>10+</div></section>`;
  for (const y of [1971, 1972, 1973]) {
    html += `<section class="year"><h2>${y}</h2><div class="months">`;
    for (let m = 0; m < 12; m++) {
      if ((y === 1971 && m < 1) || (y === 1973 && m > 6)) continue;
      const first = new Date(Date.UTC(y, m, 1)).getUTCDay(), n = new Date(Date.UTC(y, m + 1, 0)).getUTCDate();
      let cells = "SMTWTFS".split("").map((d) => `<span class="dow">${d}</span>`).join("") + "<span></span>".repeat(first);
      for (let d = 1; d <= n; d++) {
        const key = `${y}-${String(m + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`, info = by[key];
        cells += info ? `<a class="cell ${level(info.minutes)}" href="#/day/${key}" title="${fmtDate(key)}: ${info.n} conversations, ${hrs(info.minutes)}">${d}</a>`
                      : `<span class="cell">${d}</span>`;
      }
      html += `<div class="month"><h3>${MONTHS[m]}</h3><div class="grid7">${cells}</div></div>`;
    }
    html += `</div></section>`;
  }
  view.innerHTML = html;
}

// --- a day ------------------------------------------------------------------------------------------------------
async function day(date) {
  const d = await get(`/api/day/${date}`);
  const maxMin = Math.max(1, ...d.conversations.map((c) => c.minutes));
  const list = d.conversations.map((c) => `<div class="conv">
      <div class="when">${esc(c.start || "?")}<br><span class="muted">${esc(c.end || "")}</span></div>
      <div class="what">
        <div class="head"><a href="#/conv/${c.id}"><b class="mono">${c.id}</b></a> ${place(c.loc)}
          <span class="muted num">${hrs(c.minutes)}</span>${c.layer === "corrected" ? ` <span class="argonbadge">Argon</span>` : ""}</div>
        <div class="bar-min" style="width:${Math.max(2, (100 * c.minutes) / maxMin)}%"></div>
        <div class="chips">${c.people.slice(0, 8).map((p) => `<a class="chip" href="#/person/${p.id}">${esc(p.name)}</a>`).join("")}${c.people.length > 8 ? `<span class="muted">+${c.people.length - 8}</span>` : ""}</div>
        ${c.topics.length ? `<div class="topics">${c.topics.map(esc).join(" · ")}</div>` : ""}
      </div></div>`).join("");
  view.innerHTML = `
    <div class="pager">${d.prev ? `<a href="#/day/${d.prev}">← ${shortDate(d.prev)}</a>` : "<span></span>"}${d.next ? `<a href="#/day/${d.next}">${shortDate(d.next)} →</a>` : "<span></span>"}</div>
    <section><div class="eyebrow">A day on the tapes</div><h1>${fmtDate(date)}</h1></section>
    <div class="stats"><div><b class="num">${d.day?.n ?? 0}</b><span>conversations</span></div><div><b class="num">${hrs(d.day?.minutes ?? 0)}</b><span>of audio</span></div>
      <div><b class="num">${(d.day?.words ?? 0).toLocaleString()}</b><span>words</span></div></div>
    <div class="cols"><section><h2>Conversations</h2><div class="convlist">${list}</div></section>
      <aside class="card"><h3>Haldeman's diary</h3><p class="muted" style="font-size:.8rem">H. R. Haldeman's dictated diary for this day and the next (NARA). He was not at every meeting.</p>
      <div class="diary">${esc(d.day?.diary || "No diary entry (the diary ends April 30, 1973).")}</div></aside></div>`;
}

// --- a conversation: transcript synced to the audio --------------------------------------------------------------
let words = [], spans = [], starts = [], current = -1, raf = 0;

async function conv(id, params) {
  const c = await get(`/api/conv/${id}`);
  words = c.words; starts = words.map((w) => w.s); current = -1;
  const parts = c.people.filter((p) => p.participant);
  const others = c.people.filter((p) => !p.participant && (p.speak_s > 30 || p.mentions));
  const peopleChips = parts.map((p) => `<a class="chip" href="#/person/${p.id}">${esc(p.name)}${p.speak_s > 30 ? `<small>${hrs(p.speak_s / 60)} speaking</small>` : ""}</a>`).join("");
  const mentioned = others.map((p) => `<a class="chip" href="#/person/${p.id}">${esc(p.name)}<small>${p.speak_s > 30 ? hrs(p.speak_s / 60) + " speaking" : p.mentions + "× named"}</small></a>`).join("");
  const outline = c.topics.length ? `<ul class="outline">${c.topics.map((t) => `<li class="${t.dashed ? "" : "top"}" style="padding-left:${t.dashed ? 0.9 + 0.8 * Math.max(0, t.level - 1) : 0}rem">${esc(t.text)}</li>`).join("")}</ul>` : `<p class="muted">NARA's subject log has no outline for this conversation.</p>`;
  const tr = c.transcript || {};
  view.innerHTML = `
    <section><div class="eyebrow">Conversation ${esc(c.id)} · <a href="#/day/${c.date}">${fmtDate(c.date)}</a></div>
      <h1>${esc(PLACES[c.loc] || c.place)}, ${esc(c.start || "time unknown")}${c.end ? "–" + esc(c.end) : ""}</h1></section>
    <div class="stats"><div><b class="num">${hrs(c.minutes)}</b><span>of audio</span></div><div><b class="num">${words.length.toLocaleString()}</b><span>words</span></div>
      <div><b class="num">${tr.layer === "argon" ? tr.edits : "–"}</b><span>${tr.layer === "argon" ? "Argon fixes" : "not yet corrected by Argon"}</span></div>
      ${tr.unclear ? `<div><b class="num">${tr.unclear}</b><span>stretches marked unclear</span></div>` : ""}
      <div><b class="num">${words.filter((w) => w.fresh).length.toLocaleString()}</b><span>words in no earlier transcript</span></div></div>
    ${c.prior.length ? `<p class="muted" style="font-size:.85rem">Already transcribed elsewhere: ${c.prior.map(esc).join("; ")}.</p>`
      : !c.easynixon ? `<p class="muted" style="font-size:.85rem">Easy Nixon has no transcript of this conversation.</p>` : ""}
    <div class="cols">
      <section class="card"><h3>Transcript</h3>
        <div class="keys"><span><span class="w ar">orange underline</span> fixed by Argon (hover for why)</span><span><span class="w ar rep">dashed</span> name fix repeated</span>
          <span><span class="w unsure">faded</span> recognizer unsure</span><span><span class="w wh">dotted</span> heard only by Whisper</span><span><span class="w unclear">[unclear]</span> too garbled to keep</span>
          <span><span class="w fresh">green</span> in no earlier transcript (Easy Nixon, WSPF, NARA, FRUS)</span>
          <span>Click a word to play from there.</span></div>
        <div class="transcript" id="tx"></div></section>
      <aside class="stack" style="display:flex;flex-direction:column;gap:1rem">
        <div class="card"><h3>People</h3><div class="chips">${peopleChips || '<span class="muted">none listed</span>'}</div>
          ${mentioned ? `<p class="eyebrow" style="margin-top:.8rem">Also heard or named</p><div class="chips">${mentioned}</div>` : ""}</div>
        <div class="card"><h3>NARA subject log</h3>${outline}</div>
        ${c.summary ? `<div class="card"><h3>NARA record</h3><p style="font-size:.9rem">${esc(c.summary)}</p></div>` : ""}
        ${c.alt_source ? `<p class="muted" style="font-size:.8rem">The audio here comes from an alternate copy; timings may drift.</p>` : ""}
      </aside></div>`;
  renderTranscript(document.getElementById("tx"), params.get("q"));
  if (c.audio.length) {
    player.hidden = false;
    document.getElementById("pl-title").textContent = `${c.id} · ${shortDate(c.date)} · ${PLACES[c.loc] || c.place}`;
    if (!audio.src.endsWith(`/audio/${c.id}`)) { audio.src = `/audio/${c.id}`; }
    const t = parseFloat(params.get("t"));
    if (!isNaN(t)) { audio.currentTime = t; }
  } else { player.hidden = true; }
}

function renderTranscript(el, query) {
  const terms = (query || "").toLowerCase().match(/[a-z0-9']+/g) || [];
  let html = "", turn = null, first = -1;
  const close = () => { if (turn) html += `</div></div>`; };
  words.forEach((w, i) => {
    const prev = words[i - 1];
    if (!turn || w.sp !== turn.sp || (prev && w.s - prev.e > 4)) {
      close();
      const n = speakerName(w.sp);
      turn = { sp: w.sp };
      html += `<div class="turn"><div class="who" style="color:${speakerColor(n.base)}">${esc(n.label)}<span class="t">${mmss(w.s)}</span></div><div>`;
    }
    const cls = ["w"];
    if (w.argon) cls.push(w.t === "[unclear]" ? "unclear" : "ar", w.argon.repeated ? "rep" : "");
    if (w.unsure) cls.push("unsure");
    if (w.whisper) cls.push("wh");
    if (w.fresh) cls.push("fresh");
    let text = esc(w.t);
    if (terms.length && terms.some((t) => w.t.toLowerCase().replace(/[^a-z0-9']/g, "") === t)) { text = `<mark>${text}</mark>`; if (first < 0) first = i; }
    html += `<span class="${cls.join(" ").trim()}" data-i="${i}">${text}</span> `;
  });
  close();
  el.innerHTML = html || `<p class="muted">No transcript.</p>`;
  spans = Array.from(el.querySelectorAll(".w"));
  if (first >= 0) spans[first].scrollIntoView({ block: "center" });
}

function seekTo(i) { if (!audio.src) return; audio.currentTime = words[i].s; audio.play().catch(() => {}); }
function wordAt(t) { let lo = 0, hi = starts.length - 1, r = -1; while (lo <= hi) { const m = (lo + hi) >> 1; if (starts[m] <= t) { r = m; lo = m + 1; } else hi = m - 1; } return r; }
function sync() {
  const i = wordAt(audio.currentTime);
  if (i !== current) {
    if (current >= 0 && spans[current]) spans[current].classList.remove("now");
    if (i >= 0 && spans[i]) {
      spans[i].classList.add("now");
      if (follow.checked) { const r = spans[i].getBoundingClientRect(); if (r.top < 80 || r.bottom > innerHeight - 110) spans[i].scrollIntoView({ block: "center" }); }
    }
    current = i;
  }
  if (!audio.paused) raf = requestAnimationFrame(sync);
}
audio.addEventListener("play", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(sync); });
audio.addEventListener("seeked", sync);
audio.addEventListener("waiting", () => { document.getElementById("pl-title").dataset.wait = "1"; });

const tip = document.createElement("div"); tip.className = "tip"; tip.hidden = true; document.body.appendChild(tip);
view.addEventListener("click", (e) => { const s = e.target.closest(".w[data-i]"); if (s) seekTo(+s.dataset.i); });
view.addEventListener("mouseover", (e) => {
  const s = e.target.closest(".w[data-i]");
  const w = s && words[+s.dataset.i];
  if (!w || !w.argon) { tip.hidden = true; return; }
  tip.innerHTML = `<div>was: <b>${esc(w.argon.was)}</b></div>${w.argon.why ? `<div>${esc(w.argon.why)}</div>` : ""}${w.argon.repeated ? "<div>(repeated from Argon's fix elsewhere in this tape)</div>" : ""}`;
  const r = s.getBoundingClientRect(); tip.style.left = Math.min(r.left, innerWidth - 420) + "px"; tip.style.top = r.bottom + 6 + "px"; tip.hidden = false;
});
view.addEventListener("mouseout", (e) => { if (e.target.closest(".w")) tip.hidden = true; });

// --- a person -----------------------------------------------------------------------------------------------------
async function person(id) {
  const d = await get(`/api/person/${id}`);
  const p = d.person;
  const months = []; for (let y = 1971; y <= 1973; y++) for (let m = 1; m <= 12; m++) { if ((y === 1971 && m < 2) || (y === 1973 && m > 7)) continue; months.push(`${y}-${String(m).padStart(2, "0")}`); }
  const part = {}, ment = {};
  d.conversations.forEach((c) => { const k = c.date.slice(0, 7); if (c.participant) part[k] = (part[k] || 0) + 1; else if (c.mentions || c.speak_s > 30) ment[k] = (ment[k] || 0) + 1; });
  const mx = Math.max(1, ...months.map((k) => (part[k] || 0) + (ment[k] || 0)));
  const spark = months.map((k) => `<span style="flex:1;display:flex;flex-direction:column-reverse;height:100%" title="${k}: ${part[k] || 0} as participant, ${ment[k] || 0} heard or named"><i style="height:${(100 * (part[k] || 0)) / mx}%"></i><i class="m" style="height:${(100 * (ment[k] || 0)) / mx}%"></i></span>`).join("");
  const topicRows = d.topics.map((t, k) => `<tr class="topicrow" data-k="${k}"><td>${esc(t.topic)}</td><td class="num">${t.n}</td><td class="num">${shortDate(t.first)}</td><td class="num">${shortDate(t.last)}</td></tr>`).join("");
  view.innerHTML = `
    <section><div class="eyebrow">Person</div><h1>${esc(p.name)}</h1></section>
    <div class="stats"><div><b class="num">${p.n_part.toLocaleString()}</b><span>conversations NARA lists them in</span></div>
      <div><b class="num">${hrs(p.speak_min)}</b><span>of their voice (classifier, ${p.n_speak} conversations)</span></div>
      <div><b class="num">${p.n_mention_convs.toLocaleString()}</b><span>conversations that say “${esc(p.surname || p.name)}”</span></div></div>
    <section class="card"><h3>By month</h3><div class="spark">${spark}</div><div class="sparkaxis"><span>Feb 1971</span><span>1972</span><span>Jul 1973</span></div>
      <div class="legend"><span class="cell l3"></span> listed as participant <span class="cell l2"></span> only heard or named</div></section>
    <div class="cols">
      <section class="card"><h3>Topics of their conversations</h3><p class="muted" style="font-size:.8rem">From NARA's subject logs of the conversations they are listed in. Click a topic to list those conversations.</p>
        <div class="tablewrap"><table><thead><tr><th>Topic</th><th>Conversations</th><th>First</th><th>Last</th></tr></thead><tbody id="topics">${topicRows || '<tr><td colspan="4" class="muted">No subject-log topics.</td></tr>'}</tbody></table></div></section>
      <aside class="card"><h3>Most often with</h3><div class="chips">${d.company.map((c) => `<a class="chip" href="#/person/${c.id}">${esc(c.name)}<small>${c.n}</small></a>`).join("")}</div></aside></div>
    <section class="card"><h3 id="clist-title">Conversations</h3><div class="tablewrap"><table><thead><tr><th>Date</th><th>Time</th><th>Place</th><th>Length</th><th>Role</th><th></th></tr></thead><tbody id="clist"></tbody></table></div></section>`;
  const fill = (rows, title) => {
    document.getElementById("clist-title").textContent = title;
    document.getElementById("clist").innerHTML = rows.slice(0, 600).map((c) => `<tr><td class="num"><a href="#/day/${c.date}">${shortDate(c.date)}</a></td><td class="num">${esc(c.start || "")}</td><td>${place(c.loc)}</td>
      <td class="num">${hrs(c.minutes)}</td><td>${[c.participant ? "participant" : "", c.speak_s > 30 ? hrs(c.speak_s / 60) + " speaking" : "", c.mentions ? c.mentions + "× named" : ""].filter(Boolean).join(", ")}</td>
      <td><a href="#/conv/${c.id}${c.mentions && !c.participant ? "?q=" + encodeURIComponent((p.surname || "").toLowerCase()) : ""}">${c.id}</a>${c.layer === "corrected" ? ' <span class="argonbadge">Argon</span>' : ""}</td></tr>`).join("");
  };
  fill(d.conversations, `Conversations (${d.conversations.length.toLocaleString()})`);
  document.getElementById("topics").addEventListener("click", (e) => {
    const row = e.target.closest(".topicrow"); if (!row) return;
    const t = d.topics[+row.dataset.k], on = row.classList.contains("on");
    document.querySelectorAll(".topicrow").forEach((r) => r.classList.remove("on"));
    if (on) return fill(d.conversations, `Conversations (${d.conversations.length.toLocaleString()})`);
    row.classList.add("on");
    const set = new Set(t.convs);
    fill(d.conversations.filter((c) => set.has(c.id)), `Conversations on “${t.topic}” (${set.size})`);
    document.getElementById("clist-title").scrollIntoView({ block: "start", behavior: "smooth" });
  });
}

// --- people and search ----------------------------------------------------------------------------------------------
async function people(q) {
  const rows = await get(`/api/people${q ? "?q=" + encodeURIComponent(q) : ""}`);
  view.innerHTML = `<section><div class="eyebrow">People</div><h1>Who is on the tapes</h1></section>
    <form id="pf"><input id="pq" type="search" value="${esc(q)}" placeholder="Filter by name" aria-label="Filter people by name" style="padding:.45rem .7rem;border:1px solid var(--rule);border-radius:4px;background:var(--card);color:var(--ink);font:inherit;width:min(100%,24rem)"></form>
    <div class="tablewrap card"><table><thead><tr><th>Name</th><th>Listed in</th><th>Voice</th><th>Named in</th></tr></thead><tbody>
    ${rows.map((p) => `<tr><td><a href="#/person/${p.id}">${esc(p.name)}</a></td><td class="num">${p.n_part.toLocaleString()}</td><td class="num">${p.speak_min ? hrs(p.speak_min) : "–"}</td><td class="num">${p.n_mention_convs.toLocaleString()}</td></tr>`).join("")}
    </tbody></table></div>`;
  document.getElementById("pf").addEventListener("submit", (e) => { e.preventDefault(); location.hash = `#/people?q=${encodeURIComponent(document.getElementById("pq").value.trim())}`; });
}

async function search(q) {
  document.getElementById("q").value = q;
  const rows = await get(`/api/search?q=${encodeURIComponent(q)}`);
  view.innerHTML = `<section><div class="eyebrow">Search</div><h1>“${esc(q)}”</h1><p class="muted">${rows.length >= 120 ? "First 120" : rows.length} conversations, best matches first. Transcripts are machine-made: try sound-alike spellings too.</p></section>
    <div class="convlist">${rows.map((r) => `<div class="conv"><div class="when"><a href="#/day/${r.date}">${shortDate(r.date)}</a><br><span class="muted">${esc(r.start || "")}</span></div>
      <div class="what"><div class="head"><a href="#/conv/${r.id}?q=${encodeURIComponent(q)}"><b class="mono">${r.id}</b></a> ${place(r.place)} <span class="muted num">${hrs(r.minutes)}</span>${r.layer === "corrected" ? ' <span class="argonbadge">Argon</span>' : ""}</div>
      <div class="topics">${r.snippet}</div></div></div>`).join("")}</div>`;
}

route();

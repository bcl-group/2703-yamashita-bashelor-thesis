"use strict";

// ---------- 共通 ----------

const $ = (sel) => document.querySelector(sel);

/** 要素を作る。文字列は必ず textContent で入れる（innerHTML は使わない）。 */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") node.textContent = v;
    else if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) node.append(c);
  return node;
}

async function api(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (res.status === 204) return null;
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

/** Python 側の normalize と同じ処理（表示用）。 */
function normalize(text) {
  return text.trim().toLowerCase().replace(/^[^a-z]+|[^a-z]+$/g, "").replace(/\s+/g, " ");
}

let POS_LIST = [];

// ---------- タブ ----------

function showTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (name === "register") $("#word").focus();
  if (name === "list") loadList();
  if (name === "quiz") nextQuiz();
}

document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- 登録 ----------

const reg = {
  lemma: null,      // 編集中の見出し語
  query: "",        // 検索したときの入力
  senses: [],       // [{pos, meanings: []}]
  wordnet: {},      // {品詞: [訳]}
  ejdict: [],       // [訳]
  saved: false,
};

async function lookup() {
  const text = $("#word").value;
  if (!normalize(text)) {
    $("#lookup-note").textContent = "英単語を入力してください";
    return;
  }
  let r;
  try {
    r = await api("GET", `/api/lookup?word=${encodeURIComponent(text)}`);
  } catch (e) {
    $("#lookup-note").textContent = e.message;
    return;
  }
  reg.lemma = r.lemma;
  reg.query = text;
  reg.wordnet = r.wordnet;
  reg.ejdict = r.ejdict;
  reg.saved = Boolean(r.saved);
  reg.senses = structuredClone(r.saved ? r.saved.senses : r.default);
  const hasTranslation = Object.keys(r.wordnet).length > 0 || r.ejdict.length > 0;
  if (reg.senses.length === 0) reg.senses = [{ pos: POS_LIST[0], meanings: [] }];

  const notes = [];
  if (r.lemma !== normalize(text)) notes.push(`${r.lemma} として検索`);
  if (!hasTranslation) notes.push("辞書に訳が見つかりません。品詞と意味を入力してください");
  else if (reg.saved) notes.push("登録済みの内容を表示しています");
  $("#lookup-note").textContent = notes.join("　/　");

  $("#example").value = r.saved ? r.saved.example : "";
  $("#editor").hidden = false;
  renderEditor();
  if (!hasTranslation) document.querySelector(".sense .add")?.focus();
}

function senseFor(pos) {
  let s = reg.senses.find((x) => x.pos === pos);
  if (!s) {
    s = { pos, meanings: [] };
    reg.senses.push(s);
  }
  return s;
}

function addMeaning(pos, meaning) {
  const s = senseFor(pos);
  if (!s.meanings.includes(meaning)) s.meanings.push(meaning);
  renderEditor();
}

function renderEditor() {
  $("#lemma").textContent = reg.lemma;
  $("#saved-badge").hidden = !reg.saved;
  $("#delete").hidden = !reg.saved;

  const box = $("#senses");
  box.replaceChildren();
  reg.senses.forEach((s, i) => {
    const select = el("select", {
      onchange: (e) => {
        // 同じ品詞の行があればまとめる
        const other = reg.senses.find((x, j) => j !== i && x.pos === e.target.value);
        if (other) {
          for (const m of s.meanings) if (!other.meanings.includes(m)) other.meanings.push(m);
          reg.senses.splice(i, 1);
        } else {
          s.pos = e.target.value;
        }
        renderEditor();
      },
    }, POS_LIST.map((p) => el("option", { value: p, text: p })));
    select.value = s.pos;

    const chips = s.meanings.map((m, k) =>
      el("span", { class: "chip" }, [
        m,
        el("button", {
          type: "button", title: "削除", text: "×",
          onclick: () => { s.meanings.splice(k, 1); renderEditor(); },
        }),
      ]));

    const add = el("input", {
      type: "text", class: "add", placeholder: "意味を追加して Enter",
      onkeydown: (e) => {
        if (e.key !== "Enter" || e.isComposing) return;
        e.preventDefault();
        const v = e.target.value.trim();
        if (v) {
          if (!s.meanings.includes(v)) s.meanings.push(v);
          renderEditor();
          box.querySelectorAll(".add")[i]?.focus();
        } else {
          save();
        }
      },
    });

    const remove = el("button", {
      type: "button", class: "remove", title: "この品詞を削除", text: "✕",
      onclick: () => { reg.senses.splice(i, 1); renderEditor(); },
    });

    box.append(el("div", { class: "sense" }, [select, ...chips, add, remove]));
  });

  renderCandidates();
}

function renderCandidates() {
  // WordNet の訳のうち、まだ保存内容に入っていないもの
  const wnBody = $("#cand-wordnet .cand-body");
  wnBody.replaceChildren();
  for (const [pos, list] of Object.entries(reg.wordnet)) {
    const current = reg.senses.find((s) => s.pos === pos)?.meanings ?? [];
    const rest = list.filter((m) => !current.includes(m));
    if (rest.length === 0) continue;
    wnBody.append(el("div", { class: "cand-group" }, [
      el("span", { class: "pos", text: pos }),
      ...rest.map((m) => el("button", { type: "button", class: "cand", text: m, onclick: () => addMeaning(pos, m) })),
    ]));
  }
  $("#cand-wordnet").hidden = wnBody.childElementCount === 0;

  const all = reg.senses.flatMap((s) => s.meanings);
  const ejBody = $("#cand-ejdict .cand-body");
  ejBody.replaceChildren(...reg.ejdict
    .filter((m) => !all.includes(m))
    .map((m) => el("button", { type: "button", class: "cand", text: m, onclick: (e) => openPosMenu(e.currentTarget, m) })));
  $("#cand-ejdict").hidden = ejBody.childElementCount === 0;
}

function openPosMenu(anchor, meaning) {
  const menu = $("#pos-menu");
  menu.replaceChildren(...POS_LIST.map((p) =>
    el("button", { type: "button", text: p, onclick: () => { closePosMenu(); addMeaning(p, meaning); } })));
  const rect = anchor.getBoundingClientRect();
  menu.style.left = `${rect.left + window.scrollX}px`;
  menu.style.top = `${rect.bottom + window.scrollY + 4}px`;
  menu.hidden = false;
}

function closePosMenu() {
  $("#pos-menu").hidden = true;
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#pos-menu") && !e.target.closest("#cand-ejdict .cand")) closePosMenu();
});

async function save() {
  if (!reg.lemma) return;
  const body = { senses: reg.senses, example: $("#example").value };
  try {
    await api("PUT", `/api/words/${encodeURIComponent(reg.lemma)}`, body);
  } catch (e) {
    $("#save-msg").textContent = e.message;
    return;
  }
  const word = reg.lemma;
  resetRegister();
  flash(`「${word}」を保存しました`);
}

async function removeCurrent() {
  if (!reg.lemma || !confirm(`「${reg.lemma}」を削除しますか？`)) return;
  await api("DELETE", `/api/words/${encodeURIComponent(reg.lemma)}`);
  const word = reg.lemma;
  resetRegister();
  flash(`「${word}」を削除しました`);
}

function resetRegister() {
  reg.lemma = null;
  reg.query = "";
  $("#editor").hidden = true;
  $("#word").value = "";
  $("#lookup-note").textContent = "";
  $("#word").focus();
}

let flashTimer;
function flash(text) {
  $("#save-msg").textContent = "";
  $("#lookup-note").textContent = text;
  clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    if (!reg.lemma) $("#lookup-note").textContent = "";
  }, 2000);
}

$("#lookup-form").addEventListener("submit", (e) => {
  e.preventDefault();
  // 検索結果を表示中で入力が変わっていなければ保存、変わっていれば検索し直す
  if (reg.lemma && $("#word").value === reg.query) save();
  else lookup();
});

$("#save").addEventListener("click", save);
$("#delete").addEventListener("click", removeCurrent);
$("#add-sense").addEventListener("click", () => {
  const unused = POS_LIST.find((p) => !reg.senses.some((s) => s.pos === p)) ?? "その他";
  reg.senses.push({ pos: unused, meanings: [] });
  renderEditor();
  [...document.querySelectorAll(".sense .add")].pop()?.focus();
});

// 編集欄でフォーカスが入力欄にないときの Enter、どこでも Ctrl+Enter で保存
document.addEventListener("keydown", (e) => {
  if ($("#tab-register").hidden || !reg.lemma || e.key !== "Enter" || e.isComposing) return;
  const tag = document.activeElement?.tagName;
  if (e.ctrlKey || e.metaKey || !["INPUT", "TEXTAREA", "SELECT", "BUTTON"].includes(tag)) {
    e.preventDefault();
    save();
  }
});

// ---------- 一覧 ----------

let listTimer;
async function loadList() {
  const params = new URLSearchParams({ q: $("#q").value, pos: $("#filter-pos").value, sort: $("#sort").value });
  const words = await api("GET", `/api/words?${params}`);
  $("#list-count").textContent = `${words.length} 語`;
  $("#word-list").replaceChildren(...words.map((w) =>
    el("li", { onclick: () => openWord(w.word) }, [
      el("span", { class: "w", text: w.word }),
      el("span", { class: "m" }, w.senses.flatMap((s) => [
        el("span", { class: "p", text: s.pos }), `${s.meanings.join("、")}　`,
      ])),
      el("button", {
        type: "button", class: "del", title: "削除", text: "削除",
        onclick: async (e) => {
          e.stopPropagation();
          if (!confirm(`「${w.word}」を削除しますか？`)) return;
          await api("DELETE", `/api/words/${encodeURIComponent(w.word)}`);
          loadList();
        },
      }),
    ])));
}

function openWord(word) {
  showTab("register");
  $("#word").value = word;
  lookup();
}

$("#q").addEventListener("input", () => {
  clearTimeout(listTimer);
  listTimer = setTimeout(loadList, 300);
});
$("#filter-pos").addEventListener("change", loadList);
$("#sort").addEventListener("change", loadList);

// ---------- クイズ ----------

const quiz = { entry: null, revealed: false };

async function nextQuiz() {
  const exclude = quiz.entry ? `?exclude=${encodeURIComponent(quiz.entry.word)}` : "";
  quiz.entry = await api("GET", `/api/quiz/next${exclude}`);
  quiz.revealed = false;
  $("#quiz-answer").hidden = true;
  $("#quiz-buttons").hidden = true;
  if (!quiz.entry) {
    $("#quiz-word").textContent = "単語がまだありません";
    $("#quiz-hint").hidden = true;
    $("#quiz-stats").textContent = "";
    return;
  }
  $("#quiz-word").textContent = quiz.entry.word;
  $("#quiz-hint").hidden = false;
  const r = quiz.entry.review;
  $("#quiz-stats").textContent = `覚えた ${r.correct} 回 / まだ ${r.wrong} 回`;
}

function reveal() {
  if (!quiz.entry || quiz.revealed) return;
  quiz.revealed = true;
  const e = quiz.entry;
  $("#quiz-answer").replaceChildren(
    ...e.senses.map((s) => el("p", {}, [el("strong", { text: `${s.pos}　` }), s.meanings.join("、")])),
    ...(e.example ? [el("p", { class: "ex", text: e.example })] : []),
  );
  $("#quiz-answer").hidden = false;
  $("#quiz-hint").hidden = true;
  $("#quiz-buttons").hidden = false;
}

async function answer(result) {
  if (!quiz.revealed) return;
  await api("POST", `/api/quiz/${encodeURIComponent(quiz.entry.word)}`, { result });
  nextQuiz();
}

$("#quiz-card").addEventListener("click", reveal);
$("#quiz-correct").addEventListener("click", () => answer("correct"));
$("#quiz-wrong").addEventListener("click", () => answer("wrong"));

document.addEventListener("keydown", (e) => {
  if ($("#tab-quiz").hidden) return;
  if (e.key === " ") { e.preventDefault(); reveal(); }
  else if (e.key === "ArrowRight") answer("correct");
  else if (e.key === "ArrowLeft") answer("wrong");
});

// ---------- 起動 ----------

(async () => {
  const st = await api("GET", "/api/status");
  POS_LIST = st.pos_list;
  $("#filter-pos").append(...POS_LIST.map((p) => el("option", { value: p, text: p })));
  if (!st.dictionary) {
    const b = $("#banner");
    b.replaceChildren("辞書がありません: ", el("code", { text: "uv run python 単語帳アプリ/setup_dict.py" }), " を実行してください");
    b.hidden = false;
  }
})();

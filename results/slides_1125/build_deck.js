// 11/25 の進捗発表のスライドを作る（pptxgenjs）。
//   node results/slides_1125/build_deck.js
// 出力: thesis/進捗発表_2026-11-25.pptx
// 図は results/slides_1125/make_figures.py と fig_architectures.png（docs/images/slide_architectures.svg を PNG にしたもの）。
//
// AI っぽさを避けるための約束（調べた特徴の逆を取る）:
//   - 背景は白。帯・アクセント線・影・グラデーション・角丸カード・アイコン・絵文字を使わない
//   - 色はグラフのモデルの区別（A〜D の 4 色）だけに使う。文字は黒と灰色
//   - タイトルはそのページの主張を具体的に書く（数値を入れる）。長さはそろえない
//   - 1 枚 1 メッセージ。図を大きく、文字は少なく。本文は左揃え
//   - 配置はページごとに内容に合わせて変える（同じ型を繰り返さない）
//   - 参考文献はスライドの左下に書く（研究室の発表の掟）
const path = require('path');
const pptxgen = require(process.env.PPTXGENJS || 'pptxgenjs');

const ROOT = path.resolve(__dirname, '..', '..');
const FIG = (f) => path.join(__dirname, f);
const OUT = process.env.OUT || path.join(ROOT, 'thesis', '進捗発表_2026-11-25.pptx');

const FONT = 'Yu Gothic';
const MATH = 'Times New Roman';
const INK = '1A1A1A';
const INK2 = '5C5C5C';
const LINE = 'B8B8B8';
const COL = { A: '2A78D6', B: 'EB6834', C: '1BAF7A', D: 'EDA100' };

const pres = new pptxgen();
pres.layout = 'LAYOUT_WIDE'; // 13.333 × 7.5 in
pres.author = '山下 陽平';
pres.title = '神経細胞モデルの 1 ステップを置き換える代理モデル';
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

const N_PAGES = 15;
let page = 0;

// 変数は斜体（Times New Roman）、それ以外は本文の書体。'$V$' のように $ で囲んだ部分を斜体にする
// 数式の中では x^{11}（上付き）と W_{hh}（下付き）も書ける。'\n' は同じ段落の中での改行
function runs(s, base = {}) {
  const out = [];
  s.split('\n').forEach((line, li, lines) => {
    line.split(/(\$[^$]+\$)/).forEach((p) => {
      if (!p) return;
      if (!p.startsWith('$')) { out.push({ text: p, options: { ...base } }); return; }
      const m = { ...base, italic: true, fontFace: MATH };
      p.slice(1, -1).split(/(\^\{[^}]*\}|_\{[^}]*\})/).forEach((q) => {
        if (!q) return;
        if (q.startsWith('^{')) out.push({ text: q.slice(2, -1), options: { ...m, italic: false, superscript: true } });
        else if (q.startsWith('_{')) out.push({ text: q.slice(2, -1), options: { ...m, subscript: true } });
        else out.push({ text: q, options: { ...m, italic: !/^[0-9.\s]+$/.test(q) } });
      });
    });
    if (li < lines.length - 1 && out.length) {
      out[out.length - 1].options = { ...out[out.length - 1].options, breakLine: true };
    }
  });
  return out;
}

// 箇条書きなどの段落の並び。items: [{t: '文', lv: 0|1, b: 太字, c: 色}]
function paras(items, size = 18) {
  const out = [];
  items.forEach((it, i) => {
    const base = { fontSize: it.size || size, color: it.c || INK, bold: !!it.b, fontFace: FONT };
    const rs = runs(it.t, base);
    rs[0].options = {
      ...rs[0].options,
      bullet: it.bullet === false ? false : it.lv ? { indent: 18 } : false,
      indentLevel: it.lv || 0,
      paraSpaceBefore: i === 0 ? 0 : (it.gap ?? 8),
    };
    rs[rs.length - 1].options = { ...rs[rs.length - 1].options, breakLine: i < items.length - 1 };
    out.push(...rs);
  });
  return out;
}

// label を渡すとページ番号の代わりにそれを出す（付録のページ。本編の番号はずらさない）
function newSlide(title, notes, label) {
  const s = pres.addSlide();
  s.background = { color: 'FFFFFF' };
  if (!label) page += 1;
  if (title) {
    s.addText(runs(title, { fontSize: 28, bold: true, color: INK, fontFace: FONT }),
      { x: 0.6, y: 0.35, w: 12.1, h: 0.85, valign: 'middle', margin: 0, isTextBox: true });
  }
  s.addText(label || `${page} / ${N_PAGES}`, { x: 11.9, y: 7.0, w: 0.9, h: 0.3, fontSize: 12, color: INK2, align: 'right',
    margin: 0, fontFace: FONT, isTextBox: true });
  if (notes) s.addNotes(notes);
  return s;
}

function refs(s, lines) {
  s.addText(lines.join('\n'), { x: 0.6, y: 7.5 - 0.3 - 0.22 * lines.length, w: 10.8, h: 0.22 * lines.length,
    fontSize: 10.5, color: INK2, margin: 0, valign: 'bottom', fontFace: FONT, isTextBox: true });
}

function box(s, x, y, w, h, text, opt = {}) {
  s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: opt.fill || 'FFFFFF' },
    line: { color: opt.line || INK2, width: 1.25 } });
  if (text) s.addText(runs(text, { fontSize: opt.size || 16, color: INK, fontFace: FONT }),
    { x, y, w, h, align: 'center', valign: 'middle', margin: 2, isTextBox: true });
}

function arrow(s, x1, y1, x2, y2, opt = {}) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001,
    h: Math.abs(y2 - y1) || 0.001, flipH: x2 < x1, flipV: y2 < y1,
    line: { color: opt.color || INK2, width: 1.5, endArrowType: opt.noHead ? undefined : 'triangle',
      dashType: opt.dash ? 'dash' : 'solid' } });
}

function text(s, t, x, y, w, h, opt = {}) {
  s.addText(Array.isArray(t) ? t : runs(t, { fontSize: opt.size || 18, color: opt.c || INK, bold: !!opt.b,
    fontFace: FONT }), { x, y, w, h, margin: 0, valign: opt.valign || 'top', align: opt.align || 'left',
    isTextBox: true, ...(opt.extra || {}) });
}

// ---------------------------------------------------------------- 1. 表紙
{
  const s = newSlide(null,
    '本日は、神経細胞シミュレーションを速くするための代理モデルについて、HH モデルで 4 つの構成を比べた結果を報告します。');
  text(s, '神経細胞モデルの 1 ステップを置き換える代理モデル', 0.9, 2.0, 11.8, 0.9, { size: 32, b: true });
  text(s, 'HH モデルで「隠れ層も次の時刻に戻すか」と「式に沿って分けるか」を比べる', 0.9, 2.95, 11.5, 0.6, { size: 22, c: INK2 });
  text(s, paras([
    { t: '山下 陽平', size: 20 },
    { t: '山口大学 理学部 物理・情報科学科　生体情報システム研究室', size: 16, c: INK2, gap: 4 },
    { t: '2026 年 11 月 25 日　進捗報告', size: 16, c: INK2, gap: 4 },
  ]), 0.9, 4.6, 11.5, 1.3);
}

// ---------------------------------------------------------------- 2. 背景
{
  const s = newSlide('区画モデルの計算量は「細胞の数 × 区画の数」で増える',
    '脳の情報処理を調べる方法の一つが、ニューロンを数理モデルにしてシミュレーションすることです。\n' +
    '樹状突起の働きまで再現するには、細胞を多数の区画に分けたモデルが必要で、計算量は細胞の数と区画の数に比例して増えます。\n' +
    '式の中に指数関数が多く、1 細胞ずつ評価するのが重いところです。');
  text(s, paras([
    { t: 'ヒトの脳のニューロンは約 $10^{11}$ 個 [1]' },
    { t: '樹状突起での計算を再現するには、細胞を数十の区画に分ける [2, 3]' },
    { t: '1 区画のモデルの数十〜数百倍の計算量になる [4]', gap: 8 },
  ], 18), 0.6, 1.55, 5.8, 2.6);
  text(s, paras([
    { t: '各区画で、毎ステップ', c: INK2, size: 16 },
    { t: '指数関数を含む微分方程式を数値積分する', size: 18, b: true, gap: 2 },
  ]), 0.6, 4.4, 5.4, 1.0);

  // 右：Traub 型の区画モデルの模式（細胞体 1 + 樹状突起の区画の列）
  const x0 = 6.7, y0 = 1.6;
  text(s, '例：Traub モデル（1 細胞 = 19 区画）[5]', x0, y0, 6.0, 0.4, { size: 16, c: INK2 });
  const cw = 0.29, ch = 0.5, gy = 2.55;
  for (let i = 0; i < 19; i++) {
    const isSoma = i === 9;
    s.addShape(isSoma ? pres.shapes.OVAL : pres.shapes.RECTANGLE, {
      x: x0 + i * (cw + 0.03), y: gy + (isSoma ? -0.12 : 0), w: cw, h: isSoma ? ch + 0.24 : ch,
      fill: { color: isSoma ? 'D9D9D9' : 'F2F2F2' }, line: { color: INK2, width: 1 } });
  }
  text(s, '樹状突起', x0, gy + 0.75, 2.8, 0.35, { size: 14, c: INK2 });
  text(s, '細胞体', x0 + 9 * (cw + 0.03) - 0.25, gy + 0.75, 0.9, 0.35, { size: 14, c: INK2 });
  text(s, '樹状突起', x0 + 3.3, gy + 0.75, 2.7, 0.35, { size: 14, c: INK2, align: 'right' });
  text(s, paras([
    { t: '区画ごとに膜電位 $V$ とゲート変数・Ca 濃度などを持つ', size: 16 },
    { t: '細胞が $B$ 個なら、1 ステップで 19 $B$ 区画分の式を解く', size: 16, gap: 6 },
  ]), x0, 3.65, 6.0, 1.0);

  // 本研究の方針：1 ステップを NN に置き換え、全細胞を 1 回の行列積でまとめる
  text(s, '本研究：1 ステップの計算をニューラルネット（NN）に置き換える', 0.6, 5.35, 12.1, 0.4, { size: 18, b: true });
  text(s, '全細胞が同じ重みを使うので、$B$ 個の細胞の 1 ステップを行列積でまとめて計算できる（GPU 向き）',
    0.6, 5.8, 12.1, 0.4, { size: 16, c: INK2 });
  refs(s, ['[1] E. R. Kandel ら, カンデル神経科学 第 2 版, p. 57, 2022.　[2] A. Gidon et al., Science, 367, pp. 83–87, 2020.',
    '[3] T. Branco et al., Science, 329, pp. 1671–1675, 2010.　[4] 小林泰良, 博士論文, 電気通信大学, p. 17, 2022.',
    '[5] R. D. Traub et al., J. Neurophysiol., 66, pp. 635–650, 1991.']);
}

// ---------------------------------------------------------------- 3. 目的と問題
{
  const s = newSlide('目的：19 区画の Traub モデルを代理モデルで速く計算する',
    '最終的な目的は、Traub モデルの代理モデルを作り、精度と速度の両方で元の微分方程式と比べることです。\n' +
    'いきなり 19 区画で試すと原因の切り分けが難しいので、1 区画の HH モデルで作り方を決めています。\n' +
    '代理モデルは自分の出力を次の入力に戻して回すので、1 ステップの小さな誤差が 10 万ステップ分たまります。ここが一番の難しさです。');
  text(s, paras([
    { t: '最終目標', c: INK2, size: 16 },
    { t: 'Traub モデル [1] の代理モデルを作り、精度と速度を微分方程式（ODE）と比べる', gap: 2 },
    { t: '今回', c: INK2, size: 16, gap: 18 },
    { t: '1 区画の HH モデル [2] で、作り方を決める', gap: 2 },
    { t: '問い 1　出力に加えて隠れ層も戻す（Elman 型）と良いか', gap: 12, b: true, size: 17 },
    { t: '問い 2　HH の式に沿って NN を 3 つに分けると良いか', gap: 6, b: true, size: 17 },
  ]), 0.6, 1.5, 6.2, 4.2);

  // 右：閉ループの図
  const x = 7.1, y = 2.0;
  box(s, x, y, 1.9, 1.3, '$V, m, h, n$\n($t$)', { size: 17 });
  box(s, x + 2.4, y + 0.15, 1.3, 1.0, 'NN', { size: 18, fill: 'F2F2F2' });
  box(s, x + 4.2, y, 1.4, 1.3, '$V, m, h, n$\n($t+dt$)', { size: 17 });
  arrow(s, x + 1.9, y + 0.65, x + 2.4, y + 0.65);
  arrow(s, x + 3.7, y + 0.65, x + 4.2, y + 0.65);
  text(s, '電流 $I$($t$)', x + 2.35, y - 0.75, 1.6, 0.35, { size: 16, align: 'center' });
  arrow(s, x + 3.05, y - 0.38, x + 3.05, y + 0.15);
  // 戻り（点線）
  arrow(s, x + 4.9, y + 1.3, x + 4.9, y + 1.75, { dash: true, noHead: true });
  arrow(s, x + 4.9, y + 1.75, x + 0.95, y + 1.75, { dash: true, noHead: true });
  arrow(s, x + 0.95, y + 1.75, x + 0.95, y + 1.3, { dash: true });
  text(s, '出力を次の時刻の入力に戻す（閉ループ）', x, y + 1.85, 5.6, 0.35, { size: 15, c: INK2, align: 'center' });
  text(s, paras([
    { t: '$dt$ = 0.01 ms なので、1 s 回すと $10^{5}$ ステップ', size: 17 },
    { t: '→ 1 ステップの誤差が積み重なり、スパイクの時刻がずれる', size: 17, gap: 6 },
  ]), x, y + 2.6, 5.6, 1.0);
  refs(s, ['[1] R. D. Traub et al., J. Neurophysiol., 66, pp. 635–650, 1991.',
    '[2] A. L. Hodgkin, A. F. Huxley, J. Physiol., 117, pp. 500–544, 1952.']);
}

// ---------------------------------------------------------------- 4. 4 構成
{
  const s = newSlide('比べた 4 構成：何を次の時刻に戻すか × 式に沿って分けるか',
    '4 つとも、出力の V, m, h, n を次の時刻の入力に戻して回します。出力を戻すつなぎ方を Jordan 型と呼びます。\n' +
    '列の違いは、それに加えて隠れ層も次の時刻の自分に戻すかどうかです。隠れ層を戻すつなぎ方を Elman 型と呼び、右の B と D がこれを足したものです。\n' +
    'A は 5 つの値を入れて 4 つの変化量を出す MLP 1 個です。B は A に隠れ層を戻す重み W_hh を足したものです。\n' +
    '行の違いは分けるかどうかです。C と D は HH の式の 3 つの部分、つまり速度定数、ゲート変数の更新、膜電位の更新をそれぞれ別のネットにしたものです。');
  // 図は 1280 × 640（docs/images/slide_architectures.svg）。下の注と重ならないよう幅 10.4 in にする
  s.addImage({ path: FIG('fig_architectures.png'), x: (13.333 - 10.4) / 2, y: 1.25, w: 10.4, h: 10.4 * 640 / 1280 });
  text(s, '隠れ層：A 256 × 3 層、B 128 × 2 層（＋ $W_{hh}$）、C 128 × 2 層を 3 個、D 64 を 3 個',
    (13.333 - 10.4) / 2, 6.55, 10.4, 0.35, { size: 15, c: INK2 });
}

// ---------------------------------------------------------------- 5. 学習
{
  const s = newSlide('ずれた状態からでも HH と同じ向きに進むように学習する',
    '閉ループでは、予測が少しずれると、教師データの軌道に乗っていない状態が入力に来ます。\n' +
    'そこで、教師の状態に雑音を足してずらし、その状態で HH の式を 1 ステップ解いた値を正解にします。図の矢印がその正解の向きです。\n' +
    'さらに、外しやすい電流を足し、自分で回して出てきた状態も学習データに加えます。これが DAgger です。');
  s.addImage({ path: FIG('fig_perturb.png'), x: 0.5, y: 1.35, w: 6.2, h: 6.2 * 920 / 1280 });
  text(s, paras([
    { t: '1　ずらした状態で学習する', b: true },
    { t: '$V$ に 2 mV、ゲートに 0.02 の雑音を足し、その状態での HH の 1 ステップを正解にする', size: 16, gap: 4 },
    { t: '2　外しやすい電流を足す', b: true, gap: 16 },
    { t: 'ランプ・閾値付近などの電流 256 本', size: 16, gap: 4 },
    { t: '3　自分で回した状態を足す（DAgger [1]）', b: true, gap: 16 },
    { t: '閉ループで出てきた状態と、そこでの正解を学習データに加える（3 回）', size: 16, gap: 4 },
    { t: '損失：$ΔV$ は相対誤差で測る', size: 16, c: INK2, gap: 16 },
  ]), 7.0, 1.6, 5.8, 4.8);
  refs(s, ['[1] S. Ross et al., Proc. AISTATS, pp. 627–635, 2011.']);
}

// ---------------------------------------------------------------- 6. 評価
{
  const s = newSlide('スパイク時刻が ±2 ms 以内に入るかで採点する',
    '採点は、HH のスパイクの時刻の前後 2 ms 以内に代理モデルのスパイクがあるかで行います。\n' +
    'Γ は、一致した数から偶然の一致の期待値を引いて正規化したもので、完全に一致すれば 1、偶然と同じなら 0 です。\n' +
    'データは 3 つに分けています。今日の結果は開発用テスト 26 本の値で、最終テスト 54 本はまだ使っていません。');
  s.addImage({ path: FIG('fig_gamma.png'), x: 0.6, y: 1.35, w: 7.4, h: 7.4 * 500 / 1600 });
  text(s, paras([
    { t: 'coincidence factor Γ [1, 2]', b: true },
    { t: '一致した数から、偶然の一致の\n期待値を引いて正規化した値', size: 16, gap: 4 },
    { t: '完全一致で 1、偶然と同じなら 0', size: 16, gap: 8 },
  ]), 8.4, 1.6, 4.4, 1.8);

  // データの分け方（表）
  const rows = [
    [{ text: '電流のセット', options: { bold: true } }, { text: '本数', options: { bold: true, align: 'right' } },
      { text: '使い方', options: { bold: true } }],
    ['検証', { text: '17', options: { align: 'right' } }, '学習の途中の時点と、各構成の最良モデルを選ぶ'],
    ['開発用テスト', { text: '26', options: { align: 'right' } }, '作り方を比べる（今日の結果）'],
    ['最終テスト', { text: '54', options: { align: 'right' } }, '選んだ 4 モデルに 1 回だけ使う（未実施）'],
  ];
  s.addTable(rows, { x: 0.6, y: 4.15, w: 9.6, colW: [2.2, 0.9, 6.5], fontSize: 16, fontFace: FONT, color: INK,
    border: { type: 'solid', pt: 0.75, color: LINE }, margin: [4, 8, 4, 8], rowH: 0.42 });
  text(s, 'どの電流も、学習には使っていない', 0.6, 5.95, 9.6, 0.35, { size: 15, c: INK2 });
  refs(s, ['[1] W. M. Kistler et al., Neural Comput., 9, pp. 1015–1045, 1997.',
    '[2] R. Jolivet et al., J. Neurosci. Methods, 169, pp. 417–424, 2008.']);
}

// ---------------------------------------------------------------- 7. 結果：波形
{
  const s = newSlide('ランプ電流では、D だけ発火の始まりが 62 ms 遅れる',
    '電流を 0 から 25 まで 500 ms かけて上げたときの膜電位です。灰色の太線が HH です。\n' +
    'HH は 390.7 ms で発火し始めて 10 回発火します。A、B、C は 10 回とも 2 ms 以内で一致しました。\n' +
    'D は 452.7 ms まで発火せず、5 回しか発火しませんでした。\n' +
    '（考察はここでは言わない。カンペの「7 枚目」を見る）');
  s.addImage({ path: FIG('fig_wave_ramp.png'), x: 0.5, y: 1.35, w: 8.7, h: 8.7 * 1240 / 2300 });
  text(s, paras([
    { t: '電流：0 → 25 µA/cm² を\n500 ms で上げる（検証用）', size: 15, c: INK2 },
    { t: 'HH', b: true, gap: 16 },
    { t: '390.7 ms から 10 回発火', size: 16, gap: 2 },
    { t: 'A, B, C', b: true, gap: 12 },
    { t: '10 回とも ±2 ms 以内で一致', size: 16, gap: 2 },
    { t: 'D', b: true, gap: 12 },
    { t: '452.7 ms から 5 回だけ', size: 16, gap: 2 },
    { t: '発火が始まったときの電流\nHH 19.5、D 22.6 µA/cm²', size: 16, gap: 4 },
  ]), 9.5, 1.6, 3.4, 4.8);
}

// ---------------------------------------------------------------- 8. 結果：精度
{
  const s = newSlide('出力だけを戻す MLP 1 個（A）が平均 Γ で最も高い',
    '開発用テスト 26 本での成績です。左が平均 Γ、右がすべてのスパイクが一致した電流の本数です。\n' +
    '平均 Γ は A が 0.958 で最も高く、隠れ層も戻した B、D、分けた C、D はいずれも下がりました。\n' +
    'B は全部一致した本数では 24 本で最も多いのですが、外した 2 本で大きく外しているため、平均 Γ は 0.917 です。\n' +
    'どの構成も学習の seed は 1 本なので、seed によるばらつきはまだ見ていません。');
  s.addImage({ path: FIG('fig_accuracy.png'), x: 0.5, y: 1.35, w: 9.2, h: 9.2 * 1160 / 2300 });
  text(s, paras([
    { t: '開発用テスト 26 本', size: 15, c: INK2 },
    { t: '各構成とも学習の seed は 1 本', size: 15, c: INK2, gap: 2 },
    { t: '平均 Γ', b: true, gap: 18 },
    { t: 'A 0.958\nB, C, D は 0.904〜0.927', size: 17, gap: 2 },
    { t: '全一致の本数', b: true, gap: 14 },
    { t: 'B が 24 本で最多。ただし\n外した 2 本の Γ は 0 前後', size: 16, gap: 2 },
    { t: '検証 17 本での全一致\nA 17、B 16、C 16、D 15', size: 15, c: INK2, gap: 14 },
  ]), 9.95, 1.6, 3.0, 4.8);
}

// ---------------------------------------------------------------- 8-2. 生データ（26 本 × 4 モデル）
{
  const s = newSlide('4 モデルの生データ：外したのはランプ・正弦波・揺らぎの電流',
    '【表の読み方（事実のメモ。説明は自分の言葉で）】\n' +
    '・1 行が 1 本のテスト電流。HH の列は正解のスパイク数、A〜D の列は各モデルが出したスパイク数\n' +
    '・色のないセルは、すべてのスパイクが HH と ±2 ms 以内で一致した電流（Γ = 1）\n' +
    '・色のあるセルは外した電流。下の小さい数字がその電流の Γ。色が濃いほど Γ が低い\n' +
    '・本数が同じでも外すことがある（B の ramp 0-30：17 本どうしだが Γ = −0.16。全部の時刻がずれている）\n' +
    '【事実】\n' +
    '・26 本中、4 モデルとも外したのは ramp 0-20 だけ。HH が最後に 1 回だけ発火する電流で、外すと Γ = 0 になる\n' +
    '・A が外したのは 3 本（ramp 0-20、random pulse 5ms、OU 5±3）。うち 2 本は Γ 0.92 と 0.98 で、ほぼ合っている\n' +
    '・B は ramp 0-30 で大きく外した。C は sin 10Hz で 16 本中 12 本しか出さなかった\n' +
    '・D は 8 本で外した。そのうち 4 本は Γ 0.92〜0.98 の小さなずれ\n' +
    '・平均 Γ は 26 本の Γ の平均なので、Γ がほぼ 0 の電流が 1 本あると、平均が約 0.04 下がる');
  s.addImage({ path: FIG('fig_raw26.png'), x: 0.6, y: 1.3, w: 12.1, h: 12.1 * 1120 / 2400 });
  text(s, '数字：スパイクの数（HH が正解）　色なし：全スパイクが ±2 ms 以内で一致（Γ = 1）　色付き：外した電流とその Γ',
    0.6, 7.0 - 0.05, 11.0, 0.35, { size: 14, c: INK2 });
}

// ---------------------------------------------------------------- 8-3. 外した電流の波形
{
  const s = newSlide('外し方はモデルごとに違う：発火しない・時刻がずれる・余分に出る',
    '【図の読み方】灰色の太線が HH、色の線が外したモデル。右上の文字が、描いているモデル\n' +
    '【事実】\n' +
    '・ramp 0→20：HH は 492.9 ms に 1 回だけ発火する。A, B, D は発火せず、C は 499.9 ms（7 ms 遅れ）\n' +
    '・ramp 0→30：B は 17 回とも約 3 ms 早く発火する（323.5 ms に対して 320.5 ms）。±2 ms の外なので Γ が負になる。C は一致\n' +
    '・sin 10 Hz：2 周期目の最初のスパイクが HH 108.9 ms、C 118.8 ms、D 117.3 ms。C はこの後 4 本足りなくなる\n' +
    '・OU 5±3：HH が発火しない 38 ms 付近で、A と D が余分に 1 回発火する（A 37.9 ms、D 36.4 ms）\n' +
    '（共通点の解釈はここでは言わない。考察のページで話す。カンペの「10 枚目」）');
  s.addImage({ path: FIG('fig_fails.png'), x: 0.9, y: 1.35, w: 11.2, h: 11.2 * 1160 / 2400 });
}

// ---------------------------------------------------------------- 9. 結果：速度（GPU）
{
  const s = newSlide('GPU では、細胞 1,024 個までは HH より速く、8,192 個からは遅い',
    '【何を表すか】横軸は同時に計算する細胞の数、縦軸は 1 ステップの時間（両方とも対数）。黒が HH の陽的 Euler 法\n' +
    '条件：学内サーバの GPU（Tesla V100）、fp32、CUDA Graph（毎ステップの命令をまとめて GPU に渡し、起動の手間を省く方法）。HH も同じ条件\n' +
    '【事実】細胞 1 個：A 42 µs、B 37 µs、C 99 µs、D 105 µs、HH 191 µs\n' +
    '細胞 1,024 個：B 91、A 136、D 165、C 197、HH 200 µs（4 構成とも HH より速い）\n' +
    '細胞 65,536 個：HH 272、D 1,371、B 1,974、A 4,195、C 4,305 µs（HH の 5.0〜15.8 倍）\n' +
    'HH は細胞 1 個から 65,536 個で 191 → 272 µs（1.4 倍）\n' +
    '（理由はここでは言わない。13 枚目で話す）');
  s.addImage({ path: FIG('fig_speed_gpu.png'), x: 0.5, y: 1.35, w: 8.2, h: 8.2 * 1080 / 1920 });
  text(s, paras([
    { t: 'GPU：Tesla V100\nfp32、CUDA Graph', size: 15, c: INK2 },
    { t: '細胞 1 個', b: true, gap: 16 },
    { t: 'B 37 µs、A 42 µs、HH 191 µs', size: 16, gap: 2 },
    { t: '細胞 1,024 個', b: true, gap: 14 },
    { t: '4 構成とも HH（200 µs）より速い', size: 16, gap: 2 },
    { t: '細胞 65,536 個', b: true, gap: 14 },
    { t: 'HH 272 µs\nD 1,371　B 1,974　A 4,195　C 4,305', size: 16, gap: 2 },
    { t: 'HH の 5.0〜15.8 倍', size: 16, gap: 2 },
    { t: '行列積の演算回数', b: true, gap: 14 },
    { t: '1 細胞・1 ステップあたり', size: 15, c: INK2, gap: 2 },
    { t: 'A 26.7 万　C 10.5 万\nB 6.8 万　D 2.8 万', size: 16, gap: 2 },
  ]), 9.0, 1.5, 3.9, 5.3);
}

// ---------------------------------------------------------------- 10. 考察
{
  const s = newSlide('HH では、隠れ層に持ち越す情報がない',
    'A が最も良かった理由として、3 つの仮説を考えました。\n' +
    '1 つ目は、HH が V, m, h, n の 4 つが分かれば次の時刻が決まる系だということです。状態をすべて入力に戻しているので、隠れ層に持ち越しておくことがありません。B の 3 本中 2 本で、W_hh を学習する前の時点が検証で最良だったことも、これと合います。\n' +
    '2 つ目は、分けると誤差が NN の境目で重なる可能性です。\n' +
    '3 つ目の、A が大きいから良いという仮説は、同じ作り方の 128 × 2 層でも Γ 0.959 だったので棄却できます。');
  const col = (x, head, verdict, body) => {
    text(s, head, x, 1.55, 3.85, 1.2, { size: 18, b: true });
    text(s, verdict, x, 2.8, 3.85, 0.35, { size: 15, c: INK2 });
    text(s, paras(body, 16), x, 3.3, 3.85, 3.2);
  };
  col(0.6, '仮説 1\nHH の次の時刻は $V, m, h, n$ だけで決まる', '支持する結果がある', [
    { t: '状態を全部入力に戻すので、隠れ層に持ち越すことがない' },
    { t: 'B（A ＋ $W_{hh}$）の seed 3 本中 2 本は\n$W_{hh}$ を学習する前の時点が検証で最良', gap: 10 },
  ]);
  col(4.75, '仮説 2\n分けると、誤差が NN の境目で重なる', '可能性がある（未検証）', [
    { t: 'C < A、D < B' },
    { t: 'D はランプで発火の始まりが遅れた', gap: 10 },
    { t: '$α, β$ → ゲート → $V$ と誤差が伝わる可能性', gap: 4 },
    { t: '確かめ方：閾値付近の電流で、各 NN の 1 ステップの誤差を比べる', gap: 10, c: INK2 },
  ]);
  col(8.9, '仮説 3\nA は NN が大きいので良い', '棄却', [
    { t: '同じ作り方の 128 × 2 層でも\n平均 Γ 0.959、26 本中 24 本が全一致' },
    { t: '→ 大きさより、状態を全部戻す形が効いている', gap: 10 },
  ]);
  // 列の区切り（細い縦線）
  [4.55, 8.7].forEach((x) => s.addShape(pres.shapes.LINE, { x, y: 1.6, w: 0, h: 4.9, line: { color: LINE, width: 0.75 } }));
}

// ---------------------------------------------------------------- 10-2. 考察（外した電流と速度）
{
  const s = newSlide('外したのは閾値付近の場面、細胞が多いと遅いのは演算回数の差',
    '結果のページでは事実だけを見せたので、ここで解釈を話す。詳しくはカンペの「13 枚目」');
  const col = (x, w, head, facts, claims) => {
    text(s, head, x, 1.55, w, 0.5, { size: 20, b: true });
    text(s, '結果（どのページの何か）', x, 2.2, w, 0.35, { size: 15, c: INK2 });
    text(s, paras(facts, 16), x, 2.6, w, 1.9);
    text(s, '考察', x, 4.45, w, 0.35, { size: 15, c: INK2 });
    text(s, paras(claims, 16), x, 4.85, w, 1.9);
  };
  col(0.6, 5.7, '外した電流の共通点', [
    { t: '10 枚目：外した 4 種類の電流は、どれも膜電位が閾値の近くにとどまっている場面' },
    { t: '7 枚目：D が発火を始めた電流は 22.6 µA/cm²（HH は 19.5）', gap: 8 },
  ], [
    { t: '閾値の近くでは、1 ステップの小さな誤差で「発火する・しない」が入れ替わる可能性がある' },
    { t: 'D は発火の閾値を高めに覚えている可能性がある', gap: 8 },
    { t: '確かめ方：閾値付近の電流を学習に足したときに外す本数が減るかを見る', c: INK2, gap: 8 },
  ]);
  col(6.9, 5.8, 'GPU で細胞が多いと遅い理由', [
    { t: '11 枚目：65,536 個で HH の 5.0〜15.8 倍' },
    { t: 'HH は 1 個 → 65,536 個で 1.4 倍', gap: 8 },
    { t: '演算回数：A 26.7 万、D 2.8 万', gap: 8 },
  ], [
    { t: '細胞が少ないと、計算量より GPU への命令の回数で時間が決まる可能性がある' },
    { t: '細胞が多いと計算量で決まり、演算の多い NN が不利になる可能性がある', gap: 8 },
    { t: '確かめ方：fp16 で行列積を速くして、分かれ目が動くかを見る', c: INK2, gap: 8 },
  ]);
  s.addShape(pres.shapes.LINE, { x: 6.6, y: 1.6, w: 0, h: 5.0, line: { color: LINE, width: 0.75 } });
}

// ---------------------------------------------------------------- 11. まとめ
{
  const s = newSlide('まとめ：Traub には A（出力だけを戻す MLP 1 個）の作り方を移す',
    '目的は、Traub モデルの代理モデルを作るために、HH で作り方を決めることでした。\n' +
    '4 構成を比べると、出力だけを戻す MLP 1 個が最も正確で、隠れ層も戻しても、式に沿って分けても良くなりませんでした。\n' +
    '速度は GPU で、細胞 1,024 個までは HH より速く、65,536 個では 5.0〜15.8 倍遅いという結果でした。');
  text(s, paras([
    { t: '目的', c: INK2, size: 16 },
    { t: 'Traub モデルの代理モデルを作るため、HH モデルで作り方を決める', gap: 2 },
    { t: '結果', c: INK2, size: 16, gap: 18 },
    { t: '出力だけを戻す MLP 1 個（A）が最も正確（平均 Γ 0.958）', gap: 2 },
    { t: '隠れ層も戻す（B）、式に沿って分ける（C）、両方（D）では 0.904〜0.927 に下がった', gap: 8 },
    { t: 'GPU では、細胞 1,024 個までは HH より速く、65,536 個では 5.0〜15.8 倍遅い', gap: 8 },
  ]), 0.6, 1.5, 7.0, 4.5);
  // 2×2 の平均 Γ（4 構成の図と同じ並び）
  const x0 = 8.4, y0 = 2.1, cw = 2.1, ch = 1.15;
  text(s, '平均 Γ（開発用テスト 26 本）', x0 - 1.0, 1.55, 5.2, 0.35, { size: 15, c: INK2 });
  text(s, '出力を戻す', x0, y0, cw, 0.35, { size: 15, c: INK2, align: 'center' });
  text(s, '＋隠れ層も戻す', x0 + cw, y0, cw, 0.35, { size: 15, c: INK2, align: 'center' });
  text(s, '1 個', x0 - 1.0, y0 + 0.4, 0.9, ch, { size: 15, c: INK2, valign: 'middle', align: 'right' });
  text(s, '3 個', x0 - 1.0, y0 + 0.4 + ch, 0.9, ch, { size: 15, c: INK2, valign: 'middle', align: 'right' });
  const cell = (k, v, i, j) => {
    s.addShape(pres.shapes.RECTANGLE, { x: x0 + j * cw, y: y0 + 0.4 + i * ch, w: cw, h: ch,
      fill: { color: 'FFFFFF' }, line: { color: LINE, width: 1 } });
    s.addText([{ text: k + '  ', options: { fontSize: 18, bold: true, color: COL[k], fontFace: FONT } },
      { text: v, options: { fontSize: 24, bold: k === 'A', color: INK, fontFace: FONT } }],
      { x: x0 + j * cw, y: y0 + 0.4 + i * ch, w: cw, h: ch, align: 'center', valign: 'middle', margin: 0, isTextBox: true });
  };
  cell('A', '0.958', 0, 0); cell('B', '0.917', 0, 1); cell('C', '0.927', 1, 0); cell('D', '0.904', 1, 1);
}

// ---------------------------------------------------------------- 12. 今後
{
  const s = newSlide('12 月は Traub に移り、速度を先に測る',
    '今後は Traub モデルに移ります。\n' +
    '速度は重みの値に関係なく、NN の形と細胞の数で決まるので、学習が終わる前に乱数の重みで測れます。勝てる見込みを先に確かめます。\n' +
    '12 月 20 日の時点で、すべてを NN にするか、一部の区画だけ置き換えるかを決めます。');
  const rows = [
    [{ text: '時期', options: { bold: true } }, { text: 'やること', options: { bold: true } }],
    ['11/25 まで', 'HH の 4 モデルを最終テスト 54 本で 1 回だけ採点する'],
    ['11/26〜12/6', '速度を先に測る：GPU 上で ODE と代理モデル（乱数の重み）を比べ、損益分岐の細胞数を出す\n教師データ：200 軌道 × 2〜5 s（学習 70 / 検証 15 / テスト 15 %、軌道ごとに分ける）'],
    ['12/7〜12/20', '学習と閉ループの微調整。精度（Γ、バースト、樹状突起の Ca）を評価する'],
    ['12/20', 'すべてを NN にするか、一部の区画だけ置き換えるかを決める'],
    ['12/27', '卒論の題目を決める'],
    ['1/20、1/31', '初稿、提出'],
  ];
  s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [2.0, 10.1], fontSize: 16, fontFace: FONT, color: INK,
    border: { type: 'solid', pt: 0.75, color: LINE }, margin: [5, 8, 5, 8], valign: 'middle' });
}

// ---------------------------------------------------------------- 付録. ハイパーパラメータ
// 値の出どころ: 各モデルの結果 JSON の args（results/hh_improve/{r8_mlp_hd_w256, sp_r_init, m3_hdf_s0}.json、
// results/hh_rnn3/p_h64_cl1.json）と、.pt の cfg・重みの形（パラメータ数）。B の出発点は results/hh_improve/mm_rel.json
{
  const s = newSlide('付録：4 モデルのハイパーパラメータ',
    '質問されたとき用。値は各モデルの学習の結果ファイル（JSON の args）と、保存した重み（.pt）から取った。\n' +
    '・4 つとも Adam、学習率は余弦減衰で最後に 1/100 まで下げる（最初の 2% は線形に上げる）。勾配のノルムは 1 で切る。float64\n' +
    '・A と C の学習率は、事前学習と DAgger をまとめて 1 本の余弦減衰（合計 11 万回）\n' +
    '・B の出発点は、A と同じ作り方の 128 × 2 層（mm_rel、事前学習 5 万回、外しやすい電流と DAgger なし）。W_hh = 0 から始める\n' +
    '・B は seed 3 本のうち seed 0 だけ W_hh が育った。seed 1, 2 は W_hh = 0 の時点が検証で最良だった\n' +
    '・形がそろっていない（A のパラメータ数は D の約 9 倍）。各構成で検証 17 本の成績が一番良いものを選んだため',
    '付録');
  const H = (t, k) => ({ text: t, options: { bold: true, color: k ? COL[k] : INK } });
  const rows = [
    [H(''), H('A　MLP 1 個', 'A'), H('B　A ＋ W_hh', 'B'), H('C　MLP 3 個', 'C'), H('D　RNN 3 個', 'D')],
    ['隠れ層', '256 × 3 層', '128 × 2 層 ＋ W_hh', '128 × 2 層 × 3 個', '64 × 1 層 × 3 個'],
    ['活性化関数', 'SiLU', 'SiLU', 'SiLU', 'tanh'],
    ['パラメータ数', '13.4 万', '3.4 万', '5.4 万', '1.4 万'],
    ['始めの重み', '乱数', '128 × 2 層の MLP\n（W_hh = 0）', '乱数', '乱数（W_hh = 0）'],
    ['学習回数', '5 万 ＋ 2 万 × 3', '2 万', '5 万 ＋ 2 万 × 3', '2 万'],
    ['学習率（最初）', '1e-3', '3e-4', '1e-3', '1e-3'],
    ['バッチ', '1024', '64', '1024', '256'],
    ['1 回に流す長さ', '1 ステップ', '32 ステップ', '1 ステップ', '200 ステップ'],
    ['状態の雑音', 'V 2 mV、ゲート 0.02', '同じ', '同じ', '同じ'],
    ['外しやすい電流', '256 本 × 3 s', 'なし', '256 本 × 3 s', '256 本 × 1 s'],
    ['DAgger', '3 回（256 本 × 1 s）', 'なし', '3 回（256 本 × 1 s）', '3 回（128 本 × 1 s）'],
    ['閉ループの損失', 'なし', 'なし', 'なし', '50 ステップ先まで'],
  ];
  s.addTable(rows, { x: 0.6, y: 1.3, w: 12.1, colW: [2.1, 2.5, 2.5, 2.5, 2.5], fontSize: 14, fontFace: FONT,
    color: INK, border: { type: 'solid', pt: 0.75, color: LINE }, margin: [3, 6, 3, 6], rowH: 0.38, valign: 'middle' });
  text(s, '学習回数の「5 万 ＋ 2 万 × 3」は、事前学習 5 万回のあと DAgger を 2 万回ずつ 3 回\n' +
    '共通：Adam、学習率は余弦減衰で 1/100 まで、勾配のノルムを 1 で切る、float64、刻み幅 0.01 ms',
    0.6, 6.6, 11.0, 0.55, { size: 13, c: INK2 });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log('wrote', f));

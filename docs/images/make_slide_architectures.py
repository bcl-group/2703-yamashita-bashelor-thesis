import re
import sys

# 使い方: python docs/images/make_slide_architectures.py docs/images/slide_architectures.svg
OUTPATH = sys.argv[1]


def M(s):
    """数式（LaTeX 風）：ラテン文字・ギリシャ小文字は斜体のセリフ体、Δ・数字・記号は立体"""
    out = re.sub(r'([A-Za-zα-ω]+)', r'<tspan font-style="italic">\1</tspan>', s)
    return f'<tspan class="m">{out}</tspan>'


WHH = ('<tspan class="m"><tspan font-style="italic">W</tspan>'
       '<tspan font-size="70%" font-style="italic" dy="5">hh</tspan></tspan><tspan dy="-5"> </tspan>')
MLP_FILL = 'fill="#efeeea" stroke="#52514e" stroke-width="2"'
RNN_FILL = MLP_FILL
C = ['#2a78d6', '#eb6834', '#1baf7a']


def loop(x, y):
    """隠れ層が次の時刻の自分に戻る小さなループ矢印（Elman 型の W_hh）。(x, y) はループの左下"""
    return f'    <path d="M{x},{y} A9,9 0 1 1 {x + 16},{y}" class="ln" marker-end="url(#ar)"/>'

OUT = '''    <rect x="0" y="60" width="74" height="150" rx="8" fill="#f6f5f2" stroke="#52514e"/>
    <text x="37" y="96" class="t" text-anchor="middle">{V}</text>
    <text x="37" y="126" class="t" text-anchor="middle">{mhn}</text>
    <text x="37" y="150" class="t" text-anchor="middle">({t} )</text>
    <text x="37" y="190" class="t" text-anchor="middle">{I}</text>
    <rect x="476" y="60" width="70" height="150" rx="8" fill="#f6f5f2" stroke="#52514e"/>
    <text x="511" y="96" class="t" text-anchor="middle">{V}</text>
    <text x="511" y="126" class="t" text-anchor="middle">{mhn}</text>
    <text x="511" y="152" class="t" text-anchor="middle" font-size="17">({tdt} )</text>
    <path d="M511,212 L511,230 L37,230 L37,214" class="lp" marker-end="url(#ar)"/>'''.format(
    V=M('V'), mhn=M('m h n'), t=M('t'), I=M('I') + '(' + M('t') + ' )', tdt=M('t+dt'))


def one(fill, name, rec):
    stripes = ('\n' + loop(420, 92)) if rec else ''
    return f'''    <path d="M76,135 L100,135" class="ln" marker-end="url(#ar)"/>
    <rect x="104" y="60" width="352" height="150" rx="10" {fill}/>{stripes}
    <text x="286" y="118" class="t" text-anchor="middle">{name}</text>
    <text x="282" y="182" class="t" text-anchor="middle">{M("V, m, h, n, I")} → {M("ΔV, Δm, Δh, Δn")}</text>
    <path d="M458,135 L472,135" class="ln" marker-end="url(#ar)"/>'''


IO = [('V', 'α, β'), ('α, β, m, h, n', 'Δm, Δh, Δn'), ('m, h, n, V, I', 'ΔV')]


def three(fill, nm, rec):
    s = []
    for i, (y, c) in enumerate(zip([60, 113, 166], C)):
        a, b = IO[i]
        s.append(f'''    <path d="M76,{y + 22} L100,{y + 22}" class="ln" marker-end="url(#ar)"/>
    <rect x="104" y="{y}" width="352" height="44" rx="8" {fill}/>
    <text x="122" y="{y + 28}" class="u">{nm}{i + 1}</text>
    <text x="196" y="{y + 28}" class="u">{M(a)} → {M(b)}</text>''')
        if rec:
            s.append(loop(426, y + 32))
        if i > 0:   # ブロック 1 の α, β はブロック 2 にだけ渡す（出力には行かない）
            s.append(f'    <path d="M458,{y + 22} L472,{y + 22}" class="ln" marker-end="url(#ar)"/>')
    # ブロック 1 の α, β をブロック 2 へ渡す
    s.append('    <path d="M176,104 L176,111" class="ln" marker-end="url(#ar)"/>')
    return '\n'.join(s)


def panel(x, y, title, sub, body, cm):
    return f'''
  <!-- {cm} -->
  <g transform="translate({x},{y})">
    <text x="0" y="36" class="h">{title}</text>
{body}
{OUT}
  </g>'''


W, H = 1280, 640
X1, X2, Y1, Y2 = 140, 712, 48, 316
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Yu Gothic, Meiryo, sans-serif">
  <!-- スライド用のポンチ絵：候補 4 モデルの 2×2（列 = 何を次の時刻に戻すか（Jordan 型／Jordan＋Elman 型）、行 = 1 個でまとめるか、HH の式に沿って 3 個に分けるか）。
       色は使わない。破線 = 出力を戻す（4 つとも共通）、小さなループ = 隠れ層を戻す（B と D だけ）。
       変数は数式の書体（セリフ体の斜体、Δ は立体）。生成: docs/images/make_slide_architectures.py -->
  <defs>
    <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#52514e"/>
    </marker>
    <style>
      .t  {{ font-size: 20px; fill: #0b0b0b; }}
      .s  {{ font-size: 16px; fill: #52514e; }}
      .u  {{ font-size: 17px; fill: #0b0b0b; }}
      .h  {{ font-size: 22px; font-weight: bold; fill: #0b0b0b; }}
      .m  {{ font-family: 'Cambria Math', 'Times New Roman', serif; font-size: 112%; }}
      .ln {{ stroke: #52514e; stroke-width: 2; fill: none; }}
      .lp {{ stroke: #52514e; stroke-width: 2; fill: none; stroke-dasharray: 6 4; }}
    </style>
  </defs>
  <rect width="{W}" height="{H}" fill="#ffffff"/>

  <text x="{X1 + 273}" y="30" class="h" text-anchor="middle">出力を戻す（Jordan 型）</text>
  <text x="{X2 + 273}" y="30" class="h" text-anchor="middle">出力＋隠れ層を戻す（Jordan＋Elman 型）</text>
  <text x="20" y="{Y1 + 130}" class="t">1 個で</text>
  <text x="20" y="{Y1 + 156}" class="t">まとめて</text>
  <text x="20" y="{Y2 + 118}" class="t">3 個に</text>
  <text x="20" y="{Y2 + 144}" class="t">式に沿って</text>
  <text x="20" y="{Y2 + 170}" class="t">分ける</text>
  <line x1="20" y1="{Y2 - 18}" x2="1260" y2="{Y2 - 18}" stroke="#e4e3df" stroke-width="2"/>
  <line x1="{X2 - 14}" y1="12" x2="{X2 - 14}" y2="{Y2 + 240}" stroke="#e4e3df" stroke-width="2"/>
''' + panel(X1, Y1, 'A　MLP', 'MLP 1 本で HH の式全体を置き換える',
            one(MLP_FILL, 'MLP', False), 'A: Markov MLP') \
    + panel(X2, Y1, 'B　RNN', f'A に、前の時刻の中間出力を戻す重み {WHH}を足す',
            one(RNN_FILL, 'RNN', True), 'B: MLP を RNN に置き換えたもの') \
    + panel(X1, Y2, 'C　MLP 3 個', 'HH の式の 3 つの部分を 1 つずつ MLP で置き換える',
            three(MLP_FILL, 'MLP', False), 'C: MLP 3 個') \
    + panel(X2, Y2, 'D　RNN 3 個', 'C の各 MLP を RNN にする（並列）',
            three(RNN_FILL, 'RNN', True), 'D: RNN 3 個（並列）') + f'''

  <!-- 凡例（最小限） -->
  <g transform="translate(20,{H - 62})">
    <path d="M0,9 L36,9" class="lp"/>
    <text x="44" y="15" class="s">出力を次の時刻の入力に戻す（Jordan 型、4 つとも）</text>
{loop(470, 13)}
    <text x="508" y="15" class="s">隠れ層を次の時刻の自分に戻す（Elman 型、{WHH}）</text>
    <text x="0" y="45" class="s">ネットは変化量 Δ を出し、今の値に足して次の時刻の値にする（{M("V")}({M("t+dt")} ) = {M("V")}({M("t")} ) + {M("ΔV")} など）</text>
  </g>
</svg>
'''
open(OUTPATH, 'w', encoding='utf-8').write(svg)
print('saved', OUTPATH)

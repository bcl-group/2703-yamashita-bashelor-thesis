"""BPTT 説明用の図（SVG）を生成する。

(a) 1時刻分の結合を拡大し、各矢印に w_{jj'} を付けた図
(b) 時間方向に展開し、矢印の束を行列 W にまとめた図
"""
import math
from pathlib import Path

# スクリプトと同じフォルダに保存する（フォルダを移動しても動くように）
OUT = Path(__file__).with_name("bptt_unrolled.svg")

C_HX = "#2563eb"   # 入力→隠れ層
C_HH = "#ea580c"   # 隠れ層→次時刻の隠れ層（再帰辺）
C_YH = "#16a34a"   # 隠れ層→出力
C_NODE = "#334155"
C_PREV = "#94a3b8"
FONT = "'Noto Sans JP','Yu Gothic','Hiragino Sans',sans-serif"

parts = []


def add(s):
    parts.append(s)


def sym(base, sup=None, sub=None, size=16):
    """base^sup_sub を tspan で組む（SVG text の中身）。"""
    s = f'<tspan font-style="italic">{base}</tspan>'
    small = round(size * 0.7)
    if sup:
        s += f'<tspan dy="{-size*0.4:.1f}" font-size="{small}">{sup}</tspan>'
    if sub:
        dy = size * 0.75 if sup else size * 0.3
        if sup:
            # 上付きのあとに下付きを並べる
            s += f'<tspan dy="{dy:.1f}" font-size="{small}">{sub}</tspan>'
            s += f'<tspan dy="{-(dy - size*0.4):.1f}">&#8203;</tspan>'
        else:
            s += f'<tspan dy="{dy:.1f}" font-size="{small}">{sub}</tspan>'
            s += f'<tspan dy="{-dy:.1f}">&#8203;</tspan>'
    elif sup:
        s += f'<tspan dy="{size*0.4:.1f}">&#8203;</tspan>'
    return s


def text(x, y, content, size=16, anchor="middle", color="#0f172a", weight="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}" font-family="{FONT}">{content}</text>')


def node(x, y, r, label, color=C_NODE, fill="#ffffff", size=15, dashed=False):
    dash = ' stroke-dasharray="4 3"' if dashed else ""
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="2"{dash}/>')
    if label:
        text(x, y + size * 0.35, label, size=size, color=color)


def arrow(p, q, r_from, r_to, color, dashed=False, width=2.0):
    (x1, y1), (x2, y2) = p, q
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    ux, uy = dx / L, dy / L
    sx, sy = x1 + ux * r_from, y1 + uy * r_from
    ex, ey = x2 - ux * (r_to + 2), y2 - uy * (r_to + 2)
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    mid = {C_HX: "ahx", C_HH: "ahh", C_YH: "ayh"}[color]
    add(f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="{color}" '
        f'stroke-width="{width}"{dash} marker-end="url(#{mid})"/>')


W, H = 900, 960
add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add('<defs>')
for mid, c in (("ahx", C_HX), ("ahh", C_HH), ("ayh", C_YH)):
    add(f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')

# ---------------- (a) 1時刻分の結合 ----------------
text(30, 34, "(a) 1時刻分の結合（隠れノード h₁ に入る矢印と、出力への矢印だけを描いた）",
     size=17, anchor="start", weight="bold")

R = 22
h = [(420, 240), (560, 240), (700, 240)]
prev = [(150, 140), (150, 240), (150, 340)]
xs = [(390, 390), (490, 390)]
yk = (560, 80)

# 層の名前
text(150, 92, sym("h", "(t−1)", size=17) + "　1時刻前", size=17, color=C_PREV)
text(800, 246, sym("h", "(t)", size=17) + "　隠れ層", size=17, anchor="start")
text(620, 397, sym("x", "(t)", size=17) + "　入力", size=17, anchor="start")
text(620, 87, sym("ŷ", "(t)", size=17) + "　出力", size=17, anchor="start")

# 矢印（ノードより先に描く）
for j, p in enumerate(prev):
    arrow(p, h[0], R, R, C_HH, dashed=True)
for i, p in enumerate(xs):
    arrow(p, h[0], R, R, C_HX)
for j, p in enumerate(h):
    arrow(p, yk, R, R, C_YH)

# ノード
for j, p in enumerate(prev):
    node(*p, R, f"h{'₁₂₃'[j]}", color=C_PREV, dashed=True)
for j, p in enumerate(h):
    node(*p, R, f"h{'₁₂₃'[j]}", fill="#fff7ed" if j == 0 else "#ffffff")
for i, p in enumerate(xs):
    node(*p, R, f"x{'₁₂'[i]}")
node(*yk, R, "ŷ₁")

# 矢印のラベル（to-from の順：w_{行き先, 出発点}）
text(270, 172, sym("w", "hh", "11"), color=C_HH)
text(270, 230, sym("w", "hh", "12"), color=C_HH)
text(270, 322, sym("w", "hh", "13"), color=C_HH)
text(394, 322, sym("w", "hx", "11"), anchor="end", color=C_HX)
text(470, 322, sym("w", "hx", "12"), anchor="start", color=C_HX)
text(476, 160, sym("w", "yh", "11"), anchor="end", color=C_YH)
text(568, 166, sym("w", "yh", "12"), anchor="start", color=C_YH)
text(644, 160, sym("w", "yh", "13"), anchor="start", color=C_YH)

# 式
f1 = (sym("h", "(t)", "1") + " = σ( "
      + f'<tspan fill="{C_HX}">' + sym("w", "hx", "11") + " " + sym("x", "(t)", "1")
      + " + " + sym("w", "hx", "12") + " " + sym("x", "(t)", "2") + "</tspan>"
      + " + "
      + f'<tspan fill="{C_HH}">' + sym("w", "hh", "11") + " " + sym("h", "(t−1)", "1")
      + " + " + sym("w", "hh", "12") + " " + sym("h", "(t−1)", "2")
      + " + " + sym("w", "hh", "13") + " " + sym("h", "(t−1)", "3") + "</tspan>"
      + " + " + sym("b", None, "1") + " )")
text(450, 448, f1, size=16)
f2 = (sym("ŷ", "(t)", "1") + " = " + sym("l", None, "y") + "( "
      + f'<tspan fill="{C_YH}">' + sym("w", "yh", "11") + " " + sym("h", "(t)", "1")
      + " + " + sym("w", "yh", "12") + " " + sym("h", "(t)", "2")
      + " + " + sym("w", "yh", "13") + " " + sym("h", "(t)", "3") + "</tspan>"
      + " + " + sym("b", None, "y") + " )")
text(450, 482, f2, size=16)
text(450, 512, "添字は to-from の順：" + sym("w", None, "jj′", size=14)
     + " は「ノード j′ から j へ」の重み。h₂, h₃ に入る矢印も同じように付く（省略）",
     size=14, color="#475569")

add('<line x1="30" y1="534" x2="870" y2="534" stroke="#cbd5e1" stroke-width="1"/>')

# ---------------- (b) 時間方向に展開 ----------------
text(30, 568, "(b) 時間方向に展開（(a) の矢印の束を行列 W にまとめた）", size=17, anchor="start", weight="bold")

cols = [110, 320, 540, 760]   # h(0), t=1, t=2, t=3
HY0, HY1 = 690, 766           # 隠れ層の箱
OY0, OY1 = 600, 656           # 出力の箱
IY0, IY1 = 810, 880           # 入力の箱


def box(cx, y0, y1, w, color, fill="#ffffff", dashed=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    add(f'<rect x="{cx - w/2}" y="{y0}" width="{w}" height="{y1 - y0}" rx="10" '
        f'fill="{fill}" stroke="{color}" stroke-width="2"{dash}/>')


for n, cx in enumerate(cols):
    t = str(n)
    # 隠れ層
    is0 = n == 0
    box(cx, HY0, HY1, 130, C_PREV if is0 else C_NODE, dashed=is0)
    text(cx, HY0 + 24, sym("h", f"({t})", size=16), size=16, color=C_PREV if is0 else "#0f172a")
    for k in range(3):
        node(cx - 35 + 35 * k, HY0 + 52, 10, "", color=C_PREV if is0 else C_NODE)
    if is0:
        text(cx, HY1 + 26, "初期状態", size=14, color="#475569")
        continue
    # 出力
    box(cx, OY0, OY1, 80, C_NODE)
    text(cx, OY0 + 22, sym("ŷ", f"({t})", size=15), size=15)
    node(cx, OY0 + 40, 8, "")
    # 入力
    box(cx, IY0, IY1, 100, C_NODE)
    text(cx, IY0 + 24, sym("x", f"({t})", size=15), size=15)
    node(cx - 20, IY0 + 50, 9, "")
    node(cx + 20, IY0 + 50, 9, "")
    # 矢印
    arrow((cx, IY0), (cx, HY1), 0, 0, C_HX, width=3)
    text(cx - 16, (IY0 + HY1) / 2 + 6, sym("W", "hx"), anchor="end", color=C_HX)
    arrow((cx, HY0), (cx, OY1), 0, 0, C_YH, width=3)
    text(cx - 16, (HY0 + OY1) / 2 + 6, sym("W", "yh"), anchor="end", color=C_YH)
    # 前の時刻から
    px = cols[n - 1]
    arrow((px + 65, (HY0 + HY1) / 2), (cx - 65, (HY0 + HY1) / 2), 0, 0, C_HH, dashed=True, width=3)
    text((px + cx) / 2, (HY0 + HY1) / 2 - 10, sym("W", "hh"), color=C_HH)
    text(cx, IY1 + 26, f"t = {t}", size=15, weight="bold")

text(450, 936, "どの時刻でも同じ "
     + f'<tspan fill="{C_HX}">' + sym("W", "hx") + "</tspan>, "
     + f'<tspan fill="{C_HH}">' + sym("W", "hh") + "</tspan>, "
     + f'<tspan fill="{C_YH}">' + sym("W", "yh") + "</tspan>"
     + " を使う（重みの共有）。(a) の破線の矢印 9 本分をまとめたものが "
     + sym("W", "hh") + "（3×3 行列）", size=14, color="#475569")

add('</svg>')
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("\n".join(parts), encoding="utf-8")
print(OUT)

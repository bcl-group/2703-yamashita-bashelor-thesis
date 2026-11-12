"""フロー関数学習モデルのアーキテクチャ図（SVG）を生成する。

Aguiar et al. (L4DC 2024) の図1を踏襲し、公開実装（flow-learning の 2407-l4dc ブランチ,
flow_model/model.py）の構成を書き込んだもの。設定は §4.1 の HH-FS（run_paper.sh の fs）。

    初期状態 x = (V/100, n, m, h) ∈ R^4
      → 符号化器 E: 4→24→24→24→24（隠れ3層 tanh、出力層は線形）→ h_0（c_0 = 0）
      → LSTM（1層・隠れ24、入力 (ω_j, τ_j) ∈ R^2）を k_t + 1 = 4 回
      → 潜在表現の線形補間 z~ = (1 − τ_3) h_3 + τ_3 h_4      （model.py の h_shift）
      → 復号器 D: 24→24→24→24→1 → ŷ ≈ V/100

図の例は t = 35 ms（Δ = 10 ms なので k_t = 3, τ_3 = 0.5）。
"""
from pathlib import Path

# スクリプトと同じフォルダに保存する（フォルダを移動しても動くように）
OUT = Path(__file__).with_name("flow_architecture.svg")

W, H = 1290, 625
SHIFT = -58            # 上の余白を詰める（全体を上へずらす）

C_NET = "#2563eb"      # 符号化器・復号器（線）
C_NET_BG = "#dbeafe"   # 符号化器・復号器（塗り）
C_RNN = "#475569"      # LSTM セル（線）
C_RNN_BG = "#f1f5f9"   # LSTM セル（塗り）
C_INT = "#ea580c"      # 補間ブロックと補間に入る経路
C_INT_BG = "#ffedd5"
C_LINE = "#334155"     # 通常の信号
C_SUB = "#64748b"      # 補足の文字
C_TEXT = "#0f172a"
FONT = "'Noto Sans JP','Yu Gothic','Hiragino Sans',sans-serif"

parts = []


def add(s):
    parts.append(s)


def it(s):
    """斜体（変数名）。"""
    return f'<tspan font-style="italic">{s}</tspan>'


def sub(s, size):
    """下付き文字。後ろの文字のベースラインを元に戻す。"""
    small = round(size * 0.72)
    dy = size * 0.3
    return (f'<tspan dy="{dy:.1f}" font-size="{small}">{s}</tspan>'
            f'<tspan dy="{-dy:.1f}">&#8203;</tspan>')


def text(x, y, content, size=16, anchor="middle", color=C_TEXT, weight="normal"):
    add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-weight="{weight}" font-family="{FONT}">{content}</text>')


def box(x, y, w, h, stroke, fill, dash=None, rx=8, sw=2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(points, color=C_LINE, width=2, arrow=True, dash=None):
    """折れ線の矢印。points = [(x, y), ...]"""
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    marker = f' marker-end="url(#arr-{color[1:]})"' if arrow else ""
    ds = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{ds}{marker}/>')


def ztilde(x, y, size=16, color=C_TEXT):
    """z~ を描く（結合文字のチルダは位置がずれるので、記号を重ねて描く）。左端が x。"""
    text(x, y, it("z"), size=size, anchor="start", color=color)
    text(x + size * 0.36, y - size * 0.55, "~", size=size * 0.8, color=color)
    return x + size * 0.62     # 続きの文字を置く x


def h_(j, size=15):
    return it("h") + sub(j, size)


def c_(j, size=15):
    return it("c") + sub(j, size)


# ---------------------------------------------------------------- 配置
CY = 330                       # 横一列（x → 符号化器 → LSTM）の中心の高さ
CELL_W, CELL_H = 110, 76
CELL_X = [350 + 170 * j for j in range(4)]   # LSTM セルの左端
CELL_CX = [x + CELL_W / 2 for x in CELL_X]
TAUS = ["1", "1", "1", "0.5"]

# 区間 j を、セル j の上の [L_j, L_j + 170] に対応させる（10 ms ↔ 170 px）
SEG_L = [CELL_CX[j] - 85 for j in range(4)]
PX_PER_MS = 17.0

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
add('<defs>')
for col in (C_LINE, C_INT, C_SUB):
    add(f'<marker id="arr-{col[1:]}" viewBox="0 0 10 10" refX="9" refY="5" '
        f'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{col}"/></marker>')
add('</defs>')
add(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
add(f'<g transform="translate(0,{SHIFT})">')

# ---------------------------------------------------------------- 上段：区分定数の入力電流
AX_Y = 205                     # 時間軸の高さ
OMEGA_H = [42, 78, 24, 62]     # 各区間の電流値（絵の高さ。値そのものに意味はない）
text(SEG_L[0] - 12, 112, "入力電流 " + it("u") + "(" + it("t") + ")", size=15,
     anchor="end", color=C_SUB)
text(SEG_L[0] - 12, 132, "（Δ ごとに一定）", size=13, anchor="end", color=C_SUB)
# 波形は説明用。§4.1 の学習データの電流は 100 ms ごとに変わる（hhfs_flow.py の UniformSqWave）
text(SEG_L[0] - 12, 152, "※値は説明用。§4.1 の学習データ", size=11, anchor="end", color=C_SUB)
text(SEG_L[0] - 12, 167, "では 100 ms ごとに変化", size=11, anchor="end", color=C_SUB)
line([(SEG_L[0] - 6, AX_Y), (SEG_L[3] + 200, AX_Y)], color=C_SUB, width=1.5)
text(SEG_L[3] + 206, AX_Y + 5, it("t"), size=15, anchor="start", color=C_SUB)

for j in range(4):
    x0 = SEG_L[j]
    y = AX_Y - OMEGA_H[j]
    if j < 3:
        add(f'<path d="M {x0} {y} L {x0 + 170} {y}" stroke="{C_LINE}" stroke-width="2.5"/>')
    else:
        # 最後の区間：t = 35 ms までが使われ、残り 5 ms は使わない
        x_t = x0 + 5 * PX_PER_MS
        add(f'<path d="M {x0} {y} L {x_t} {y}" stroke="{C_LINE}" stroke-width="2.5"/>')
        add(f'<path d="M {x_t} {y} L {x0 + 170} {y}" stroke="{C_SUB}" stroke-width="1.5" '
            f'stroke-dasharray="4 4"/>')
    if j > 0:   # 区間の切り替わり（縦線）
        yp = AX_Y - OMEGA_H[j - 1]
        add(f'<path d="M {x0} {yp} L {x0} {y}" stroke="{C_LINE}" stroke-width="2.5"/>')
    text(x0 + 85, y - 10, it("ω") + sub(str(j), 16), size=16)

# 時間軸の目盛
for j, lab in enumerate(["0", "10", "20", "30"]):
    x0 = SEG_L[j]
    add(f'<path d="M {x0} {AX_Y - 4} L {x0} {AX_Y + 4}" stroke="{C_SUB}" stroke-width="1.5"/>')
    text(x0, AX_Y + 22, lab, size=13, color=C_SUB)
text(SEG_L[3] + 170, AX_Y + 22, "40 ms", size=13, color=C_SUB)
add(f'<path d="M {SEG_L[3] + 170} {AX_Y - 4} L {SEG_L[3] + 170} {AX_Y + 4}" '
    f'stroke="{C_SUB}" stroke-width="1.5"/>')

# 予測したい時刻 t = 35 ms
x_t = SEG_L[3] + 5 * PX_PER_MS
add(f'<path d="M {x_t} {AX_Y - 110} L {x_t} {AX_Y + 4}" stroke="{C_INT}" stroke-width="1.5" '
    f'stroke-dasharray="5 4"/>')
text(x_t, AX_Y - 116, "予測する時刻 " + it("t") + " = 35 ms", size=14, color=C_INT)

# Δ の幅
yb = AX_Y + 36
add(f'<path d="M {SEG_L[0]} {yb - 5} L {SEG_L[0]} {yb + 5} M {SEG_L[0]} {yb} '
    f'L {SEG_L[0] + 170} {yb} M {SEG_L[0] + 170} {yb - 5} L {SEG_L[0] + 170} {yb + 5}" '
    f'stroke="{C_SUB}" stroke-width="1.2" fill="none"/>')
text(SEG_L[0] + 85, yb - 5, "Δ = 10 ms", size=12, color=C_SUB)

# ---------------------------------------------------------------- 左：初期状態と符号化器
box(12, CY - 46, 110, 92, C_LINE, "#ffffff")
text(67, CY - 18, "初期状態", size=15)
text(67, CY + 8, it("x") + " ∈ ℝ⁴", size=17)
text(67, CY + 32, "(" + it("V") + "/100, " + it("n") + ", " + it("m") + ", " + it("h") + ")",
     size=11, color=C_SUB)

ENC_X = 145
box(ENC_X, CY - 52, 130, 104, C_NET, C_NET_BG)
text(ENC_X + 65, CY - 24, "符号化器 " + it("E"), size=15, weight="bold", color=C_NET)
text(ENC_X + 65, CY + 2, "4→24→24→24→24", size=13)
text(ENC_X + 65, CY + 22, "隠れ3層：tanh", size=11, color=C_SUB)
text(ENC_X + 65, CY + 38, "出力層：線形", size=11, color=C_SUB)
line([(122, CY), (ENC_X - 2, CY)])

# ---------------------------------------------------------------- 中央：時間方向に展開した LSTM
# 重み共有の囲み
box(CELL_X[0] - 8, CY - 70, CELL_X[3] + CELL_W + 22 - (CELL_X[0] - 8), 140,
    C_RNN, "none", dash="6 5", rx=12, sw=1.3)
text(CELL_X[0] + 2, CY - 47, "LSTM：1層・隠れ状態 24 次元・4 セルとも同じ重み", size=13,
     anchor="start", color=C_RNN)

for j in range(4):
    x = CELL_X[j]
    box(x, CY - CELL_H / 2, CELL_W, CELL_H, C_RNN, C_RNN_BG)
    text(x + CELL_W / 2, CY + 6, "LSTM", size=16, weight="bold", color=C_RNN)

    # 下からの入力 (ω_j, τ_j)
    cx = CELL_CX[j]
    col = C_INT if j == 3 else C_TEXT
    text(cx, CY + 104, "(" + it("ω") + sub(str(j), 15) + ", " + it("τ") + sub(str(j), 15)
         + ")", size=15, color=col)
    text(cx, CY + 124, it("τ") + sub(str(j), 13) + " = " + TAUS[j], size=13,
         color=C_INT if j == 3 else C_SUB)
    line([(cx, CY + 84), (cx, CY + CELL_H / 2 + 2)])

# セル間の状態 (h_j, c_j)
line([(ENC_X + 130, CY), (CELL_X[0] - 2, CY)])
text((ENC_X + 130 + CELL_X[0]) / 2 + 4, CY - 12, "(" + h_("0") + ", " + c_("0") + ")",
     size=15)
text((ENC_X + 130 + CELL_X[0]) / 2 + 4, CY + 22, it("c") + sub("0", 12) + " = 0", size=12,
     color=C_SUB)
for j in range(3):
    xa, xb = CELL_X[j] + CELL_W, CELL_X[j + 1]
    line([(xa, CY), (xb - 2, CY)])
    text((xa + xb) / 2, CY - 12, "(" + h_(str(j + 1)) + ", " + c_(str(j + 1)) + ")", size=14)

# ---------------------------------------------------------------- 右下：線形補間と復号器
INT_X, INT_Y, INT_W, INT_H = 780, 500, 315, 92
box(INT_X, INT_Y, INT_W, INT_H, C_INT, C_INT_BG)
text(INT_X + INT_W / 2, INT_Y + 30, "潜在表現の線形補間", size=15, weight="bold", color=C_INT)
x_eq = ztilde(INT_X + 26, INT_Y + 62, size=16)
text(x_eq, INT_Y + 62, "&#160;= (1 − " + it("τ") + sub("3", 16) + ") " + h_("3", 16) + " + "
     + it("τ") + sub("3", 16) + " " + h_("4", 16) + "  ∈ ℝ²⁴", size=16, anchor="start")

# h_3 をセル3とセル4の間から分岐させる
x_h3 = (CELL_X[2] + CELL_W + CELL_X[3]) / 2
add(f'<circle cx="{x_h3}" cy="{CY}" r="4" fill="{C_INT}"/>')
line([(x_h3, CY), (x_h3, INT_Y - 2)], color=C_INT)
text(x_h3 - 8, INT_Y - 52, h_("3"), size=15, anchor="end", color=C_INT)

# h_4 はセル4の出力
x_r = CELL_X[3] + CELL_W
line([(x_r, CY), (x_r + 75, CY), (x_r + 75, INT_Y - 2)], color=C_INT)
text(x_r + 40, CY - 12, "(" + h_("4") + ", " + c_("4") + ")", size=14)
text(x_r + 83, INT_Y - 52, h_("4"), size=15, anchor="start", color=C_INT)

# τ_3 は最後のセルと補間の両方に入る
x_tau = CELL_CX[3] + 26
line([(x_tau, CY + 130), (x_tau, INT_Y - 2)], color=C_INT, width=1.6, dash="5 4")

# 復号器
DEC_X = 1130
box(DEC_X, INT_Y - 6, 140, 104, C_NET, C_NET_BG)
text(DEC_X + 70, INT_Y + 22, "復号器 " + it("D"), size=15, weight="bold", color=C_NET)
text(DEC_X + 70, INT_Y + 48, "24→24→24→24→1", size=13)
text(DEC_X + 70, INT_Y + 68, "隠れ3層：tanh", size=11, color=C_SUB)
text(DEC_X + 70, INT_Y + 84, "出力層：線形", size=11, color=C_SUB)
line([(INT_X + INT_W, INT_Y + INT_H / 2), (DEC_X - 2, INT_Y + INT_H / 2)])
ztilde((INT_X + INT_W + DEC_X) / 2 - 6, INT_Y + INT_H / 2 - 10, size=15)

# 出力
line([(DEC_X + 70, INT_Y + 98), (DEC_X + 70, INT_Y + 128)])
text(DEC_X + 70, INT_Y + 150, it("ŷ") + "(" + it("t") + ", " + it("x") + ", " + it("u") + ")",
     size=17)
text(DEC_X + 70, INT_Y + 172, "≈ " + it("V") + "(" + it("t") + ") / 100", size=14,
     color=C_SUB)

# ---------------------------------------------------------------- 左下：記号の説明
LX, LY = 30, 520
text(LX, LY, "記号", size=14, anchor="start", weight="bold")
legend = [
    it("ω") + sub("j", 13) + "：区間 " + it("j") + " の電流値",
    it("τ") + sub("j", 13) + "：区間 " + it("j") + " のうち進める割合（0〜1）",
    "(" + it("h") + sub("j", 13) + ", " + it("c") + sub("j", 13) + ")：" + it("j")
    + " 回更新した後の LSTM の隠れ状態・セル状態（各 24 次元）",
    it("h") + sub("4", 13) + " は (" + it("ω") + sub("3", 13) + ", 0.5) で更新した結果"
    + "（40 ms の状態ではない）",
    it("k") + sub("t", 13) + " = ⌊" + it("t") + "/Δ⌋ = 3 → セルは " + it("k") + sub("t", 13)
    + " + 1 = 4 個",
    it("τ") + sub("3", 13) + " = (35 − 3·10)/10 = 0.5",
]
for i, s in enumerate(legend):
    text(LX, LY + 26 + 24 * i, s, size=13, anchor="start", color=C_SUB)

add('</g>')
add('</svg>')
OUT.write_text("\n".join(parts), encoding="utf-8")
print(f"saved: {OUT}")

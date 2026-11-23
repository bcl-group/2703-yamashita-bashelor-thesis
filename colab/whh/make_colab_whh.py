"""
Google Colab で hh_mlp_whh.py（Markov MLP に再帰の重み W_hh を足した RNN 版）のジョブを回す一式を作る。
    colab/whh/bundle.zip         : コード（experiments/hh_surrogate、src）と出発点のモデル（mm_rel、sp_r_scr_e01、sp_r_init）
    colab/whh/run_hh_whh.ipynb   : Drive をつなぎ、bundle.zip を展開し、ノートブック内の JOBS を GPU で回す
使い方:
    1. リポジトリのルートで  uv run python -X utf8 colab/whh/make_colab_whh.py
    2. Drive に MyDrive/hh_whh_colab/ を作り、bundle.zip を置く
    3. run_hh_whh.ipynb を Colab で開き（ランタイムは GPU）、上から実行する
結果は MyDrive/hh_whh_colab/out/ に出る（<tag>.pt、<tag>.json、ft は <tag>_whh.json、ログは out/logs/）。
終わったジョブ（out/<tag>.json がある）は飛ばすので、接続が切れたら同じセルを実行し直せば続きから回る。
"""
import glob, json, os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
files = (glob.glob('experiments/hh_surrogate/*.py', root_dir=ROOT)
         + glob.glob('src/**/*.py', root_dir=ROOT, recursive=True)
         + ['results/hh_improve/mm_rel.pt', 'results/hh_improve/sp_r_scr_e01.pt', 'results/hh_improve/sp_r_init.pt'])
with zipfile.ZipFile(os.path.join(OUT, 'bundle.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(os.path.join(ROOT, f), f.replace(os.sep, '/'))
print(len(files), 'files ->', os.path.join(OUT, 'bundle.zip'))

# 1 行 = hh_mlp_whh.py の引数。--device cuda と --out は実行時に足す
JOBS = r'''# 閉ループ微調整（CPU では 1 回 200 秒で回せなかったもの）。系列 50 ms × 300 回、検証は 50 回ごと
ft --tag cft_r_mm --iters 300 --eval-every 50 --ep-ms 50 --log-every 10
ft --tag cft_m_mm --ctrl --iters 300 --eval-every 50 --ep-ms 50 --log-every 10
ft --tag cft_r_e01 --src results/hh_improve/sp_r_scr_e01.pt --iters 300 --eval-every 50 --ep-ms 50 --log-every 10
# 系列版の摂動事前学習の seed 違い（手元の seed 0 と同じ条件。ばらつきを見る）
pre --tag sp_r_scr_e01_s1 --seed 1 --init scratch --cell mlpr --eps0 0.1 --iters 30000 --lr 1e-3 --eval-every 5000
pre --tag sp_r_scr_e01_s2 --seed 2 --init scratch --cell mlpr --eps0 0.1 --iters 30000 --lr 1e-3 --eval-every 5000
pre --tag sp_r_init_s1 --seed 1 --init mm_rel --cell mlpr --eps0 0 --iters 20000 --lr 3e-4 --eval-every 5000
pre --tag sp_r_init_s2 --seed 2 --init mm_rel --cell mlpr --eps0 0 --iters 20000 --lr 3e-4 --eval-every 5000
'''

setup = r"""import os, subprocess
D = '/content/drive/MyDrive/hh_whh_colab'   # bundle.zip を置いたフォルダ
R = '/content/repo'
os.makedirs(D + '/out/logs', exist_ok=True)
os.makedirs(D + '/cache', exist_ok=True)
subprocess.run(f'rm -rf {R} && mkdir -p {R} && cd {R} && unzip -q {D}/bundle.zip', shell=True, check=True)
# 採点の正解（HH の解）のキャッシュを Drive に置き、2 回目から計算を省く
for f in ('hh_bench_reference.npz', 'hh_bench_val.npz'):
    os.symlink(f'{D}/cache/{f}', f'{R}/results/{f}')
subprocess.run('pip -q install efel', shell=True)
subprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv', shell=True)
print('CPU コア数:', os.cpu_count())"""

run = r"""# 1 ステップずつ Python で回すので CPU（カーネルの起動）で律速される。Colab の CPU は 2 コアなので同時に 2 本まで
PARALLEL = 2
JOBS = '''__JOBS__'''
import shlex, sys, time
from concurrent.futures import ThreadPoolExecutor
lines = [l.strip() for l in JOBS.splitlines() if l.strip() and not l.startswith('#')]
tag_of = lambda l: shlex.split(l)[shlex.split(l).index('--tag') + 1]
# 正解のキャッシュを 1 つのプロセスで先に作る（同時に書かないように）
subprocess.run([sys.executable, '-X', 'utf8', '-c', "import sys; sys.path.insert(0, 'experiments/hh_surrogate'); "
                "import hh_bench as hb; hb.reference(); hb.reference(val=True)"], cwd=R, check=True)

def go(l):
    t = tag_of(l)
    if os.path.exists(f'{D}/out/{t}.json'):
        return t + ' (済)'
    t0 = time.time()
    with open(f'{D}/out/logs/{t}.log', 'w') as f:
        r = subprocess.run([sys.executable, '-u', '-X', 'utf8', 'experiments/hh_surrogate/hh_mlp_whh.py']
                           + shlex.split(l) + ['--device', 'cuda', '--out', f'{D}/out'],
                           cwd=R, stdout=f, stderr=subprocess.STDOUT)
    return f'{t} rc={r.returncode} {time.time() - t0:.0f} s'

print(len(lines), 'jobs,', PARALLEL, 'parallel', flush=True)
with ThreadPoolExecutor(PARALLEL) as ex:
    for s in ex.map(go, lines):
        print(s, flush=True)""".replace('__JOBS__', JOBS)

progress = r"""# 途中経過（ログの最後の数行）
import glob
for p in sorted(glob.glob(f'{D}/out/logs/*.log')):
    print('==', os.path.basename(p))
    print(''.join(open(p, encoding='utf-8', errors='replace').readlines()[-3:]))"""

summary = r"""# 結果の一覧（テスト 26 本の全部一致した本数、Γ、検証で選んだ回）
import glob, json
for p in sorted(glob.glob(f'{D}/out/*.json')):
    d = json.load(open(p, encoding='utf-8'))
    if isinstance(d, dict) and 'test' in d:
        print(f"{os.path.basename(p):28s} {d['test']['n100']:2d}/26  Γ {d['test']['gamma']:.3f}  val best {d['best']}")"""

cells = [
    ("markdown", "# hh_mlp_whh（RNN 版 Markov MLP）を Colab の GPU で回す\n"
                 "ランタイムを GPU にして上から順に実行。Drive の `MyDrive/hh_whh_colab/` に `bundle.zip` を置いておく。\n"
                 "終わったジョブ（`out/<tag>.json` がある）は飛ばすので、切れたら実行のセルからやり直せばよい。"),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
    ("code", setup),
    ("code", run),
    ("code", progress),
    ("code", summary),
]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(os.path.join(OUT, 'run_hh_whh.ipynb'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook ->', os.path.join(OUT, 'run_hh_whh.ipynb'))

"""
Google Colab で hh_rnn3.py のジョブを回すための一式を作る。
    colab/bundle.zip      : 学習に要るコード（experiments/hh_surrogate、src）と初期値のブロック
    colab/run_hh_rnn3.ipynb : Drive をつなぎ、bundle.zip を展開し、ジョブ一覧を GPU で回すノートブック
Drive のフォルダ MyDrive/hh_rnn3_colab/ に bundle.zip・ノートブック・run_jobs.py・jobs_*.txt を置いて使う。結果は同じフォルダの out/ に出る。
使い方（リポジトリのルートで）: uv run python -X utf8 colab/make_colab.py
"""
import glob, json, os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'colab')
files = (glob.glob('experiments/hh_surrogate/*.py', root_dir=ROOT)
         + [p for p in glob.glob('src/**/*.py', root_dir=ROOT, recursive=True)]
         + ['results/hh_modular_ou/block1.npz', 'results/hh_modular_ou/block2.npz',
            'results/hh_modular_ou/block3.npz', 'results/hh_improve/k_3_w3_b3.pt'])
with zipfile.ZipFile(os.path.join(OUT, 'bundle.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(os.path.join(ROOT, f), f.replace(os.sep, '/'))
print(len(files), 'files ->', os.path.join(OUT, 'bundle.zip'))

cells = [
    ("markdown", "# hh_rnn3 を Colab の GPU で回す\n上から順に実行。ジョブ一覧は `JOBS`、同時に回す数は `PARALLEL`。終わったジョブ（out/<tag>.json がある）は飛ばす。"),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
    ("code", "import os, subprocess\nD = '/content/drive/MyDrive/hh_rnn3_colab'\nR = '/content/repo'\nos.makedirs(D + '/out/logs', exist_ok=True)\nsubprocess.run(f'rm -rf {R} && mkdir -p {R} && cd {R} && unzip -q {D}/bundle.zip', shell=True, check=True)\nos.makedirs(R + '/results', exist_ok=True)\nif not os.path.exists(R + '/results/hh_rnn3'):\n    os.symlink(D + '/out', R + '/results/hh_rnn3')\n# 参照（HH の正解）のキャッシュも Drive に置き、2 回目から計算を省く\nfor f in ('hh_bench_reference.npz', 'hh_bench_val.npz'):\n    if not os.path.exists(f'{R}/results/{f}'):\n        os.symlink(f'{D}/out/{f}', f'{R}/results/{f}')\nsubprocess.run('pip -q install efel', shell=True)\nsubprocess.run('nvidia-smi --query-gpu=name,memory.total --format=csv', shell=True)"),
    ("code", "# ジョブの実行部分は Drive の run_jobs.py（colab/run_jobs.py をアップロードしたもの）\nJOBS = 'jobs_colab_r1.txt'; PARALLEL = 3; exec(open(f'{D}/run_jobs.py', encoding='utf-8').read())"),
    ("code", "# 結果の一覧\nimport glob, json\nfor p in sorted(glob.glob(f'{D}/out/*.json')):\n    d = json.load(open(p, encoding='utf-8'))\n    if 'test' in d:\n        print(os.path.basename(p), d['test']['n100'], round(d['test']['gamma'], 3), 'val best', d['best'])"),
]
nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}},
      "cells": [{"cell_type": t, "metadata": {}, "source": s.splitlines(True),
                 **({"outputs": [], "execution_count": None} if t == "code" else {})} for t, s in cells]}
json.dump(nb, open(os.path.join(OUT, 'run_hh_rnn3.ipynb'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('notebook ->', os.path.join(OUT, 'run_hh_rnn3.ipynb'))

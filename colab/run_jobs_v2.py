# Colab のノートブックから exec で読み込む。前のセルで D（Drive のフォルダ）と R（展開先）を、
# このセルで JOBS（ジョブ一覧のファイル名）と PARALLEL（同時に回す数）を決めておく。
# 1 行 = hh_rnn3.py train の引数。out/<tag>.json がある（終わった）ジョブは飛ばす。
import os, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

lines = [l.strip() for l in open(f'{D}/{JOBS}', encoding='utf-8') if l.strip() and not l.startswith('#')]


def tag_of(l):
    a = shlex.split(l)
    return a[a.index('--tag') + 1]


# 参照（HH の正解）のキャッシュを 1 つのプロセスで先に作る（同時に書かないように）
subprocess.run([sys.executable, '-X', 'utf8', '-c',
                "import sys; sys.path.insert(0, 'experiments/hh_surrogate'); import hh_bench as hb; "
                "hb.reference(); hb.reference(val=True)"], cwd=R, check=True)


# 在庫（HH の軌道）を 1 つずつ先に作る。できていればすぐ終わる
for l in lines:
    if not os.path.exists(f'{D}/out/{tag_of(l)}.json'):
        r = subprocess.run([sys.executable, '-X', 'utf8', 'experiments/hh_surrogate/hh_rnn3.py', 'train']
                           + shlex.split(l) + ['--pools-only'], cwd=R, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        if out:
            print(tag_of(l), out[-500:], flush=True)


def go(l):
    t = tag_of(l)
    if os.path.exists(f'{D}/out/{t}.json'):
        return t + ' (済)'
    t0 = time.time()
    with open(f'{D}/out/logs/{t}.log', 'w') as f:
        r = subprocess.run([sys.executable, '-u', '-X', 'utf8', 'experiments/hh_surrogate/hh_rnn3.py', 'train']
                           + shlex.split(l), cwd=R, stdout=f, stderr=subprocess.STDOUT)
    return f'{t} rc={r.returncode} {time.time() - t0:.0f} s'


print(len(lines), 'jobs,', PARALLEL, 'parallel', flush=True)
with ThreadPoolExecutor(PARALLEL) as ex:
    for s in ex.map(go, lines):
        print(s, flush=True)

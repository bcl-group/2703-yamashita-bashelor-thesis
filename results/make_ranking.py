"""
HH 代理モデルの精度ランキングと速度ランキングを作る。

精度: テスト 26 本の平均 Γ（と 100% 一致の本数）。結果は各学習の json（test）と results/gamma/ から集める。
      検証 17 本の成績（best）がある学習はそれも並べる。
速度: results/hh_speed/<tag>.json（hh_speed.py bench の結果）。速度はネットの形だけで決まるので、
      アーキテクチャ（形）ごとに 1 本ずつ測ってあり、各モデルにはその形の速度を対応させる。

使い方（リポジトリのルートで）:
    uv run python -X utf8 results/make_ranking.py --speed cpu gpu_all
結果: results/ranking/accuracy.csv, speed_<tag>.csv, ranking.md, ranking.png
"""
import argparse
import csv
import glob
import json
import os

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT = os.path.join(ROOT, 'results', 'ranking')

# アーキテクチャ（表示名）と、その速度を測ったモデル（hh_speed.py の名前）
ARCH_SPEED = {
    'Markov MLP 128×2': 'mm_rel',
    'Markov MLP 128×2 + コンダクタンスの項': 'r8_mlp_hdf_s2',
    'Markov MLP 256×3': 'mm_w256',
    'MLP 3 個 128×2': 'm3_v2f05d_s1',
    'RNN 3 個 H50（並列）': 'k_3_w3',
    'RNN 3 個 H64（並列・直列）': 'rnn3:p_h64_cl05',
    'RNN 3 個 H128（並列・直列）': 'rnn3:s_h128_ft05',
    'RNN 3 個 GRU64': 'rnn3:s_gru64',
    'RNN 1 個（W_hh）128×2': 'rnn1:sp_r_init',
    'Markov MLP 2 本のアンサンブル': 'ens:mm_rel+mm_rel_s1',
    'HH 陽的 Euler（基準）': 'hh_euler',
}


def arch_of(src, tag, cfg=None):
    """学習の出どころ（src）とタグ・設定からアーキテクチャの表示名を決める。決められないものは None"""
    if tag.startswith('ens:') or tag.startswith('ens_'):
        return 'Markov MLP 2 本のアンサンブル' if tag.count('mm_') == 2 else None
    if src == 'hh_rnn3':
        if 'gru' in tag:
            return 'RNN 3 個 GRU64'
        return 'RNN 3 個 H128（並列・直列）' if 'h128' in tag else 'RNN 3 個 H64（並列・直列）'
    if tag.startswith('m3_'):
        return 'MLP 3 個 128×2'
    if tag.startswith('a3_') or tag.startswith(('e_s', 'e_p', 'c_s', 'c_p')):
        return 'RNN 3 個 H64（並列・直列）'   # RNN1・RNN2 が H50 / H64 の混在。速度は H64 の形で近似
    if tag == 'cl_m_mlp':
        return 'Markov MLP 128×2'
    if tag.startswith('sp_r_') or tag.startswith('cft_r'):
        return 'RNN 1 個（W_hh）128×2'
    if tag in ('hh_modular_ou',) or tag.startswith(('a3_', 'c123', 'k_')):
        return 'RNN 3 個 H50（並列）' if tag in ('hh_modular_ou', 'k_3_w3', 'k_3_w3_K2000', 'k_123_w3', 'k_123_w1',
                                                'c123_lr1e-6') else None
    if tag.startswith('r8_mlp_hdf') or tag.startswith('r8_mlp_hd_w256'):
        return 'Markov MLP 256×3' if 'w256' in tag else 'Markov MLP 128×2 + コンダクタンスの項'
    if tag in ('mm_w256', 'mm_w256_long', 'mm_w256_relu') or tag.startswith('g_relu256'):
        return 'Markov MLP 256×3'
    if tag.startswith(('mm_', 'r8_mlp', 'r9_mlp', 'g_silu', 'g_relu', 'sp_m_', 'cft_m')):
        return 'Markov MLP 128×2'
    return None


CATEGORY = {
    'Markov MLP 128×2': 'Markov MLP', 'Markov MLP 128×2 + コンダクタンスの項': 'Markov MLP',
    'Markov MLP 256×3': 'Markov MLP', 'Markov MLP 2 本のアンサンブル': 'Markov MLP（アンサンブル）',
    'MLP 3 個 128×2': 'MLP 3 個', 'RNN 1 個（W_hh）128×2': 'RNN 1 個',
    'RNN 3 個 H50（並列）': 'RNN 3 個', 'RNN 3 個 H64（並列・直列）': 'RNN 3 個', 'RNN 3 個 H128（並列・直列）': 'RNN 3 個',
    'RNN 3 個 GRU64': 'RNN 3 個', 'HH 陽的 Euler（基準）': 'HH',
}


def collect():
    rows = {}

    def add(key, src, tag, test, best=None):
        if not test or test.get('n100') is None:
            return
        rows[key] = {'key': key, 'src': src, 'tag': tag, 'arch': arch_of(src, tag),
                     'test_n100': test['n100'], 'test_mean': test.get('mean'), 'test_gamma': test.get('gamma'),
                     'val_n100': best[0] if best else None, 'val_mean': best[1] if best and len(best) > 1 else None}

    for p in glob.glob(os.path.join(ROOT, 'results', 'hh_improve', '*.json')):
        tag = os.path.basename(p)[:-5]
        try:
            d = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        if isinstance(d, dict) and isinstance(d.get('test'), dict):
            add(tag, 'hh_improve', tag, d['test'], d.get('best'))
    for p in glob.glob(os.path.join(ROOT, 'results', 'hh_rnn3', '*.json')):
        tag = os.path.basename(p)[:-5]
        d = json.load(open(p, encoding='utf-8'))
        if isinstance(d.get('test'), dict):
            add('rnn3/' + tag, 'hh_rnn3', tag, d['test'], d.get('best'))
    # results/gamma（採点し直した 26 本の Γ）: 同じタグがあれば Γ を上書き、なければ足す
    for p in glob.glob(os.path.join(ROOT, 'results', 'gamma', '*.json')):
        tag = os.path.basename(p)[:-5]
        d = json.load(open(p, encoding='utf-8'))
        if tag in rows:
            rows[tag]['test_gamma'] = d.get('gamma')
        else:
            add(tag, 'gamma', tag.replace('ens_mm_rel_mm_rel_s1_mm_sv05', 'ens:mm_rel+mm_rel_s1+mm_sv05')
                .replace('ens_mm_rel_mm_rel_s1', 'ens:mm_rel+mm_rel_s1'), d)
    for r in rows.values():
        if r['arch'] is None:
            r['arch'] = arch_of(r['src'], r['tag'])
    return [r for r in rows.values() if r['test_gamma'] is not None and r['tag'] != 'r3_smoke']


def load_speed(tag):
    p = os.path.join(ROOT, 'results', 'hh_speed', tag + '.json')
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding='utf-8'))


def speed_table(res):
    """モデル（hh_speed の名前）ごとに、B と回し方ごとの 1 ステップの時間 [µs] と neuron-s/s"""
    t = {}
    for r in res['rows']:
        if r['record'] or r.get('mm', 'fp32') != 'fp32':
            continue
        t.setdefault(r['model'], {})[(r['runner'], r['B'])] = (r['per_step_s'] * 1e6, r['neuron_s_per_s'])
    # 速度は形だけで決まる。サーバ（gpu_all）は RNN 1 個を同じ形の sp_r_scr_e01_s1 で測ったので、その値を使う
    if 'rnn1:sp_r_init' not in t and 'rnn1:sp_r_scr_e01_s1' in t:
        t['rnn1:sp_r_init'] = t['rnn1:sp_r_scr_e01_s1']
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', nargs='*', default=['cpu', 'gpu_all'])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rows = sorted(collect(), key=lambda r: (-r['test_gamma'], -r['test_n100']))
    with open(os.path.join(OUT, 'accuracy.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['rank', 'key', 'arch', 'test_gamma', 'test_n100', 'test_mean', 'val_n100',
                                          'val_mean', 'src'])
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({'rank': i, **{k: r.get(k) for k in w.fieldnames if k != 'rank'}})
    print('精度: {} モデル → {}'.format(len(rows), os.path.join(OUT, 'accuracy.csv')))

    speeds = {}
    for tag in a.speed:
        res = load_speed(tag)
        if res is None:
            print('速度の結果がない:', tag)
            continue
        speeds[tag] = (res, speed_table(res))
        with open(os.path.join(OUT, f'speed_{tag}.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            Bs = sorted({B for v in speeds[tag][1].values() for (_, B) in v})
            runners = sorted({rn for v in speeds[tag][1].values() for (rn, _) in v})
            w.writerow(['arch', 'model', 'flops_per_step'] + [f'{rn} B={B} [µs/step]' for rn in runners for B in Bs])
            for arch, m in ARCH_SPEED.items():
                v = speeds[tag][1].get(m, {})
                w.writerow([arch, m, res['flops'].get(m)] + [round(v[(rn, B)][0], 2) if (rn, B) in v else ''
                                                             for rn in runners for B in Bs])
    json.dump({'accuracy': rows}, open(os.path.join(OUT, 'accuracy.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=0)


if __name__ == '__main__':
    main()

#!/bin/bash
# jobs_eq.txt の 1 行（<rnn3|whh> <引数...>）を 1 回分の学習として回す。xargs -L 1 から呼ぶ
# 終了コードを見て done / FAILED を出す（失敗した学習を「終わった」と見誤らないため）
kind=$1; shift
tag=""; prev=""
for x in "$@"; do [ "$prev" = "--tag" ] && tag=$x; prev=$x; done
log=results/hh_eq/logs/$tag.log
case $kind in
    rnn3) uv run python -u -X utf8 experiments/hh_surrogate/hh_rnn3.py train "$@" > "$log" 2>&1 ;;
    whh)  uv run python -u -X utf8 experiments/hh_surrogate/hh_mlp_whh.py pre "$@" > "$log" 2>&1 ;;
    *)    echo "FAILED $tag: 種類 $kind が不明"; exit 1 ;;
esac
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAILED $tag $(date '+%T') rc=$rc: $(tail -n 1 "$log" | cut -c1-150)"
else
    echo "done $tag $(date '+%T'): $(grep '^\[' "$log" | tail -n 1 | cut -c1-150)"
fi
exit 0

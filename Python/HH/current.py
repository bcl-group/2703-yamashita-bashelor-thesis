import matplotlib.pyplot as plt
import numpy as np


T = 10.0  # 全体の時間（秒）
f = 1000  # サンプリング周波数 (Hz)
t = np.arange(0, T, 1 / f)  # 時間軸の配列
width = 0.05  # パルス幅（秒）
sample = int(width * f)  # パルス幅に相当するデータ数

I = np.zeros_like(t)

# 1秒間に平均して発生させたいパルス回数
rate = 2.5
trigger_prob = rate / f

i = 0
while i < len(t) - sample:
    # 確率的にパルスの発生タイミングを決定（ランダム・タイミング）
    if np.random.rand() < trigger_prob:
        # 振幅を 0.2A 〜 1.5A の間でランダムに決定（ランダム・振幅）
        amplitude = np.random.uniform(0.2, 1.5)

        # パルスを書き込む
        I[i : i + sample] = amplitude

        # パルス幅の期間は次のパルスが発生しないようにインデックスを進める
        i += sample
    else:
        i += 1


# グラフ描画
plt.rcParams["font.family"] = "sans-serif"

plt.figure(figsize=(10, 4))
plt.plot(t, I, color="purple", lw=1.5)
plt.title("タイミング ＆ 振幅の両方がランダムなパルス電流")
plt.xlabel("時間 (秒)")
plt.ylabel("電流 (A)")
plt.grid(True, linestyle="--", alpha=0.7)
plt.ylim(-0.2, 1.8)

plt.tight_layout()
plt.show()

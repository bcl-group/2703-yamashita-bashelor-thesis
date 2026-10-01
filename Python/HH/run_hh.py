import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from neuron.hh import simulate

from neuron.current import (
    step_current,
    constant_current,
    ramp_current,
    sin_current,
    pulse_train_current,
    random_pulse_current,
    random_timing_pulse_current,
)


if __name__ == '__main__':
    dt = 0.01
    T  = 50.0
    t  = np.arange(0, T, dt)

    I_ext = step_current(t, amplitude=10.0, t_start=10.0, t_end=40.0)
    # I_ext = constant_current(t, amplitude=10.0)
    # I_ext = ramp_current(t, i_start=0.0, i_end=20.0)
    # I_ext = sin_current(t, amplitude=10.0, freq=50.0, offset=10.0)
    # I_ext = pulse_train_current(t, amplitude=20.0, width=1.0, interval=10.0)
    # I_ext = random_timing_pulse_current(t, seed=0)
    # I_ext = random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=0)
    # I_ext = random_pulse_current(t, i_min=0.0, i_max=20.0, interval=10.0, seed=0)

    V_hist, m_hist, h_hist, n_hist = simulate(t, I_ext)

    peak_idx = np.where(
        (V_hist[1:-1] > V_hist[:-2]) & (V_hist[1:-1] > V_hist[2:])
    )[0] + 1

    top2_idx = peak_idx[np.argsort(V_hist[peak_idx])[-2:]]
    top2_idx = top2_idx[np.argsort(t[top2_idx])]

    print("極大値2つの時刻 t [ms] =", t[top2_idx])
    print("そのときの膜電位 V [mV] =", V_hist[top2_idx])

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.title("Hodgkin-Huxley Model")
    plt.plot(t, V_hist, 'b-', label="Membrane Potential (V)")
    plt.ylabel("Membrane Potential (mV)", color='b')
    plt.tick_params(axis='y', labelcolor='b')
    plt.grid(True)

    ax2 = plt.gca().twinx()
    ax2.plot(t, I_ext, 'r-', label="Input Current (I_ext)")
    ax2.set_ylabel(r"Input Current ($\mu$A/cm$^2$)", color='r')
    ax2.tick_params(axis='y', labelcolor='r')

    plt.subplot(2, 1, 2)
    plt.plot(t, m_hist, 'g-', label="m (Na activation)")
    plt.plot(t, h_hist, 'g--', label="h (Na inactivation)")
    plt.plot(t, n_hist, 'm-', label="n (K activation)")
    plt.xlabel("Time (ms)")
    plt.ylabel("Gating Probability")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

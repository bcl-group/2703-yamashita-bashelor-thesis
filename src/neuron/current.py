import numpy as np


def step_current(t, amplitude=10.0, t_start=10.0, t_end=40.0):
    I = np.zeros_like(t)
    I[(t >= t_start) & (t <= t_end)] = amplitude
    return I


def constant_current(t, amplitude=10.0):
    return np.full_like(t, amplitude)


def ramp_current(t, i_start=0.0, i_end=20.0):
    return np.linspace(i_start, i_end, len(t))


def sin_current(t, amplitude=10.0, freq=50.0, offset=0.0):
    return offset + amplitude * np.sin(2.0 * np.pi * freq * t / 1000.0)


def pulse_train_current(t, amplitude=10.0, width=1.0, interval=10.0, t_start=0.0):
    I = np.zeros_like(t)
    phase = np.mod(t - t_start, interval)
    I[(t >= t_start) & (phase < width)] = amplitude
    return I


def random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=None):
    rng = np.random.default_rng(seed)

    n_segments = int(np.ceil((t[-1] - t[0] + (t[1] - t[0])) / interval))

    amplitudes = rng.uniform(i_min, i_max, size=n_segments)

    segment_index = ((t - t[0]) / interval).astype(int)
    return amplitudes[segment_index]


def random_timing_pulse_current(t, i_min=5.0, i_max=30.0, width=5.0, rate=0.02, seed=None):
    rng = np.random.default_rng(seed)

    dt = t[1] - t[0]
    n_width = max(1, int(width / dt))
    trigger_prob = rate * dt

    I = np.zeros_like(t)

    i = 0
    while i < len(t) - n_width:
        if rng.random() < trigger_prob:
            I[i:i + n_width] = rng.uniform(i_min, i_max)
            i += n_width
        else:
            i += 1

    return I


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dt = 0.01
    t = np.arange(0, 200.0, dt)

    currents = [
        ('step',                step_current(t, amplitude=10.0, t_start=50.0, t_end=150.0)),
        ('constant',            constant_current(t, amplitude=10.0)),
        ('ramp',                ramp_current(t, i_start=0.0, i_end=20.0)),
        ('sin',                 sin_current(t, amplitude=10.0, freq=50.0, offset=10.0)),
        ('pulse train',         pulse_train_current(t, amplitude=20.0, width=1.0, interval=10.0)),
        ('random pulse',        random_pulse_current(t, i_min=-5.0, i_max=20.0, interval=20.0, seed=0)),
        ('random timing pulse', random_timing_pulse_current(t, seed=0)),
    ]

    fig, axes = plt.subplots(len(currents), 1, figsize=(9, 11), sharex=True)

    for ax, (name, I) in zip(axes, currents):
        ax.plot(t, I, linewidth=1.0)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

    axes[-1].set_xlabel('Time [ms]')
    fig.suptitle('Input currents defined in current.py')
    fig.tight_layout()
    plt.show()

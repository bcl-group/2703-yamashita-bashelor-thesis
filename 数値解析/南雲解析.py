import numpy as np  #NumPyライブラリ
import matplotlib as plt

# du/dt= c (- v + u -u3/3 + I(t))
# dv/dt= u - b v + a
# a、b、c はパラメータであり、ここでは a=0.7、b=0.8、c=10 に固定

a = 0.7
b = 0.8
c = 10

T = 10.0      # pulse period
w = 1.0       # pulse width
I0 = 0.5      # pulse strength

dt = 0.01
t_end = 200

# input pulse
def I(t):
    return I0 if (t % T) < w else 0.0

# FitzHugh-Nagumo equation
def f(u, v, t):
    du = c * (-v + u - u**3 / 3 + I(t))
    dv = u - b * v + a
    return du, dv

# time
t = np.arange(0, t_end, dt)

# state
u = np.zeros(len(t))
v = np.zeros(len(t))

# Euler method
for i in range(len(t) - 1):
    du, dv = f(u[i], v[i], t[i])

    u[i + 1] = u[i] + dt * du
    v[i + 1] = v[i] + dt * dv

# plot
plt.plot(t, u)
plt.xlabel("time")
plt.ylabel("u")
plt.title("FitzHugh-Nagumo")
plt.grid()
plt.show()

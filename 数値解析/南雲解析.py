import numpy as np
import matplotlib.pyplot as plt

a = 0.7
b = 0.8
c = 10

T = 10.0
w = 10.0
I0 = 0.1601

dt = 0.01
t_end = 50

def I(t):
    return I0 if (t % T) < w else 0.0

t = np.arange(0, t_end, dt)

u = np.zeros(len(t))
v = np.zeros(len(t))
I_values = np.zeros(len(t))

for i in range(len(t)-1):

    I_values[i] = I(t[i])

    du = c * (-v[i] + u[i] - u[i]**3/3 + I_values[i])
    dv = u[i] - b*v[i] + a

    u[i+1] = u[i] + dt*du
    v[i+1] = v[i] + dt*dv

I_values[-1] = I(t[-1])

# u(t)
plt.figure(figsize=(10,4))
plt.plot(t, u)
plt.xlabel("Time t[ms]")
plt.ylabel("u")
plt.title("Membrane Potential")
plt.grid()
plt.show()

# I(t)
plt.figure(figsize=(10,2))
plt.plot(t, I_values)
plt.xlabel("Time t[ms]")
plt.ylabel("I(t) [$\mu$ m]")
plt.title("Input Current")
plt.grid()
plt.show()

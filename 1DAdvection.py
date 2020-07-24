# To reproduce Fig.6.1&2 in LeVeque-FVMHP
# Please note there is a package associated with this textbook.
# Nonetheless, I have never opened it but wrote the code on my own.



import numpy as np
import matplotlib.pyplot as plt


def FiniteVolume(method, T, f, period = 1.0, u_bar = 1.0, m = 100, nu = 0.8): 

    dx = period / m
    dt = nu * dx / u_bar

    x_i = np.linspace(0, period - dx, m) + dx / 2

    Q_iInit = f(x_i)
    Q_i = Q_iInit

    t = 0.0

    while t < T:

        Q_im1 = np.append(Q_i[-1], Q_i[0 : -1])
        Q_ip1 = np.append(Q_i[1 : ], Q_i[0])

        if method == 'Upwind':
            sigma_i = np.zeros(Q_i.shape)

        elif method == 'Beam-Warming':        
            sigma_i = (Q_i - Q_im1) / dx

        elif method == 'Lax-Wendroff':        
            sigma_i = (Q_ip1 - Q_i) / dx

        elif method == 'Minimod':
            sigma_i = minimod((Q_i - Q_im1) / dx, (Q_ip1 - Q_i) / dx)

        elif method == 'Superbee':
            sigma_1 = minimod(2 * (Q_i - Q_im1) / dx, (Q_ip1 - Q_i) / dx)
            sigma_2 = minimod((Q_i - Q_im1) / dx, 2 * (Q_ip1 - Q_i) / dx)
            sigma_i = maxmod(sigma_1, sigma_2)

        elif method == 'MC limiter':
            sigma_i = minimod(minimod(2 * (Q_i - Q_im1) / dx, 2 * (Q_ip1 - Q_i) / dx), (Q_ip1 - Q_im1) / (2 * dx))
            
        else:
            print('Which method?')

        sigma_im1 = np.append(sigma_i[-1], sigma_i[0 : -1])

        Q_iNext = Q_i - nu * (Q_i - Q_im1) - (1 / 2) * nu * (dx - u_bar * dt) * (sigma_i - sigma_im1)
        Q_i = Q_iNext

        t = t + dt

    return (x_i, Q_i)



def minimod(a, b):

    slope = np.zeros(a.shape)

    for k in range(0, a.size):

        if (a[k] * b[k] > 0) & (np.absolute(a[k]) < np.absolute(b[k])):
            slope[k] = a[k]

        elif (a[k] * b[k] > 0) & (np.absolute(a[k]) > np.absolute(b[k])):
            slope[k] = b[k]

        else:
            slope[k] = 0
            
    return slope



def maxmod(a, b):

    slope = np.zeros(a.shape)

    for k in range(0, a.size):

        if (a[k] * b[k] > 0) & (np.absolute(a[k]) < np.absolute(b[k])):
            slope[k] = b[k]

        elif (a[k] * b[k] > 0) & (np.absolute(a[k]) > np.absolute(b[k])):
            slope[k] = a[k]

        else:
            slope[k] = 0
            
    return slope



fig = plt.figure()

Methods = ['Minimod', 'Superbee', 'MC limiter']
#Methods = ['Upwind', 'Lax-Wendroff', 'Beam-Warming']

Ts = (1, 5)

for (i, method) in enumerate(Methods):

    for (j, T) in enumerate(Ts):
        
        snapshot = fig.add_subplot(len(Methods), len(Ts), len(Ts) * i + j + 1)

        IC = lambda x: np.sin(np.pi * (x + 0.2)) ** 50 + (x >= 0.6) * (x <= 0.8)
        x = np.linspace(0, 1, 1001)
        snapshot.plot(x, IC(x))
     
        (x_i, Q_i) = FiniteVolume(method, T, IC)
        snapshot.plot(x_i, Q_i, 'bo', markerfacecolor = 'None', markersize = 2)
        snapshot.set_xlim((0, 1))
        snapshot.set_ylim((-0.5, 1.5))
        snapshot.set_title(method + ' at t = ' + str(T))

fig.tight_layout()
plt.show()


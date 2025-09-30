import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from matplotlib.animation import FuncAnimation
from scipy.interpolate import RegularGridInterpolator

def U_eq(rho):
    return u_max*(1-rho/rho_max)
def V_eq(rho, y):
    return v_max*(np.exp(-(b+y)/2)-np.exp(-(b-y)/2))*(1-rho/rho_max)
    
def FPK(m, n, o, dt, dx, dy, rho, u, v, d1, d2, sigma):
    for j in range(m-1):
        for i in range(n):
            for l in range(o):
                if i == n-1:
                    if l == 0:
                        rho[j+1, i, l] = 1/4*(rho[j, 0, l]+rho[j, i-1, l]+2*rho[j, i, l+1])-dt/(2*dx)*(rho[j, 0, l]*(u[j, 0, l]+d1[j, 0, l])-rho[j, i-1, l]*(u[j, i-1, l]+d1[j, i-1, l]))+sigma**2*dt/dy**2*(2*rho[j, i, l+1]-2*rho[j, i, l])
                    elif l==o-1:
                        rho[j+1, i, l] = 1/4*(rho[j, 0, l]+rho[j, i-1, l]+2*rho[j, i, l-1])-dt/(2*dx)*(rho[j, 0, l]*(u[j, 0, l]+d1[j, 0, l])-rho[j, i-1, l]*(u[j, i-1, l]+d1[j, i-1, l]))+sigma**2*dt/dy**2*(2*rho[j, i, l-1]-2*rho[j, i, l])
                    else:
                        rho[j+1, i, l] = 1/4*(rho[j, 0, l]+rho[j, i-1, l]+rho[j, i, l+1]+rho[j, i, l-1])-dt/(2*dx)*(rho[j, 0, l]*(u[j, 0, l]+d1[j, 0, l])-rho[j, i-1, l]*(u[j, i-1, l]+d1[j, i-1, l]))-dt/(2*dy)*(rho[j, i, l+1]*(v[j, i, l+1]+d1[j, i, l+1])-rho[j, i, l-1]*(v[j, i, l-1]+d2[j, i, l-1]))+sigma**2*dt/(dy**2)*(rho[j, i, l+1]-2*rho[j, i, l]+rho[j, i, l-1])
                else:
                    if l == 0:
                        rho[j+1, i, l] = 1/4*(rho[j, i+1, l]+rho[j, i-1, l]+2*rho[j, i, l+1])-dt/(2*dx)*(rho[j, i+1, l]*(u[j, i+1, l]+d1[j, i+1, l])-rho[j, i-1, l]*(u[j, i-1, l]+d1[j, i-1, l]))+sigma**2*dt/dy**2*(2*rho[j, i, l+1]-2*rho[j, i, l])
                    elif l==o-1:
                        rho[j+1, i, l] = 1/4*(rho[j, i+1, l]+rho[j, i-1, l]+2*rho[j, i, l-1])-dt/(2*dx)*(rho[j, i+1, l]*(u[j, i+1, l]+d1[j, i+1, l])-rho[j, i-1, l]*(u[j, i-1, l]+d1[j, i-1, l]))+sigma**2*dt/dy**2*(2*rho[j, i, l-1]-2*rho[j, i, l])
                    else:
                        rho[j+1, i, l] = 1/4*(rho[j, i+1, l]+rho[j, i-1, l]+rho[j, i, l+1]+rho[j, i, l-1])-dt/(2*dx)*(rho[j, i+1, l]*(u[j, i+1, l]+d1[j, i+1, l])-rho[j, i-1, l]*(u[j, i-1, l]+d1[j, i-1, l]))-dt/(2*dy)*(rho[j, i, l+1]*(v[j, i, l+1]+d2[j, i, l+1])-rho[j, i, l-1]*(v[j, i, l-1]+d2[j, i, l-1]))+sigma**2*dt/(dy**2)*(rho[j, i, l+1]-2*rho[j, i, l]+rho[j, i, l-1])
    return rho

                        
def HJBI(m, n, o, b, dt, dx, dy, rho, u, v, d1, d2, C, sigma, k = 100):
    y = np.linspace(-b, b, o)
    for j in range(m-2, -1, -1):
        for i in range(n):
            for l in range(o):
                if i == n-1:
                    if l == 0:
                        alpha2 = V_eq(rho[j+1, i, l], y[l])/(1+rho[j+1, i, l]/(v_max*rho_max))
                        C_x = (C[j+1, 0, l]-C[j+1, i-1, l])/(2*dx)
                        alpha1 = U_eq(rho[j+1, i, l])-C_x
                        C[j, i, l] = C[j+1, i, l]+dt/2*(U_eq(rho[j+1, i, l])-alpha1)**2+dt/2*(V_eq(rho[j+1, i, l], y[l])-alpha2)**2+dt/(2*v_max*rho_max)*alpha2**2*rho[j+1, i, l]-dt/(4*k**2)*(C_x**2)+dt*(alpha1-C_x/(2*k**2))*C_x+dt*sigma**2/(2*dy**2)*(2*C[j+1, i, l+1]-2*C[j+1, i, l])
                    elif l == o-1:
                        alpha2 = V_eq(rho[j+1, i, l], y[l])/(1+rho[j+1, i, l]/(v_max*rho_max))
                        C_x = (C[j+1, 0, l]-C[j+1, i-1, l])/(2*dx)
                        alpha1 = U_eq(rho[j+1, i, l])-C_x
                        C[j, i, l] = C[j+1, i, l]+dt/2*(U_eq(rho[j+1, i, l])-alpha1)**2+dt/2*(V_eq(rho[j+1, i, l], y[l])-alpha2)**2+dt/(2*v_max*rho_max)*alpha2**2*rho[j+1, i, l]-dt/(4*k**2)*(C_x**2)+dt*(alpha1-C_x/(2*k**2))*C_x+dt*sigma**2/(2*dy**2)*(2*C[j+1, i, l-1]-2*C[j+1, i, l])
                    else:
                        C_x = (C[j+1, 0, l]-C[j+1, i-1, l])/(2*dx)
                        C_y = (C[j+1, i, l+1]-C[j+1, i, l-1])/(2*dy)
                        alpha1 = U_eq(rho[j+1, i, l])-C_x
                        alpha2 = (V_eq(rho[j+1, i, l], y[l])-C_y)/(1+rho[j+1, i, l]/(v_max*rho_max))
                        C[j, i, l] = C[j+1, i, l]+dt/2*(U_eq(rho[j+1, i, l])-alpha1)**2+dt/2*(V_eq(rho[j+1, i, l], y[l])-alpha2)**2+dt/(2*v_max*rho_max)*alpha2**2*rho[j+1, i, l]-dt/(4*k**2)*(C_x**2+C_y**2)+dt*(alpha1+1/(2*k**2)*C_x)*C_x+dt*(alpha2+1/(2*k**2)*C_y)*C_y+dt*sigma**2/(2*dy**2)*(C[j+1, i, l+1]-2*C[j+1, i, l]+C[j+1, i, l-1])
                else:
                    if l == 0:
                        alpha2 = V_eq(rho[j+1, i, l], y[l])/(1+rho[j+1, i, l]/(v_max*rho_max))
                        C_x = (C[j+1, i+1, l]-C[j+1, i-1, l])/(2*dx)
                        alpha1 = U_eq(rho[j+1, i, l])-C_x
                        C[j, i, l] = C[j+1, i, l]+dt/2*(U_eq(rho[j+1, i, l])-alpha1)**2+dt/2*(V_eq(rho[j+1, i, l], y[l])-alpha2)**2+dt/(2*v_max*rho_max)*alpha2**2*rho[j+1, i, l]-dt/(4*k**2)*(C_x**2)+dt*(alpha1-C_x/(2*k**2))*C_x+dt*sigma**2/(2*dy**2)*(2*C[j+1, i, l+1]-2*C[j+1, i, l])
                    elif l == o-1:
                        alpha2 = V_eq(rho[j+1, i, l], y[l])/(1+rho[j+1, i, l]/(v_max*rho_max))
                        C_x = (C[j+1, i+1, l]-C[j+1, i-1, l])/(2*dx)
                        alpha1 = U_eq(rho[j+1, i, l])-C_x
                        C[j, i, l] = C[j+1, i, l]+dt/2*(U_eq(rho[j+1, i, l])-alpha1)**2+dt/2*(V_eq(rho[j+1, i, l], y[l])-alpha2)**2+dt/(2*v_max*rho_max)*alpha2**2*rho[j+1, i, l]-dt/(4*k**2)*(C_x**2)+dt*(alpha1-C_x/(2*k**2))*C_x+dt*sigma**2/(2*dy**2)*(2*C[j+1, i, l-1]-2*C[j+1, i, l])
                    else:
                        C_x = (C[j+1, i+1, l]-C[j+1, i-1, l])/(2*dx)
                        C_y = (C[j+1, i, l+1]-C[j+1, i, l-1])/(2*dy)
                        alpha1 = U_eq(rho[j+1, i, l])-C_x
                        alpha2 = (V_eq(rho[j+1, i, l], y[l])-C_y)/(1+rho[j+1, i, l]/(v_max*rho_max))
                        C[j, i, l] = C[j+1, i, l]+dt/2*(U_eq(rho[j+1, i, l])-alpha1)**2+dt/2*(V_eq(rho[j+1, i, l], y[l])-alpha2)**2+dt/(2*v_max*rho_max)*alpha2**2*rho[j+1, i, l]-dt/(4*k**2)*(C_x**2+C_y**2)+dt*(alpha1+1/(2*k**2)*C_x)*C_x+dt*(alpha2+1/(2*k**2)*C_y)*C_y+dt*sigma**2/(2*dy**2)*(C[j+1, i, l+1]-2*C[j+1, i, l]+C[j+1, i, l-1])

    for j in range(m):
        for i in range(n):
            for l in range(o):
                if i == n-1:
                    if l == 0 or l == o-1:
                        C_x = (C[j, 0, l]-C[j, i-1, l])/(2*dx)
                        u[j, i, l] = np.maximum(np.minimum(U_eq(rho[j, i, l])-C_x, u_max), 0)
                        v[j, i, l] = np.maximum(np.minimum(V_eq(rho[j, i, l], y[l])/(1+rho[j, i, l]/(v_max*rho_max)), v_max), -v_max)
                        d1[j, i, l] = 1/(2*k**2)*C_x
                        d2[j, i, l] = 0
                    else:
                        C_x = (C[j, 0, l]-C[j, i-1, l])/(2*dx)
                        C_y = (C[j, i, l+1]-C[j, i, l-1])/(2*dy)
                        u[j, i, l] = np.maximum(np.minimum(U_eq(rho[j, i, l])-C_x, u_max), 0)
                        v[j, i, l] = np.maximum(np.minimum((V_eq(rho[j, i, l], y[l])-C_y)/(1+rho[j, i, l]/(v_max*rho_max)), v_max), -v_max)
                        d1[j, i, l] = 1/(2*k**2)*C_x
                        d2[j, i, l] = 1/(2*k**2)*C_y
                else:
                    if l == 0 or l == o-1:
                        C_x = (C[j, i+1, l]-C[j, i-1, l])/(2*dx)
                        u[j, i, l] = np.maximum(np.minimum(U_eq(rho[j, i, l])-C_x, u_max), 0)
                        v[j, i, l] = np.maximum(np.minimum(V_eq(rho[j, i, l], y[l])/(1+rho[j, i, l]/(v_max*rho_max)), v_max), -v_max)
                        d1[j, i, l] = 1/(2*k**2)*C_x
                        d2[j, i, l] = 0
                    else:
                        C_x = (C[j, i+1, l]-C[j, i-1, l])/(2*dx)
                        C_y = (C[j, i, l+1]-C[j, i, l-1])/(2*dy)
                        u[j, i, l] = np.maximum(np.minimum(U_eq(rho[j, i, l])-C_x, u_max), 0)
                        v[j, i, l] = np.maximum(np.minimum((V_eq(rho[j, i, l], y[l])-C_y)/(1+rho[j, i, l]/(v_max*rho_max)), v_max), -v_max)
                        d1[j, i, l] = 1/(2*k**2)*C_x
                        d2[j, i, l] = 1/(2*k**2)*C_y
    return C, u, v, d1, d2
                        
def fictitious_play(iter, Cold, Cnew, uold, unew, vold, vnew, d1old, d1new, d2old, d2new):
    Cnew = iter/(iter+1)*Cnew+1/(iter+1)*Cold
    unew = iter/(iter+1)*unew+1/(iter+1)*uold
    vnew = iter/(iter+1)*vnew+1/(iter+1)*vold
    d1new = iter/(iter+1)*d1new+1/(iter+1)*d1old
    d2new = iter/(iter+1)*d2new+1/(iter+1)*d2old
    return Cnew, unew, vnew, d1new, d2new

u_max = 1.02
rho_max = 1.13
v_max = 0.15
m = 240
n = 40
o = 40
b = 0.3
L = 1
T_max = 2
sigma = 0.001
x = np.linspace(0, L, n)
y = np.linspace(-b, b, o)
t = np.linspace(0, T_max, m)
dt = t[1]-t[0]
dy = y[1]-y[0]
dx = x[1]-x[0]
Y, X = np.meshgrid(y, x)
rhoold = np.zeros((m, n, o))
Cold = -y**2/2*np.ones((m, n, o))
uold = np.zeros_like(Cold)
vold = np.zeros_like(uold)
d1old = np.zeros_like(uold)
d2old = np.zeros_like(uold)
max_iter = 50
start = time.time()
print("Starting the Fixed Point Iteration")
solutions = []
Z = 0.0565+0.9*np.exp(-1/2*(X-0.5)**2/0.07**2-1/2*Y**2/0.09**2)
rhoold[0] = Z
Cold[-1] = -y/10
grids = [[40, 40, 240], [50, 50, 350], [60, 60, 420]]#, [70, 70, 210], [80, 80, 240]]
for j in range(len(grids)):
    error = 2
    o, n, m = grids[j]
    x = np.linspace(0, L, n)
    y = np.linspace(-b, b, o)
    t = np.linspace(0, T_max, m)
    dt = t[1]-t[0]
    dy = y[1]-y[0]
    dx = x[1]-x[0]
    dt = t[1]-t[0]
    dy = y[1]-y[0]
    dx = x[1]-x[0]
    Y, X = np.meshgrid(y, x)
    Z = 0.0565+0.9*np.exp(-1/2*(X-0.5)**2/0.07**2-1/2*Y**2/0.09**2)
    # Cold = -y/10*np.ones((m, n, o))#Define the Terminal Cost
    print('Current grid size:', m,'x',n,'x',o)
    for iter in range(max_iter):
        rhonew = FPK(m, n, o, dt, dx, dy, rhoold, uold, vold, d1old, d2old, sigma)
        Cnew, unew, vnew, d1new, d2new = HJBI(m, n, o, b, dt, dx, dy, rhoold, uold, vold, d1old, d2old, Cold, sigma)
        Cnew, unew, vnew, d1new, d2new = fictitious_play(iter, Cold, Cnew, uold, unew, vold, vnew, d1old, d1new, d2old, d2new)
        error = np.linalg.norm(rhonew-rhoold)+np.linalg.norm(unew-uold)+np.linalg.norm(vnew-vold)+np.linalg.norm(d1new-d1old)+np.linalg.norm(d2new-d2old)
        clear_output(wait=True)
        print('Current grid size:', m,'x',n,'x',o)
        print('Iteration:', iter+1, 'Total Error:', error)
        print(r'Error $\rho$:', np.linalg.norm(rhonew-rhoold))
        print(r'Error $u$:', np.linalg.norm(unew-uold))
        print(r'Error $v$:', np.linalg.norm(vnew-vold))
        print(r'Error $C$:', np.linalg.norm(Cnew-Cold))
        print(r'Error $d_1$:', np.linalg.norm(d1new-d1old))
        print(r'Error $d_2$:', np.linalg.norm(d2new-d2old))
        print(np.max(vold))
        print(np.min(vold))
        print(np.max(uold))
        print(np.min(uold))
        if iter%3 == 0:
            print('Processing ', np.round((iter+1)/max_iter*100, 2),'\b% complete(/)')
        if iter%3 == 1:
            print('Processing ', np.round((iter+1)/max_iter*100, 2),'\b% complete(–)')
        if iter%3 == 2:
            print('Processing ', np.round((iter+1)/max_iter*100, 2),'\b% complete(\\)')
        rhoold = rhonew
        uold = unew
        vold = vnew
        Cold = Cnew
        d1old = d1new
        d2old = d2new
        X, T = np.meshgrid(x, t)
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, uold[:, :, int(o/2)], cmap='seismic')
        ax.plot_surface(X, T, rhoold[:, :, int(o/2)], cmap='seismic')
        plt.show()
        if error>1:
            break
    solutions.append([rhoold, uold, vold, d1old, d2old, Cold])
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, uold[:, :, int(o/2)], cmap='seismic')
    ax.plot_surface(X, T, rhoold[:, :, int(o/2)], cmap='seismic')
    plt.show()
    print('Time elapsed:', time.time()-start, ' seconds')
    
    if j<len(grids)-1:
        o1, n1, m1 = grids[j+1]
        x1 = np.linspace(0, L, n1)
        y1 = np.linspace(-b, b, o1)
        t1 = np.linspace(0, T_max, m1)
        interp_rho = RegularGridInterpolator((t, x, y), rhoold, method='cubic')
        interp_u = RegularGridInterpolator((t, x, y), uold, method='cubic')
        interp_v = RegularlGridInterpolator((t, x, y), vold, method='cubic')
        interp_d1 = RegularGridInterpolator((t, x, y), d1old, method='cubic')
        interp_d2 = RegularGridInterpolator((t, x, y), d2old, method='cubic')
        interp_C = RegularGridInterpolator((t, x, y), Cold, method='cubic')
        T1, X1, Y1 = np.meshgrid(t1, x1, y1, indexing='ij')
        points_new = np.stack([T1.ravel(), X1.ravel(), Y1.ravel()], axis=-1)
        rhoold = interp_rho(points_new).reshape(m1, n1, o1)
        uold = interp_u(points_new).reshape(m1, n1, o1)
        vold = interp_v(points_new).reshape(m1, n1, o1)
        d1old = interp_d1(points_new).reshape(m1, n1, o1)
        d2old = interp_d2(points_new).reshape(m1, n1, o1)
        Cold = interp_C(points_new).reshape(m1, n1, o1)
        if error > 1:
            break
solutions.append([rhoold, uold, vold, d1old, d2old, Cold])
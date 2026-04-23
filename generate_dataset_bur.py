import numpy as np
import sys
import time as tm
import os

# ============================================================
# 3/2-rule for rfft
# ============================================================

def pad_rfft_32(yhat, N, M):
    Nh = N // 2 + 1
    Mh = M // 2 + 1
    yhat_pad = np.zeros(Mh, dtype=complex)
    yhat_pad[:Nh] = (M / N) * yhat
    return yhat_pad

def truncate_rfft_32(yhat_pad, N, M):
    Nh = N // 2 + 1
    return (N / M) * yhat_pad[:Nh].copy()

# ============================================================
# Non-linear term with 3/2-rule pseudospectral
# ============================================================

def NL(uhat, phat, N, M, kp):
    uhat_pad = pad_rfft_32(uhat, N, M)
    phat_pad = pad_rfft_32(phat, N, M)
    u = np.fft.irfft(uhat_pad, n=M)
    px = np.fft.irfft(1j * kp * phat_pad, n=M)
    prod_hat_pad = np.fft.rfft(u * px, n=M)
    return truncate_rfft_32(prod_hat_pad, N, M)

# ============================================================
# Spatial correlated noise
# ============================================================

def make_noise_sampler(N, L, Lc):
    dx = L / N
    x = np.arange(-N//2, N//2) * dx
    kernel = (1.0 - (x / Lc)**2) * np.exp(-0.5 * (x / Lc)**2)
    spec = np.fft.rfft(kernel)
    sqrt_spec = np.sqrt(spec)

    return {
        "N": N,
        "sqrt_spec": sqrt_spec,
    }

def sample_noisehat(noise_data):
    N = noise_data["N"]
    sqrt_spec = noise_data["sqrt_spec"]

    z = np.random.normal(0.0, 1.0, size=N)
    z -= np.mean(z)
    zhat = np.fft.rfft(z)

    return sqrt_spec * zhat

# ============================================================
# pre-compute simulation constants
# ============================================================

def make_step_data(N, L, nu, dt, Lc, g):
    if N % 2 != 0:
        raise ValueError("Need to be odd")

    M = 3 * N // 2
    dx = L / N
    x = np.arange(-N//2, N//2) * dx
    k = 2.0 * np.pi * np.fft.rfftfreq(N, d=dx)
    dxp = L / M
    kp = 2.0 * np.pi * np.fft.rfftfreq(M, d=dxp)
    E = np.exp(-nu * k**2 * dt)
    E2 = np.exp(-nu * k**2 * dt / 2.0)
    noise_data = make_noise_sampler(N, L, Lc)

    return {
        "N": N,
        "M": M,
        "L": L,
        "dt": dt,
        "nu": nu,
        "g": g,
        "x": x,
        "k": k,
        "kp": kp,
        "E": E,
        "E2": E2,
        "noise_data": noise_data,
    }


# ============================================================
# Single time-step
# ============================================================

def step(vhat, noih, data):
    """
    Single timestep t -> t + dt, updating vhat in-place.

    Equation
        u_t + u u_x = nu u_xx + g * eta

    Discretization:
        - Linear term: exact (integrating factor)
        - Non-linear : RK2
        - Noise      : Euler-Maruyama
    """
    N = data["N"]
    M = data["M"]
    dt = data["dt"]
    g = data["g"]
    kp = data["kp"]
    E = data["E"]
    E2 = data["E2"]
    
    dWhat = np.sqrt(dt) * g * sample_noisehat(data["noise_data"])
    Nv = -NL(vhat, vhat, N, M, kp)
    # intermediate RK2 
    a = E2 * (vhat + 0.5 * dt * Nv)
    Na = -NL(a, a, N, M, kp)
    # final update
    vhat[:] = E * vhat + dt * E2 * Na + E * dWhat
    vhat[0] = 0.0

    return vhat,dWhat/g/np.sqrt(dt)

def counter(t0,i,Nt):
    elapsed = tm.time() - t0
    avg_time = elapsed / i
    remaining = avg_time * (Nt - i)
    print(
        f"Step {i:05d}/{Nt:05d} - ETA: {remaining/60:.2f}min "
        f"(avg {avg_time:.2e}s/it)   ",
        end="\r",
        flush=True
    )

Nx = int(  sys.argv[1])
Ne = int(  sys.argv[2])
sp = int(  sys.argv[3])
dt = float(sys.argv[4])
Re = float(sys.argv[5])
eL = float(sys.argv[6])
Lc = float(sys.argv[7])
se = int(  sys.argv[8])

m = int(sp*Ne)
eL=2*np.pi*eL
dx=eL/Nx
x=dx*(np.arange(-Nx//2,Nx//2))

g  = Re**(3/2)
nu=2/(Nx/eL/(g*g*Lc)**(1/3))

dshock = nu*Nx/eL/(g*g*Lc)**(1/3)
np.random.seed(se)

data = make_step_data(Nx, eL, nu, dt, Lc, g)

x = data["x"]
vhat = np.fft.rfft(0*x)
noih = np.fft.rfft(0*x)

iout = sp

v = np.zeros((m//iout + 1, Nx))
noise = np.zeros((m//iout + 1, Nx))

t0 = tm.time()
for n in range(m):
    if n % iout == 0:
        temp=np.fft.irfft(vhat, n=Nx)
        check=np.mean(temp)
        if check == np.nan:
            print("nan")
            break
            
        j = n // iout
        v[j] = temp

        vhat, noih = step(vhat, noih, data)
        noise[j+1] = np.fft.irfft(noih, n=Nx)
        if n>0:
            counter(t0,n,m)

    else:
        vhat, noih = step(vhat, noih, data)

v[-1] = np.fft.irfft(vhat, n=Nx)

print("\nTotal time {:.2f} min".format((tm.time()-t0)/60))

corr_th=np.fft.fftshift(np.fft.irfft(np.abs(data["noise_data"]["sqrt_spec"])**2))
vhat = np.fft.rfft(v, axis=1)
k = eL*np.fft.rfftfreq(len(x), d=(x[1]-x[0]))
vx = np.fft.irfft(1j*k[None,:]*vhat, n=len(x), axis=1)
diss = nu*np.mean(vx[:-1]**2, axis=1)
Enerv=0.5*np.mean(v[:-1]**2,axis=1)
EnerI1=g*np.mean(v[:-1]*noise[1:]*np.sqrt(dt),axis=1)/dt
EnerI2=0.5*g*g*np.mean((noise[1:]*np.sqrt(dt))**2,axis=1)/dt

np.savez("dataset_bur_{:}".format(se),
         data=v[1:],
         noise=noise[1:],
         x=x,
         corr_th=corr_th,
         Nx=Nx,
         Ne=Ne,
         spacing=sp,
         dt=dt,
         Re=Re,
         eL=eL,
         Lc=Lc,
         seed=se,
         Enerv=Enerv,
         EnerI1=EnerI1,
         EnerI2=EnerI2,
         diss=diss,
        )


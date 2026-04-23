import os
import time as tm
from collections import defaultdict
import numpy as np
from moments_funcs import *

# ============================================================
# Profiler
# ============================================================

timings = defaultdict(float)
counts = defaultdict(int)

def timer_acc(func):
    def wrapper(*args, **kwargs):
        t0 = tm.perf_counter()
        out = func(*args, **kwargs)
        dt = tm.perf_counter() - t0

        timings[func.__name__] += dt
        counts[func.__name__] += 1
        return out
    return wrapper

def print_timings(timings,counts):
    items = sorted(timings.items(), key=lambda x: x[1], reverse=True)

    maxlen = max(len(k) for k, _ in items)
    
    print("\n--- Timing report ---")
    for k, total in items:
        avg = total / counts[k]
        print(
            f"{k:<{maxlen}} : "
            f"{avg*1e3:8.2f} ms (avg over {counts[k]:5d})  "
            f"total = {total:8.3f} s"
        )
        
# ============================================================
# Continue
# ============================================================

@timer_acc
def interp_moments(y0, y1, Nt, moments_func, datyp):

    Nv_loc = moments_func(y0).shape[1]
    S = np.zeros((Nt + 1, Nv_loc),y1.dtype)
    for i in range(Nt + 1):
        a = datyp(np.cos(np.pi * i / 2 / Nt))
        b = datyp(np.sin(np.pi * i / 2 / Nt))
        temp = a * y0 + b * y1
        S[i] = np.mean(moments_func(temp),axis=0)
    return S

@timer_acc
def interp_moment_at(i, y0, y1, Nt, mom_func, datyp):
    a = datyp(np.cos(np.pi * i / 2 / Nt))
    b = datyp(np.sin(np.pi * i / 2 / Nt))
    temp = a * y0 + b * y1
    return np.mean(mom_func(temp), axis=0)
    
@timer_acc
def dot1(eta, grad):
    return np.einsum('i,xei->ex', eta, grad, optimize=True)

@timer_acc
def build_G(gmom):
    Nx, Ne, Nv = gmom.shape
    A = gmom.reshape(Nx * Ne, Nv)
    return (A.T @ A) / Ne

@timer_acc
def EM_step(x, m0, m1, sigma, dt, sq2dt, moments_func, grad_func, ridge=0.0):
    dpdx = grad_func(x)
    G0 = build_G(dpdx)
    dmdt = (m1 - m0) / dt
    eta = np.linalg.solve(G0, dmdt)

    noise = (np.random.normal(0.0, 1.0, size=x.shape)).astype(x.dtype)
    y = x + dt * dot1(eta, dpdx) + sq2dt * sigma * noise

    dpdx = grad_func(y)
    G1 = build_G(dpdx)
    p = np.mean(moments_func(y), axis=0)
    tet = np.linalg.solve(G1, (p - m1)) / (dt * sigma * sigma )

    x = y - dt * sigma * sigma * dot1(tet, dpdx)

    return x, eta, tet, G0, G1, np.mean(moments_func(x), axis=0)

# ============================================================
# Selection / naming helpers
# ============================================================

def string_to_mom(moment_name):
    A=np.zeros(4,bool)
    B=np.zeros(4,bool)
    for i in range(4):
        if moment_name[2*i]=='0':
            A[i]=False
        else:
            A[i]=True
        if moment_name[2*i+1]=='0':
            B[i]=False
        else:
            B[i]=True

    # Choose the en up to the fixed moment
    Ab=np.where(A==True)
    Bb=np.where(B==True)
    if len(Ab[0])==0:
        tempA=0
    else:
        tempA=np.max(Ab)
    if len(Bb[0])==0:
        tempB=0
    else:
        tempB=np.max(Bb)
    
    en = np.max([tempA,tempB])+1
    return A,B,en

# ============================================================
# Saving helpers
# ============================================================
@timer_acc
def save_check(name,y,tet,eta,G_0,G_1,mt,moms,Nt,dt,seed,sigma,nwin,temporary=False):
    state = np.array(np.random.get_state(), dtype=object)
    if temporary==True:
        if os.path.exists("./data/temp/")==False:
            os.mkdir("./data/temp/")
        complete_name="./data/temp/"+name
    else:
        complete_name="./data/"+name
        
    np.savez(
        complete_name,
        data=y,
        theta=tet,
        eta=eta,
        G_prenoise=G_0,
        G_postnoise=G_1,
        moments_path=mt,
        moments_avrg=moms,
        Nt=Nt,
        dt=dt,
        seed=seed,
        sigma=sigma,
        nwin=nwin,
        state=state,
        )
    
def check_existing(name,y,tet,eta,G_0,G_1,moms,mt,i0):
    if os.path.exists("./data/temp/"+name)==False:
        pass
    else:
        temp  = np.load("./data/temp/"+name,"r",allow_pickle=True)
        tet   = temp['theta']
        i0    = np.where(np.std(tet,axis=1)==0)[0][0]
        print("Starting from stopped simulation: i ==",i0)
        eta   = temp['eta']
        G_0   = temp['G_prenoise']
        G_1   = temp['G_postnoise']
        y     = temp['data']
        moms  = temp["moments_avrg"]
        mt  = temp["moments_path"]
        state = tuple(temp['state'])
        np.random.set_state(state)        
    return y,tet,eta,G_0,G_1,moms,mt,i0

@timer_acc    
def elap_time(t0,i,Nt):
    elapsed = tm.time() - t0
    avg_time = elapsed / i
    remaining = avg_time * (Nt - i)
    print(
        f"Step {i:05d}/{Nt:05d} - ETA: {remaining/60:.2f}min "
        f"(avg {avg_time:.3f}s/it)   ",
        end="\r",
        flush=True
    )

def noisegen(Ne,Nx,datyp,Lc=16):
    
    s=np.zeros((Ne,Nx),datyp)
    dx=2*np.pi/Nx
    x=np.arange(Nx)*dx
    Lc=Lc*dx
    # kern=np.fft.rfft(np.e**(-0.5*((x-np.pi)/Lc)**2))
    kern=np.fft.rfft((1-((x-np.pi)/Lc)**2)*np.e**(-0.5*((x-np.pi)/Lc)**2))
    
    temp=np.fft.rfft(np.random.normal(0, 1, size=s.shape),axis=1)
    s=(np.fft.irfft(np.sqrt(kern)*temp,axis=1)).astype(datyp)
        
    return s

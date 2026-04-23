import os
import re
import numpy as np
from functools import partial
from collections import defaultdict
import time
from moments_funcs import *

# ============================================================
# Profiler
# ============================================================

timings = defaultdict(float)
counts = defaultdict(int)

def timer_acc(func):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = func(*args, **kwargs)
        dt = time.perf_counter() - t0
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
# Post-processing functions
# ============================================================

def list_moment_codes(data_dir):
    pattern = re.compile(r"run_([01]{8})_Nt-")
    moments = set()
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            moments.add(match.group(1))

    return sorted(moments)

def find_runs_by_moment(data_dir, moment_code):
    pattern = re.compile(
        rf"run_{moment_code}_Nt-(\d+)_nwin-(\d+)_sigma-([0-9,\.]+)_seed-(\d+)_.*\.npz"
    )
    results = []
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            Nt = int(match.group(1))
            nwin = int(match.group(2))
            sigma = float(match.group(3).replace(',', '.'))
            seed = int(match.group(4))
            results.append({
                "Nt": Nt,
                "nwin": nwin,
                "sigma": sigma,
                "seed": seed,
                "file": os.path.join(data_dir, fname),
            })
    results.sort(key=lambda x: (x["nwin"], x["sigma"], x["Nt"], x["seed"]))
    return results

def deltau(v,r):
    if r==None:
        
        return v    
    s=np.concatenate((v,v[:r]))
    
    return (np.roll(s,-r)-s)[:len(v)]

def lower_bound_S(B_name):
    B    = np.load(B_name, mmap_mode='r')
    tet  = B["theta"]
    mom  = B["moments_path"]
    dt   = 1/B["Nt"]
    t    = np.arange(1,B["Nt"]+1)*dt
    dmdt = np.gradient(mom,axis=1)/dt
    temp = np.sum(tet*dmdt,axis=1)
    H_0  = (1+np.log(2*np.pi))/2
    Ht   = np.cumsum(temp)*dt+H_0
    
    return Ht[-1]

def corr(x):
    return np.fft.fftshift(np.fft.irfft(np.abs(np.fft.rfft(x)) ** 2, axis=-1) / x.shape[-1], axes=-1)

def moving_average(a, n=100): 
    a = np.asarray(a)
    ret = np.cumsum(a, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    out = np.zeros_like(a)
    out[:1 - n] = ret[n - 1:] / n
    out[1 - n:] = ret[-1] / n
    
    return out


def e_avg(x):
    return np.mean(x, axis=0)

def SF(x, ells_idx, n, absol=False):
    Ne_loc, Nx_loc = x.shape
    nl_loc = len(ells_idx)
    out = np.empty((Ne_loc, nl_loc), dtype=x.dtype)
    for i, ii in enumerate(ells_idx):
        d = x[:, ii:] - x[:, :-ii]
        if absol==True:
            d = np.abs(d)
        out[:, i] = np.mean(d**n, axis=1)

    return out

def dot0(a, b):
    return np.sum(a * b, axis=0)

def dot1(eta, grad):
    return np.einsum('i,xei->ex', eta, grad, optimize=True)

def pack_nondiag_symmetric(S):
    S = np.asarray(S)
    iu = np.triu_indices(S.shape[1])
    
    return S[:, iu[0], iu[1]]

def unpack_nondiag_symmetric(v):
    v = np.asarray(v,dtype=v.dtype)
    Ne_loc, Nv_sym = v.shape
    nl_float = (np.sqrt(1 + 8 * Nv_sym) - 1) / 2
    nl_loc = int(nl_float)
    if nl_loc * (nl_loc + 1) // 2 != Nv_sym:
        raise ValueError("Nv_sym não corresponde a nenhum nl válido.")
    S = np.zeros((Ne_loc, nl_loc, nl_loc), dtype=v.dtype)
    iu = np.triu_indices(nl_loc)
    S[:, iu[0], iu[1]] = v
    S[:, iu[1], iu[0]] = v
    
    return S

# ============================================================
# Selection / naming helpers
# ============================================================

def make_ells(dx_in: float, Nx_in: int):
    return 2.0 ** np.arange(int(np.log(Nx_in) / np.log(2))) * dx_in

def run_filename(dataset_name: str, moment_file: str, Nt: int, sigma: float, seed: int) -> str:
    dataset_base = os.path.splitext(os.path.basename(dataset_name))[0]
    moment_base = os.path.splitext(os.path.basename(moment_file))[0]
    sigma_txt = f"{sigma:.6f}".replace('.', 'p')
    
    return (
        f"run_{moment_base}_Nt-{int(Nt):04d}_sigma-{sigma_txt}"
        f"_seed-{int(seed):d}_{dataset_base}.npz")

def mom_to_string(Ond,Odi):
    moment_name=np.str()
    for i in range(len(Odi)):
        if Odi[i]==False:
            scra="0"
        else:
            scra="1"
        moment_name=moment_name+scra
        
        if Ond[i]==False:
            scra="0"
        else:
            scra="1"
        moment_name=moment_name+scra

    return moment_name


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

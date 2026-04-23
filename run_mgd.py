import numpy as np
import time as tm
import os
import sys
from moments_funcs import *
from mgd_lib import *
from functools import partial

sigma = float(sys.argv[1])
Nt    = int(  sys.argv[2])
seed  = int(  sys.argv[3])
data_path = str(sys.argv[4])
moms_path = str(sys.argv[5])
nwin  = int(sys.argv[6])
# solve_bool = sys.argv[6].lower() in ("true", "1", "yes", "y")

datyp=np.float32
temp = np.load(data_path, mmap_mode='r')
y1 = temp['data'].astype(datyp)
Ne, Nx = y1.shape

moment_name=str(np.loadtxt(moms_path,dtype=str))
Odi,Ond,en=string_to_mom(moment_name)

ells_int=(2.0**np.arange(int(np.log(Nx)/np.log(2)))).astype(int)

nl=len(ells_int)
dt=datyp(1/Nt)
sigma=datyp(sigma)
sq2dt=datyp(np.sqrt(2*dt))

print("MGD vitu!")
print("en, seed   = {:02d}, {:}".format(en,seed))
print("sigma      = {:.3e}".format(sigma))
print("Ne, Nx, Nt = {:04d}, {:04d}, {:04d}".format(Ne,Nx,Nt))
print("Nwin       = {:04d}".format(nwin))
print("Nl         = {:04d}".format(nl),flush=True)

np.random.seed(seed)
# y0 = (np.random.normal(0,1,size=y1.shape)).astype(datyp)
y0=noisegen(Ne, Nx, datyp)
y = np.copy(y0)

mom_func = partial(moments_func     , ell_int=ells_int, en=en, Odi=Odi, Ond=Ond, nwin=nwin)
grd_func = partial(grad_moments_func, ell_int=ells_int, en=en, Odi=Odi, Ond=Ond, nwin=nwin) 

# mt=interp_moments(y,y1,Nt,mom_func,datyp)
m_init = interp_moment_at(0,   y0,  y1, Nt, mom_func, datyp)
Nv=len(m_init)

print("Nv         = {:04d}".format(Nv),flush=True)

if Nt<=10000:
    Nt_save=Nt
else:
    Nt_save=10000

iout = int(Nt//Nt_save)

eta  = np.zeros((Nt_save,Nv),dtype=datyp)
tet  = np.zeros((Nt_save,Nv),dtype=datyp)
G_0  = np.zeros((Nt_save,Nv,Nv),dtype=datyp)
G_1  = np.zeros((Nt_save,Nv,Nv),dtype=datyp)
moms = np.zeros((Nt_save,Nv),dtype=datyp)
mt   = np.zeros((Nt_save,Nv),dtype=datyp)

eta_acc  = np.zeros(Nv, dtype=datyp)
tet_acc  = np.zeros(Nv, dtype=datyp)
G0_acc   = np.zeros((Nv, Nv), dtype=datyp)
G1_acc   = np.zeros((Nv, Nv), dtype=datyp)
moms_acc = np.zeros(Nv, dtype=datyp)

# Savings
sigma_txt = f"{sigma:3.2f}".replace('.', ',')
name=(f"run_{moment_name}_Nt-{int(Nt):05d}_nwin-{int(nwin):05d}_sigma-{sigma_txt}"+
      f"_seed-{int(seed):d}_{data_path}")

if os.path.exists("./data")==False:
    os.mkdir("./data")
#
i0=0
y,tet,eta,G_0,G_1,moms,mt,i0=check_existing(name,y,tet,eta,G_0,G_1,moms,mt,i0)

i0=int(i0*iout)
iblock = i0 // iout
count_acc = 0

t0 = tm.time()

save_stride = max(1, Nt // 100)

for i in range(i0,Nt,1):
    # m0 = mt[i]
    # m1 = mt[i+1]

    m0 = interp_moment_at(i,   y0,  y1, Nt, mom_func, datyp)
    m1 = interp_moment_at(i+1, y0,  y1, Nt, mom_func, datyp)

    y, eta_i, tet_i, G0_i, G1_i, moms_i = EM_step(
        y, m0, m1, sigma, dt, sq2dt, mom_func, grd_func)

    eta_acc  += eta_i
    tet_acc  += tet_i
    G0_acc   += G0_i
    G1_acc   += G1_i
    moms_acc += moms_i
    count_acc += 1
    
    if count_acc == iout or i == Nt - 1:
        eta[iblock]  = eta_acc  / count_acc
        tet[iblock]  = tet_acc  / count_acc
        G_0[iblock]  = G0_acc   / count_acc
        G_1[iblock]  = G1_acc   / count_acc
        moms[iblock] = moms_acc / count_acc
        mt[iblock]   = m1
    
        eta_acc.fill(0)
        tet_acc.fill(0)
        G0_acc.fill(0)
        G1_acc.fill(0)
        moms_acc.fill(0)
        count_acc = 0
        iblock += 1
        if i > 0:
            elap_time(t0,i,Nt)        
        
    if i % save_stride == 0 and i > 0:
        save_check(name,y,tet,eta,G_0,G_1,mt,moms,Nt,dt,seed,sigma,nwin,True)
    
print("\nTotal time {:.2f} min".format((tm.time()-t0)/60))
print_timings(timings,counts)

save_check(name,y,tet,eta,G_0,G_1,mt,moms,Nt,dt,seed,sigma,nwin,False)

temp_file = "./data/temp/" + name
if os.path.exists(temp_file):
    os.remove(temp_file)


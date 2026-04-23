import numpy as np
import sys
import os

def noisegen(x, Lc, H, m=1):
    """
    Gera m+1 realizacoes de ruido gaussiano com
    S2(r) ~ r^(2H) para r << Lc e saturando para r >> Lc
    """
    n = len(x)
    dx = x[1] - x[0]
    
    # Frequencias discretas
    f = np.fft.rfftfreq(n, d=dx)
    f[0] = 0  # evita divisão por zero
    
    # Kernel espectral regularizado
    # Saturação em altas frequências
    kern = (1 + (f*Lc)**2)**(-(H+0.5))
    
    s = np.zeros((m+1, n))
    for i in range(m+1):
        temp = np.fft.rfft(np.random.normal(0,1,n))
        s[i] = np.fft.irfft(np.sqrt(kern) * temp)
    
    # Kernel no domínio real (função de correlação)
    corr_th = np.fft.irfft(kern)
    
    return s, corr_th

def corr(x):
    return (np.fft.irfft(np.abs(np.fft.rfft(x))**2)/len(x))

Nx = int(  sys.argv[1])
Ne = int(  sys.argv[2])
H  = float(sys.argv[3])
eL = float(sys.argv[4])
Lc = float(sys.argv[5])
se = int(  sys.argv[6])


eL=2*np.pi/eL
dx=eL/Nx
x=np.arange(Nx)*dx
np.random.seed(se)

y,corr_th=noisegen(x,Lc,H,Ne-1)
corr_th=np.fft.fftshift(corr_th)

np.savez("dataset_fbm_{:}".format(se),
         data=y,
         x=x,
         corr_th=corr_th,
         Nx=Nx,
         Ne=Ne,
         H=H,
         eL=eL,
         Lc=Lc,
         seed=se
        )

# zx=np.zeros((Ne,Nx))
# ze=np.zeros((Nx,Ne))

# for i in range(Ne):
#     zx[i]=corr(y[i])

# for i in range(Nx):
#     ze[i]=corr(y[:,i])

# zx=np.mean(zx,axis=0)
# ze=np.mean(ze,axis=0)


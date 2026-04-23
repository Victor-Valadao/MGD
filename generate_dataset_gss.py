import numpy as np
import sys

def noisegen(x,Lc,eL,m):
    
    s=np.zeros((m+1,len(x)),float)
    dx=eL/len(x)
    # kern=np.fft.rfft(np.e**(-0.5*((x-np.pi)/Lc)**2))
    kern=np.fft.rfft((1-((x-np.pi)/Lc)**2)*np.e**(-0.5*((x-np.pi)/Lc)**2))
    
    temp=np.fft.rfft(np.random.normal(0, 1, size=s.shape),axis=1)
    s=np.fft.irfft(np.sqrt(kern)*temp,axis=1)
        
    return s,np.fft.irfft(kern)

def corr(x):
    return (np.fft.irfft(np.abs(np.fft.rfft(x))**2)/len(x))

Nx = int(  sys.argv[1])
Ne = int(  sys.argv[2])
eL = float(sys.argv[3])
Lc = float(sys.argv[4])
se = int(  sys.argv[5])

eL=2*np.pi*eL
dx=eL/Nx
x=np.arange(Nx)*dx
np.random.seed(se)

y,corr_th=noisegen(x,Lc,eL,Ne-1)
corr_th=np.fft.fftshift(corr_th)

np.savez("dataset_gss_{:}".format(se),
         data=y,
         x=x,
         corr_th=corr_th,
         Nx=Nx,
         Ne=Ne,
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

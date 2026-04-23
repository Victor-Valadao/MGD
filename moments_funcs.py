import numpy as np
from mgd_lib import *

# ============================================================
# Helpers for windowed moments
# ============================================================
@timer_acc
def make_spatial_windows(Nx, nwin=1):
    if nwin is None:
        nwin = 1
    nwin = int(nwin)
    if nwin < 1:
        raise ValueError("nwin must be >= 1")
    bounds = np.linspace(0, Nx, nwin + 1, dtype=int)
    windows = [np.arange(bounds[i], bounds[i + 1], dtype=int) for i in range(nwin)]
    if any(len(w) == 0 for w in windows):
        raise ValueError("nwin is too large for the given Nx; some windows are empty")
    return windows

@timer_acc
def windowed_power_means(x, windows, power):
    Ne = x.shape[0]
    nwin = len(windows)
    out = np.empty((Ne, nwin), dtype=x.dtype)
    xp = x ** power
    for k, w in enumerate(windows):
        out[:, k] = np.mean(xp[:, w], axis=1)
    return out

@timer_acc
def grad_windowed_power_means(x, windows, power):
    Ne, Nx = x.shape
    nwin = len(windows)
    grad = np.zeros((Nx, Ne, nwin), dtype=x.dtype)

    if power == 1:
        for k, w in enumerate(windows):
            grad[w, :, k] = 1.0 / len(w)
    else:
        xp = x ** (power - 1)
        for k, w in enumerate(windows):
            grad[w, :, k] = (power * xp[:, w] / len(w)).T

    return grad


# ============================================================
# Call moments and grad moments
# ============================================================

@timer_acc
def moments_func(x, ell_int, en, Odi, Ond, nwin=1):
    # MOMENTS return shape (Ne, Nv)
    Ne_loc, Nx_loc = x.shape
    nl_loc = len(ell_int)
    windows = make_spatial_windows(Nx_loc, nwin=nwin)
    nwin_loc = len(windows)

    # field moments contribute en * nwin_loc columns
    N_max = en * nwin_loc + 2 * (nl_loc**2 + nl_loc + 4)  # safe upper bound up to 4th order
    i_tot = 0
    out = np.empty((Ne_loc, N_max), dtype=x.dtype)

    for p in range(1, en + 1):
        temp = windowed_power_means(x, windows, p)
        out[:, i_tot:i_tot + nwin_loc] = temp
        i_tot += nwin_loc

    incs = increments_list(x, ell_int)

    for i in range(4):
        if Odi[i] and Ond[i]:
            print(f"Warning -- Diag and non-Diag requested for order S{i+1}. Using diagonal.")

        if Odi[i]:
            nl2 = nl_loc
            out[:, i_tot:i_tot + nl2] = inc_avg(incs, i + 1)
            i_tot += nl2

        elif Ond[i]:
            if i == 0:
                # 2-scale 1-field == impossible
                pass
            elif i == 1:
                nl2 = nl_loc * (nl_loc + 1) // 2
                out[:, i_tot:i_tot + nl2] = moments_S2_nondiag(incs)
                i_tot += nl2
            elif i == 2:
                nl2 = nl_loc * nl_loc
                out[:, i_tot:i_tot + nl2] = moments_S3_nondiag(incs)
                i_tot += nl2
            elif i == 3:
                nl2 = nl_loc * (nl_loc + 1) // 2
                out[:, i_tot:i_tot + nl2] = moments_S4_nondiag(incs)
                i_tot += nl2

    return out[:, :i_tot]

@timer_acc
def grad_moments_func(x, ell_int, en, Odi, Ond, nwin=1):
    Ne_loc, Nx_loc = x.shape
    nl_loc = len(ell_int)
    windows = make_spatial_windows(Nx_loc, nwin=nwin)
    nwin_loc = len(windows)

    # field moments contribute en * nwin_loc columns
    N_max = en * nwin_loc + 2 * (nl_loc**2 + nl_loc + 4)   # upper bound
    i_tot = 0
    out = np.empty((Nx_loc, Ne_loc, N_max), dtype=x.dtype)

    for p in range(1, en + 1):
        temp = grad_windowed_power_means(x, windows, p)
        out[:, :, i_tot:i_tot + nwin_loc] = temp
        i_tot += nwin_loc

    incs = increments_list(x, ell_int)

    for i in range(4):
        if Odi[i] and Ond[i]:
            print(f"Warning -- Diag and non-Diag requested for order S{i+1}. Using diagonal.")

        if Odi[i]:
            temp = grad_inc_avg(incs, ell_int, i + 1, Nx_loc, Ne_loc)
            nl2 = temp.shape[2]
            out[:, :, i_tot:i_tot + nl2] = temp
            i_tot += nl2

        elif Ond[i]:
            if i == 0:
                # 2-scale 1-field == impossible
                pass
            elif i == 1:
                nl2 = nl_loc * (nl_loc + 1) // 2
                out[:, :, i_tot:i_tot + nl2] = grad_moments_S2_nondiag(incs, ell_int, Nx_loc, Ne_loc)
                i_tot += nl2
            elif i == 2:
                nl2 = nl_loc * nl_loc
                out[:, :, i_tot:i_tot + nl2] = grad_moments_S3_nondiag(incs, ell_int, Nx_loc, Ne_loc)
                i_tot += nl2
            elif i == 3:
                nl2 = nl_loc * (nl_loc + 1) // 2
                out[:, :, i_tot:i_tot + nl2] = grad_moments_S4_nondiag(incs, ell_int, Nx_loc, Ne_loc)
                i_tot += nl2

    return out[:, :, :i_tot]


# ============================================================
# Helpers
# ============================================================

@timer_acc
def increments_list(x, ells_int):
    return [x[:, int(sh):] - x[:, :-int(sh)] for sh in ells_int]


@timer_acc
def inc_avg(incs, order):
    nl_loc = len(incs)
    Ne_loc = incs[0].shape[0]
    out = np.zeros((Ne_loc, nl_loc), dtype=incs[0].dtype)

    for i in range(nl_loc):
        out[:, i] = np.mean(incs[i]**order, axis=1)

    return out


@timer_acc
def grad_inc_avg(incs, ell_int, order, Nx, Ne):
    # Gradients of < (delta_ell x)^order >, return shape (Nx, Ne, nl)
    nl_loc = len(ell_int)
    grad = np.zeros((Nx, Ne, nl_loc), dtype=incs[0].dtype)
    for i, (ii, d) in enumerate(zip(ell_int, incs)):
        Nl = d.shape[1]
        c = (order / Nl) * (d ** (order - 1))   # shape (Ne, Nl)

        grad[ii:ii + Nl, :, i] += c.T
        grad[:Nl, :, i] -= c.T

    return grad


# ============================================================
# 2-scale moments
# ============================================================

@timer_acc
def moments_S2_nondiag(incs):
    nl = len(incs)
    cols = []
    for i in range(nl):
        di = incs[i]
        for j in range(i, nl):
            dj = incs[j]
            Nl = min(di.shape[1], dj.shape[1])
            cols.append(np.mean(di[:, :Nl] * dj[:, :Nl], axis=1))

    return np.column_stack(cols)


@timer_acc
def moments_S3_nondiag(incs):
    nl = len(incs)
    cols = []
    for i in range(nl):
        di = incs[i]
        for j in range(nl):
            dj = incs[j]
            Nl = min(di.shape[1], dj.shape[1])
            cols.append(np.mean(di[:, :Nl] * (dj[:, :Nl] ** 2), axis=1))

    return np.column_stack(cols)


@timer_acc
def moments_S4_nondiag(incs):
    nl = len(incs)
    cols = []
    for i in range(nl):
        di = incs[i]
        for j in range(i, nl):
            dj = incs[j]
            Nl = min(di.shape[1], dj.shape[1])
            cols.append(np.mean((di[:, :Nl] ** 2) * (dj[:, :Nl] ** 2), axis=1))

    return np.column_stack(cols)


# ============================================================
# Symmetric non-diagonal S2 and S4
# Pairs (i,j) with j >= i
# ============================================================

@timer_acc
def grad_moments_S2_nondiag(incs, ells_idx, Nx, Ne):
    nl = len(ells_idx)
    Nv_sym = nl * (nl + 1) // 2

    grad = np.zeros((Nx, Ne, Nv_sym), dtype=incs[0].dtype)

    col = 0
    for i, ii in enumerate(ells_idx):
        di = incs[i]
        for j, jj in enumerate(ells_idx[i:], start=i):
            dj = incs[j]
            Nl = min(di.shape[1], dj.shape[1])
            grad[ii:ii + Nl, :, col] += (dj[:, :Nl]).T / Nl
            grad[jj:jj + Nl, :, col] += (di[:, :Nl]).T / Nl
            grad[:Nl, :, col] -= ((di[:, :Nl] + dj[:, :Nl])).T / Nl
            col += 1

    return grad


@timer_acc
def grad_moments_S3_nondiag(incs, ells_idx, Nx, Ne):
    nl = len(ells_idx)
    Nv = nl * nl

    grad = np.zeros((Nx, Ne, Nv), dtype=incs[0].dtype)

    col = 0
    for i, ii in enumerate(ells_idx):
        di = incs[i]
        for j, jj in enumerate(ells_idx):
            dj = incs[j]
            Nl = min(di.shape[1], dj.shape[1])

            grad[ii:ii + Nl, :, col] += (dj[:, :Nl] ** 2).T / Nl
            grad[jj:jj + Nl, :, col] += (di[:, :Nl] * dj[:, :Nl]).T * 2.0 / Nl
            grad[:Nl, :, col] -= (dj[:, :Nl] ** 2 + 2.0 * di[:, :Nl] * dj[:, :Nl]).T / Nl
            col += 1

    return grad


@timer_acc
def grad_moments_S4_nondiag(incs, ells_idx, Nx, Ne):
    nl = len(ells_idx)
    Nv_sym = nl * (nl + 1) // 2

    grad = np.zeros((Nx, Ne, Nv_sym), dtype=incs[0].dtype)

    col = 0
    for i, ii in enumerate(ells_idx):
        di = incs[i]
        for j, jj in enumerate(ells_idx[i:], start=i):
            dj = incs[j]
            Nl = min(di.shape[1], dj.shape[1])

            grad[ii:ii + Nl, :, col] += (di[:, :Nl] * (dj[:, :Nl] ** 2)).T * 2.0 / Nl
            grad[jj:jj + Nl, :, col] += (dj[:, :Nl] * (di[:, :Nl] ** 2)).T * 2.0 / Nl
            grad[:Nl, :, col] -= (di[:, :Nl] * (dj[:, :Nl] ** 2) + dj[:, :Nl] * (di[:, :Nl] ** 2)).T * 2.0 / Nl
            col += 1

    return grad

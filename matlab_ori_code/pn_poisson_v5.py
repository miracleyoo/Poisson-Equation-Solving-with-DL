# Generated with SMOP  0.41-beta
from const_val import *


# pn_poisson_v5.m


@function
def pn_poisson_v5(dx=None, dy=None, nx1=None, ny1=None, eps_stat=None, q=None, p_charge=None, n_charge=None,
                  bg_charge=None, phi=None, Vp=None, Vn=None, *args, **kwargs):
    varargin = pn_poisson_v5.varargin
    nargin = pn_poisson_v5.nargin

    k = 0
    delta_phi = 0.01
    delta_phi_max = 3e-06
    net_charge = bg_charge - n_charge + p_charge
    phi_temp = zeros(*phi.shape)
    # for k=0:POISSON_ITER_MAX
    delta_phi_temp = zeros(*phi.shape)
    while abs(delta_phi) > 3e-06:

        for j in arange(1, ny1).reshape(-1):
            phi[j, 1] = Vp
            phi[j, nx1] = Vn
        # for top and bottom
        for i in arange(2, nx1 - 1).reshape(-1):
            phi[1, i] = phi[2, i]
            phi[ny1, i] = phi[ny1 - 1, i]
        for j in arange(1, ny1).reshape(-1):
            for i in arange(1, nx1).reshape(-1):
                phi_temp[j, i] = phi[j, i]
        for j in arange(2, ny1 - 1).reshape(-1):
            for i in arange(2, nx1 - 1).reshape(-1):
                phi[j, i] = 0.25*(phi_temp[j, i + 1] + phi[j, i - 1] + phi[j - 1, i] + phi_temp[j + 1, i] +
                    dx*dx*net_charge[j, i]*q/eps_stat)
                delta_phi_temp = phi[j, i] - phi_temp[j, i]
                if abs(delta_phi_temp) > abs(delta_phi_max):
                    delta_phi_max = delta_phi_temp
        delta_phi = delta_phi_max
        delta_phi_max = 3e-06
        k = k + 1


        # after getting the final phi, check the boundary again
    # for left and right
    for j in arange(1, ny1).reshape(-1):
        phi[j, 1] = Vp
        phi[j, nx1] = Vn

    # for top and bottom
    for i in arange(2, nx1 - 1).reshape(-1):
        phi[1, i] = phi[2, i]
        phi[ny1, i] = phi[ny1 - 1, i]

    fx = matlabarray(np.zeros((ny1, nx1)))
    for i in arange(1, ny1).reshape(-1):
        for j in arange(2, (nx1 - 1)).reshape(-1):
            fx[i, j] = ((phi[i, j - 1] - phi[i, j + 1]) / dx) / 2
            # because E=-d(phi)/dx!!

    # for left and right boundary
    fx[:, 1] = fx[:, 2]
    fx[:, -1] = fx[:, -2]
    fy = matlabarray(np.zeros((ny1, nx1)))
    for j in arange(1, nx1).reshape(-1):
        for i in arange(2, (ny1 - 1)).reshape(-1):
            fy[i, j] = (phi[i - 1, j] - phi[i + 1, j]) / (dot(2, dy))

    # for top and bottom boundary
    fy[1, :] = fy[2, :]
    fy[-1, :] = fy[-2, :]
    return fx, fy, phi, k

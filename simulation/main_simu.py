# coding: utf-8
# Author: Zhongyang Zhang

import pickle
import time

import matlab.engine

from global_val import *
from matlab import double as dbl


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('==> [%s]:\t' % self.name, end='')
        print('Elapsed Time: %s (s)' % (time.time() - self.tstart))


eng = matlab.engine.start_matlab()
Vp_all = [0]
Vn_all = [0]

# with Timer('make_GaAs_scatTable'):
#     scatGaAs, GmG, GmL = eng.make_GaAs_scatTable(T, 0.0, de, 2.0, float(cdop), nargout=3)
# with Timer('make_GaAs_hole_scatTable_v2'):
#     scatGaAs_hole, Gmh, Gml = eng.make_GaAs_hole_scatTable_v2(T, de, 2.0, float(cdop), nargout=3)
#
# scatGaAs = np.array(scatGaAs)
# scatGaAs_hole = np.array(scatGaAs_hole)
#
# Gm = [GmG, GmL, Gmh, Gml]
# Gm_max = max(Gmh, Gml)
#
# # ------------related to configuration--------------------------------------
# nx1 = int(round(Ltot / dx) + 1)
# nx = nx1 - 1
# ny1 = int(round(Ttot / dy) + 1)
# ny = ny1 - 1
# bottom_pts = np.arange(2, nx + 1)
# left_pts = np.arange(1, nx1 * ny1 + 1, nx1)
# right_pts = np.arange(nx1, nx1 * ny1 + 1, nx1)
# top_pts = np.arange(nx1 * ny1 - nx1 + 2, nx1 * ny1 - 1)
# p_icpg = [0] * len(left_pts)
# n_icpg = [0] * len(right_pts)
#
# # for a in range(len(Vp_all)):
# #     for b in range(len(Vn_all)):
# Vp = Vp_all[0] - contact_potential / 2
# Vn = Vn_all[0] + contact_potential / 2
# with Timer('pn_init_v2'):
#     particles, valley, bg_charge, cpsp, N, p_icpg, n_icpg, xmax, ymax = \
#         eng.pn_init_v2(float(max_particles), dope_type, dx, dy, Ltot, nx1, ny1, cdop, ppc, bk, T,
#                    q, h, dbl(alpha), dbl(eM), dbl(Gm), Lp, A, B, C, emR, hd, nargout=9)
#
# valley = np.array(valley)
# particles = np.array(particles)
# bg_charge = np.array(bg_charge)
# n_icpg = np.array(n_icpg)
# p_icpg = np.array(p_icpg)
#
# # --------Initial Charge/Field Computations---------------------------------
# with Timer('pn_charge_v2'):
#     charge_p, charge_n = eng.pn_charge_v2(dbl(particles.tolist()), dbl(valley.tolist()), nx1, ny1,
#                                           dx, dy, max_particles, cpsp, nargout=2)
# charge_n = np.array(charge_n)
# charge_p = np.array(charge_p)
#
# phi = np.zeros((ny1, nx1))

with Timer('init_core'):
    nx1, ny1, charge_p, charge_n, bg_charge, phi, Vp, Vn, xmax, ymax, qD, cpsp, scatGaAs,\
        scatGaAs_hole, Gm, p_icpg, n_icpg, left_pts, right_pts, valley, particles = eng.init_core(nargout=21)

with Timer('pn_poisson_v5'):
    fx, fy, phi = eng.pn_poisson_v5(dx, dy, nx1, ny1, eps_stat, q, charge_p,
        charge_n, bg_charge, phi, Vp, Vn, nargout=3)

# fx = np.array(fx)
# fy = np.array(fy)
# phi = np.array(phi)

number = dbl(np.zeros((tsteps, 4)).tolist())
current_n = dbl(np.zeros((tsteps, 1)).tolist())
current_p = dbl(np.zeros((tsteps, 1)).tolist())
particle_num = dbl(np.zeros((tsteps + 1, 4)).tolist())

# p_enter = np.zeros((tsteps, 1))
# p_exit = np.zeros((tsteps, 1))
# p_real = np.zeros((tsteps, 1))
# n_enter = np.zeros((tsteps, 1))
# n_exit = np.zeros((tsteps, 1))
# n_real = np.zeros((tsteps, 1))

# input_ = np.zeros((ny1, nx1, 2, tsteps))
# ii = 1


for ti in range(0, tsteps):
    with Timer('iteration_core'):
        valley, charge_n, charge_p, n_real, p_real, number = eng.iteration_core(
            valley, particles, number, fx, fy, p_icpg, n_icpg, left_pts, right_pts, scatGaAs, scatGaAs_hole,
            Gm, float(ti+1), nx1, ny1, xmax, ymax, qD, cpsp, nargout=6)

    # p_temp = 0
    # n_temp = 0
    # t = (ti - 1) * dt
    # tdt = t + dt
    # for n in range(max_particles):
    #     if valley[n, 0] != 9:
    #         ts = particles[n, 3]
    #         t1 = t
    #         while ts < tdt:
    #             tau = ts - t1
    #             iv = int(valley[n, 0])
    #             if iv != 9:
    #                 if iv == 1 or iv == 2:
    #                     kx = particles[n, 0]
    #                     ky = particles[n, 1]
    #                     kz = particles[n, 2]
    #                     x = particles[n, 4]
    #                     y = particles[n, 5]
    #                     i = int(min(np.floor(y / dy) + 1, ny1))
    #                     j = int(min(np.floor(x / dx) + 1, nx1))
    #                     i = max(i, 1)
    #                     j = max(j, 1)
    #                     dkx = - (q / h) * fx[i, j] * tau
    #                     dky = - (q / h) * fy[i, j] * tau
    #                     sk = kx * kx + ky * ky + kz * kz
    #                     gk = (h * h / (2 * eM[iv])) * sk * (1 / q)
    #                     x = x + (h / eM[iv]) * (kx + 0.5 * dkx) / (np.sqrt(1 + 4 * alpha[iv] * gk)) * tau
    #                     y = y + (h / eM[iv]) * (ky + 0.5 * dky) / (np.sqrt(1 + 4 * alpha[iv] * gk)) * tau
    #                     particles[n, 0] = kx + dkx
    #                     particles[n, 1] = ky + dky
    #                 else:
    #                     kx = particles[n, 0]
    #                     ky = particles[n, 1]
    #                     kz = particles[n, 2]
    #                     x = particles[n, 4]
    #                     y = particles[n, 5]
    #                     i = int(min(np.floor(y / dy) + 1, ny1))
    #                     j = int(min(np.floor(x / dx) + 1, nx1))
    #                     i = max(i, 1)
    #                     j = max(j, 1)
    #                     dkx = (q / h) * fx[i, j] * tau
    #                     dky = (q / h) * fy[i, j] * tau
    #                     if iv == 3:
    #                         kf = np.sqrt((kx + dkx) ** 2 + (ky + dky) ** 2 + kz * kz)
    #                         cos_theta = kz / kf
    #                         sin_theta = np.sqrt(1 - cos_theta ** 2)
    #                         sin_phi = ky / kf / sin_theta
    #                         cos_phi = (kx + dkx) / kf / sin_theta
    #                         g = ((B / A) ** 2 + (C / A) ** 2 * (sin_theta ** 2 * cos_theta ** 2 +
    #                             sin_theta ** 4 * cos_phi ** 2 * sin_phi ** 2)) ** 0.5
    #                         mh = emR / (abs(A) * (1 - g))
    #                         x = x + (h / mh) * (kx + 0.5 * dkx) * tau
    #                         y = y + (h / mh) * ky * tau
    #                         particles[n, 0] = kx + dkx
    #                         particles[n, 1] = ky + dky
    #                     elif iv == 4:
    #                         kf = np.sqrt((kx + dkx) ** 2 + (ky + dky) ** 2 + kz * kz)
    #                         cos_theta = kz / kf
    #                         sin_theta = np.sqrt(1 - cos_theta ** 2)
    #                         sin_phi = ky / kf / sin_theta
    #                         cos_phi = (kx + dkx) / kf / sin_theta
    #                         g = ((B / A) ** 2 + (C / A) ** 2 * (sin_theta ** 2 * cos_theta ** 2 +
    #                             sin_theta ** 4 * cos_phi ** 2 * sin_phi ** 2)) ** 0.5
    #                         ml = emR / (abs(A) * (1 + g))
    #                         ef = (h * h * abs(A) / (2 * emR)) * kf ** 2 * (1 / q) * (1 + g)
    #                         x = x + (h / ml) * (kx + 0.5 * dkx) * tau
    #                         y = y + (h / ml) * ky * tau
    #                         particles[n, 0] = kx + dkx
    #                         particles[n, 1] = ky + dky
    #                 # Boundary Condition -- the former change is incorrect,
    #                 # only kx or ky one has to change 921
    #                 if x < 0:
    #                     valley[n, 0] = 9
    #                     if iv == 1 or iv == 2:
    #                         p_temp = p_temp - 1
    #                     else:
    #                         p_temp = p_temp + 1
    #                 else:
    #                     if x > xmax:
    #                         valley[n, 0] = 9
    #                         if iv == 1 or iv == 2:
    #                             n_temp = n_temp + 1
    #                         else:
    #                             n_temp = n_temp - 1
    #                 if y > ymax:
    #                     y = ymax - (y - ymax)
    #                     particles[n, 1] = - particles[n, 1]
    #                 else:
    #                     if y < 0:
    #                         y = - y
    #                         particles[n, 1] = - particles[n, 1]
    #                 particles[n, 4] = x
    #                 particles[n, 5] = y
    #                 # Scatter----------------------------
    #                 if valley[n, 0] != 9:
    #                     # valley = dbl(valley)
    #                     with Timer('pn_scat_v2'):
    #                         particle, valley[n, 1] = eng.pn_scat_v2(dbl(particles[n, :].tolist()),
    #                             dbl(valley[n, 1].tolist()), dbl(scatGaAs.tolist()), dbl(scatGaAs_hole.tolist()), de,
    #                             q, h, dbl(eM), dbl(alpha), qD, hw0, A, B, C, emR, n, hwij, Egl, Egg,
    #                             hwe, g100, g111, nargout=2)
    #                     particle = np.array(particle)
    #                     valley = np.array(valley)
    #                     particles[n, :] = particle[0, :]
    #                 t1 = ts
    #                 ts = t1 - np.log(np.random.rand()) / Gm[iv]
    #             else:
    #                 ts = tdt
    #
    #         tau = tdt - t1
    #         iv = int(valley[n, 0])
    #         if iv != 9:
    #             if iv == 1 or iv == 2:
    #                 kx = particles[n, 0]
    #                 ky = particles[n, 1]
    #                 kz = particles[n, 2]
    #                 x = particles[n, 4]
    #                 y = particles[n, 5]
    #                 i = int(min(np.floor(y / dy) + 1, ny1))
    #                 j = int(min(np.floor(x / dx) + 1, nx1))
    #                 i = max(i, 1)
    #                 j = max(j, 1)
    #                 dkx = -(q / h) * fx[i, j] * tau
    #                 dky = -(q / h) * fy[i, j] * tau
    #                 sk = kx * kx + ky * ky + kz * kz
    #                 gk = (h * h / (2 * eM[iv])) * sk * (1 / q)
    #                 x = x + (h / eM[iv]) * (kx + 0.5 * dkx) / (np.sqrt(1 + 4 * alpha[iv] * gk)) * tau
    #                 y = y + (h / eM[iv]) * (ky + 0.5 * dky) / (np.sqrt(1 + 4 * alpha[iv] * gk)) * tau
    #                 particles[n, 0] = kx + dkx
    #                 particles[n, 1] = ky + dky
    #             else:
    #                 t0 = time.time()
    #                 kx = particles[n, 0]
    #                 ky = particles[n, 1]
    #                 kz = particles[n, 2]
    #                 x = particles[n, 4]
    #                 y = particles[n, 5]
    #                 i = int(min(np.floor(y / dy) + 1, ny1))
    #                 j = int(min(np.floor(x / dx) + 1, nx1))
    #                 i = max(i, 1)
    #                 j = max(j, 1)
    #                 # print(i, j)
    #                 dkx = (q / h) * fx[i, j] * tau
    #                 dky = (q / h) * fy[i, j] * tau
    #                 if iv == 3:
    #                     kf = np.sqrt((kx + dkx) ** 2 + (ky + dky) ** 2 + kz * kz)
    #                     cos_theta = kz / kf
    #                     sin_theta = np.sqrt(1 - cos_theta ** 2)
    #                     sin_phi = ky / kf / sin_theta
    #                     cos_phi = (kx + dkx) / kf / sin_theta
    #                     g = ((B / A) ** 2 + (C / A) ** 2 * (sin_theta ** 2 * cos_theta ** 2 +
    #                         sin_theta ** 4 * cos_phi ** 2 * sin_phi ** 2)) ** 0.5
    #                     mh = emR / (abs(A) * (1 - g))
    #                     x = x + (h / mh) * (kx + 0.5 * dkx) * tau
    #                     y = y + (h / mh) * ky * tau
    #
    #                     particles[n, 0] = kx + dkx
    #                     particles[n, 1] = ky + dky
    #                 else:
    #                     if iv == 4:
    #                         kf = np.sqrt((kx + dkx) ** 2 + (ky + dky) ** 2 + kz * kz)
    #                         cos_theta = kz / kf
    #                         sin_theta = np.sqrt(1 - cos_theta ** 2)
    #                         sin_phi = ky / kf / sin_theta
    #                         cos_phi = (kx + dkx) / kf / sin_theta
    #                         g = ((B / A) ** 2 + (C / A) ** 2 * (sin_theta ** 2 * cos_theta ** 2 +
    #                             sin_theta ** 4 * cos_phi ** 2 * sin_phi ** 2)) ** 0.5
    #                         ml = emR / (abs(A) * (1 + g))
    #                         ef = (h * h * abs(A) / (2 * emR)) * kf ** 2 * (1 / q) * (1 + g)
    #                         x = x + (h / ml) * (kx + 0.5 * dkx) * tau
    #                         y = y + (h / ml) * ky * tau
    #
    #                         particles[n, 0] = kx + dkx
    #                         particles[n, 1] = ky + dky
    #             # Boundary Condition-----the former change is incorrect, only kx or ky one has to change
    #             # 921----------------
    #             if x < 0:
    #                 valley[n, 0] = 9
    #                 if iv == 1 or iv == 2:
    #                     p_temp = p_temp - 1
    #                 else:
    #                     p_temp = p_temp + 1
    #             else:
    #                 if x > xmax:
    #                     valley[n, 0] = 9
    #                     if iv == 1 or iv == 2:
    #                         n_temp = n_temp + 1
    #                     else:
    #                         n_temp = n_temp - 1
    #             if y > ymax:
    #                 y = ymax - (y - ymax)
    #                 particles[n, 1] = - particles[n, 1]
    #             else:
    #                 if y < 0:
    #                     y = - y
    #                     particles[n, 1] = - particles[n, 1]
    #             particles[n, 4] = x
    #             particles[n, 5] = y
    #             particles[n, 3] = ts
    # Renew--------------------
    # with Timer('pn_renew_v6'):
    #     particles, valley, p_added, n_added, number = eng.pn_renew_v6(dbl(particles.tolist()), dbl(valley.tolist()),
    #         Ttot, dx, dy, nx1, ny1, float(max_particles), dbl(p_icpg.tolist()), dbl(n_icpg.tolist()), bk, T, q, h, dbl(alpha),
    #         dbl(eM), emR, dbl(Gm), tdt, dbl(left_pts.tolist()), dbl(right_pts.tolist()), Ltot, A, B, C, float(ti),
    #         dbl(number.tolist()), hd, nargout=5)
    #
    # particles = np.array(particles)
    # valley = np.array(valley)
    # number = np.array(number)
    #
    # p_real[ti, 0] = p_added - p_temp
    # n_real[ti, 0] = n_added - n_temp
    # # p_real is how many positive particles are injected in ti
    # # n_real is how many negative particles are injected in ti
    # # Charge Computation-------
    # with Timer('pn_charge_v2'):
    #     charge_p, charge_n = eng.pn_charge_v2(dbl(particles.tolist()),
    #         dbl(valley.tolist()), nx1, ny1, dx, dy, float(max_particles), cpsp, nargout=2)

    with Timer('pn_poisson_v5'):
        fx, fy, phi = eng.pn_poisson_v5(dx, dy, nx1, ny1, eps_stat, q, charge_p,
            charge_n, bg_charge, phi, Vp, Vn, nargout=3)
    # fx = np.array(fx)
    # fy = np.array(fy)
    # phi = np.array(phi)

    with Timer('statistics_core'):
        current_n, current_p, particle_num = eng.statistics_core(valley, particle_num, n_real, p_real,
            current_n, current_p, float(max_particles), float(ti+1), nargout=3)

    # valley_np = np.array(valley)
    # n_real_np = np.array(n_real)
    # p_real_np = np.array(p_real)
    # if ti == 0:
    #     current_n[ti, 0] = current_n[ti, 0]
    # else:
    #     current_n[ti, 0] = current_n[ti - 1, 0] + n_real_np[ti, 0]
    # # net anode current----
    # if ti == 0:
    #     current_p[ti, 0] = current_p[ti, 0]
    # else:
    #     current_p[ti, 0] = current_p[ti - 1, 0] + p_real_np[ti, 0]
    # index_v1 = (valley_np[:, 0] == 1)
    # index_v1 = index_v1[:, 0] * np.arange(max_particles)
    # index_v1 = index_v1 * (index_v1 != 0)
    # index_v2 = (valley_np[:, 0] == 2)
    # index_v2 = index_v2[:, 0] * np.arange(max_particles)
    # index_v2 = index_v2 * (index_v2 != 0)
    # index_v3 = (valley_np[:, 0] == 3)
    # index_v3 = index_v3[:, 0] * np.arange(max_particles)
    # index_v3 = index_v3 * (index_v3 != 0)
    # index_v4 = (valley_np[:, 0] == 4)
    # index_v4 = index_v4[:, 0] * np.arange(max_particles)  # ?
    # index_v4 = index_v4 * (index_v4 != 0)
    # particle_num[ti, :] = [len(index_v1), len(index_v2), len(index_v3), len(index_v4)]
    print('==> progress: %d finished.' % (ti+1))
    # print("==> Step{}".format(counter))
aa = Vp_all[0] * 100
bb = Vn_all[0] * 100
mtemp2 = 'Vp=' + str(aa) + 'Vn=' + str(bb) + 'dx=' + str(dx * 1000000000.0) + 'nm'
pickle.dump(mtemp2, open('results.pkl', 'wb+'))

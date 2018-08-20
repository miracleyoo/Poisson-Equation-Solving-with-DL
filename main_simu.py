# coding: utf-8
# Author: Zhongyang Zhang

import time
import pickle
import matlab.engine
from global_val import *
from dl_solver import *


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('==> [%s]:\t' % self.name, end='')
        print('Elapsed Time: %s (s)' % (time.time() - self.tstart))


def gen_input(Vn, Vp, nx1, ny1, net_charge):
    net_charge = np.array(net_charge)[np.newaxis, :]
    border_cond = np.zeros((1, int(ny1), int(nx1)))
    border_cond[0, :, 0] = Vn
    border_cond[0, :, -1] = Vp
    model_input = np.concatenate((net_charge, border_cond), axis=0)
    return model_input[np.newaxis, :]


eng = matlab.engine.start_matlab()
Vp_all = [0, 0.5, 1.2, 1.8, 2.25, 2.7]
Vn_all = [0]
USE_DL = True

with Timer('init_core'):
    nx1, ny1 = eng.init_core(matlab.double(Vp_all), matlab.double(Vn_all), nargout=2)

with Timer('init_dl_core'):
    opt, net = dl_init()

for a in range(len(Vp_all)):
    for b in range(len(Vn_all)):
        save_prefix = 'Vp=' + str(Vp_all[a] * 100) + 'Vn=' + str(Vn_all[b] * 100) + \
                      'dx=' + str(dx * 1000000000.0) + 'nm'
        if USE_DL:
            save_prefix += '_USE_DL'
        else:
            save_prefix += '_USE_SIM'
        save_name = save_prefix + '.mat'
        simu_res = []

        with Timer('subinit_core'):
            net_charge, Vp, Vn = eng.subinit_core(save_name, float(a + 1), float(b + 1), nargout=3)

        phi = matlab.double(np.zeros((int(ny1), int(nx1))).tolist())

        if USE_DL:
            with Timer('dl_solver'):
                model_input = gen_input(Vn, Vp, nx1, ny1, net_charge)
                phi = dl_solver(model_input, net, opt)
            simu_res.append((model_input, phi))
            phi = matlab.double(phi.squeeze().reshape(9, 41).tolist())
            fx, fy = eng.fxy_core(phi, nargout=2)
        else:
            with Timer('pn_poisson_v5'):
                fx, fy, phi = eng.pn_poisson_v5(phi, save_name, nargout=3)

        for ti in range(0, tsteps):
            with Timer('iteration_core'):
                eng.iteration_core(fx, fy, float(ti + 1), save_name, nargout=0)

            if USE_DL:
                with Timer('dl_solver'):
                    model_input = gen_input(Vn, Vp, nx1, ny1, net_charge)
                    phi = dl_solver(model_input, net, opt)
                simu_res.append((model_input, phi))
                phi = matlab.double(phi.squeeze().reshape(9, 41).tolist())
                fx, fy = eng.fxy_core(phi, nargout=2)
            else:
                with Timer('pn_poisson_v5'):
                    fx, fy, phi = eng.pn_poisson_v5(phi, save_name, nargout=3)

            with Timer('statistics_core'):
                eng.statistics_core(float(ti + 1), save_name, nargout=0)

            print('==> progress: %d finished.' % (ti + 1))
        print("==> Simulation Finished(Vn=%f,Vp=%f). %s file saved." % (Vn, Vp, save_name))
        pickle.dump(simu_res, open('./source/simulation_res/' + save_prefix + '.pkl', 'wb+'))

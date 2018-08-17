# coding: utf-8
# Author: Zhongyang Zhang

import time
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
Vp_all = [0]
Vn_all = [0]
USE_DL = False

with Timer('init_core'):
    nx1, ny1 = eng.init_core(nargout=2)

with Timer('init_dl_core'):
    opt, net = dl_init()

for a in range(len(Vp_all)):
    for b in range(len(Vn_all)):
        save_name = 'Vp=' + str(Vp_all[a] * 100) + 'Vn=' + str(Vn_all[b] * 100) +\
                    'dx=' + str(dx * 1000000000.0) + 'nm.mat'

        with Timer('subinit_core'):
            net_charge, phi, Vp, Vn = eng.subinit_core(save_name, float(a+1), float(b+1), nargout=4)

        if USE_DL:
            with Timer('dl_solver'):
                phi = dl_solver(gen_input(Vn, Vp, nx1, ny1, net_charge), net, opt)
                phi = matlab.double(phi.squeeze().reshape(9, 41).tolist())
                fx, fy = eng.fxy_core(phi, nargout=2)
        else:
            with Timer('pn_poisson_v5'):
                fx, fy, phi = eng.pn_poisson_v5(save_name, nargout=3)

        for ti in range(0, tsteps):
            with Timer('iteration_core'):
                eng.iteration_core(fx, fy, float(ti+1), save_name, nargout=0)

            if USE_DL:
                with Timer('dl_solver'):
                    phi = dl_solver(gen_input(Vn, Vp, nx1, ny1, net_charge), net, opt)
                phi = matlab.double(phi.squeeze().reshape(9, 41).tolist())
                fx, fy = eng.fxy_core(phi, nargout=2)
            else:
                with Timer('pn_poisson_v5'):
                    fx, fy, phi = eng.pn_poisson_v5(save_name, nargout=3)

            with Timer('statistics_core'):
                eng.statistics_core(float(ti+1), save_name, nargout=0)

            print('==> progress: %d finished.' % (ti+1))
        print("==> Simulation Finished(Vn=%f,Vp=%f). %s file saved." % (Vn, Vp, save_name))



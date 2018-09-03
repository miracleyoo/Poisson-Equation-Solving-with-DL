# coding: utf-8
# Author: Zhongyang Zhang
# Email : mirakuruyoo@gmail.com

import numpy as np
# -------Simulation Input Settings------------------------------------------
tsteps = 20000
max_particles = 400000
POISSON_ITER_MAX = 12000
dt = 1e-14
# include hole
dope_type = 1
cdop = 1e+23
cdop2 = 2e+20
cdop3 = 0
hd = 3
de = 0.002
T = 300.0
# --------Simulation Settings-----------------------------------------------
dx = 1e-08
dy = 1e-08
# --------Device Geomrety---------------------------------------------------
Ttot = 8e-08
Ltot = 4e-07
Lp = Ltot / 2
Ln = Ltot / 2
# ---------Constants--------------------------------------------------------
bk = 1.38066e-23
q = 1.60219e-19
h = 1.05459e-34
emR = 9.10953e-31
eps_o = 8.85419e-12
# ---------GaAs Specific Constants------------------------------------------
Eg = 1.424
Egg = 0
Egl = 0.29
emG = 0.067*emR
emL = 0.35*emR
emh = 0.62*emR
eml = 0.074*emR
alpha_G = (1 / Eg)*(1 - emG / emR) ** 2
alpha_L = (1 / (Eg + Egl))*(1 - emL / emR) ** 2

eC = [Egg, Egl]
eM = [emG, emL, emh, eml]
alphas = [alpha_G, alpha_L, 0.0, 0.0]

eps_stat = 12.9*eps_o
eps_inf = 10.92*eps_o
eps_p = 1 / ((1 / eps_inf) - (1 / eps_stat))
qD = float(np.sqrt(q*q*cdop / (eps_stat*bk*T)))
ni = 1.8e+12
contact_potential = float((bk*T / q)*(np.log(cdop*cdop / ni ** 2)))
hw0 = 0.03536
hwij = 0.03
hwe = hwij
# inverse band mass parameters
A = - 7.65
B = - 4.82
C = 7.7
g100 = B / A
g111 = float(np.sqrt((B / A) ** 2 + (C / A) ** 2 / 3))
ppc = 8

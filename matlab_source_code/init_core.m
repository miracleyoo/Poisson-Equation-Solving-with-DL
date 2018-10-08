% function [nx1, ny1, charge_p, charge_n, bg_charge, phi, Vp, Vn, xmax, ymax, qD, cpsp, scatGaAs, scatGaAs_hole, Gm, p_icpg, n_icpg, left_pts, right_pts, valley, particles] = init_core()
function [nx1, ny1] = init_core(Vp_all, Vn_all)
%-------Simulation Input Settings------------------------------------------
tsteps=20000;
dt=10e-15;
%include hole
dope_type=1;
cdop=1e23;
cdop2=2e20;
cdop3=0;%for substrate0.1
hd=3;%highly doped mear contact
max_particles=40e4;%40e4;
de=0.002;
T=300;
POISSON_ITER_MAX=12000;
ppc=8;

%--------Simulation Settings-----------------------------------------------
dx=1e-8;
dy=1e-8;

%--------Device Geomrety---------------------------------------------------
Ttot=0.08e-6;
Ltot=0.4e-6;
Lp=Ltot/2;
Ln=Ltot/2;

%---------Constants--------------------------------------------------------
bk=1.38066e-23;                 %Boltzmann's Constant 
q=1.60219e-19;                  %Charge of Electron
h=1.05459e-34;                  %Planck's Constant (/2pi)
emR=9.10953e-31;                %Mass of Resting Electron
eps_o=8.85419e-12;              %Vacuum Permittivity

%---------GaAs Specific Constants------------------------------------------
Eg=1.424;                       %Band Gap for GaAs
Egg=0;                          %Energy difference between two generate Gamma Bands
Egl=0.29;                       %Energy difference between Gamma and L Bands
eC=[Egg Egl];
emG=0.067*emR;                  %Mass of Electron in Gamma Band
emL=0.350*emR;                  %Mass of Electron in L Band
emh=0.62*emR;
eml=0.074*emR;
eM=[emG,emL,emh,eml];
alpha_G=(1/Eg)*(1-emG/emR)^2;   %Alpha for Non-Parabolicity of Gamma Band
alpha_L=(1/(Eg+Egl))*(1-emL/emR)^2;%Alpha for Non-Parabolicty of L Band
alphas=[alpha_G,alpha_L,0,0];%for holes, alpha=0(assume parabolic)

eps_stat=12.9*eps_o;            %Static Permittivity for GaAs
eps_inf=10.92*eps_o;            %Optical Permittivity for GaAs
eps_p=1/((1/eps_inf)-(1/eps_stat)); 

qD=sqrt(q*q*cdop/(eps_stat*bk*T)); %Inverse Debye Length

ni=1.8e12;%GaAs at 300k
contact_potential=bk*T/q*log(cdop*cdop/ni^2);

hw0=0.03536;
hwij=0.03;
hwe=hwij;

%inverse band mass parameters
A=-7.65;
B=-4.82;
C=7.7;
g100=B/A;
g111=sqrt((B/A)^2+(C/A)^2/3);

%---------Create Scattering Table------------------------------------------
[scatGaAs,GmG,GmL]=make_GaAs_scatTable(T,0,de,2,cdop);%change 2 to Vmax, 108
[scatGaAs_hole,Gmh,Gml]=make_GaAs_hole_scatTable_v2(T,de,2,cdop);
Gm=[GmG,GmL,Gmh,Gml];

%------------related to configuration--------------------------------------
nx1=round(Ltot/dx)+1;
nx=nx1-1;

ny1=round(Ttot/dy)+1;
ny=ny1-1;
left_pts=1:nx1:nx1*ny1;
right_pts=nx1:nx1:nx1*ny1;
p_icpg(length(left_pts))=0;
n_icpg(length(right_pts))=0;
save('./source/simulation_res/intermediate_file/init_data.mat');
end


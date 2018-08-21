function [net_charge, Vp, Vn] = subinit_core(save_name, a, b)
%UNTITLED5 此处显示有关此函数的摘要
%   此处显示详细说明
temp_a = a;
temp_b = b;
temp_save_name = save_name;
load('init_data.mat');
a = temp_a;
b = temp_b;
save_name = temp_save_name;
clear temp_a;
clear temp_b;
clear temp_save_name;

Vp=Vp_all(a)-contact_potential/2;
Vn=Vn_all(b)+contact_potential/2;

%---------Initialize Particles---------------------------------------------
[particles,valley,bg_charge,cpsp,N,p_icpg,n_icpg,xmax,ymax]=pn_init_v2(max_particles,dope_type,dx,dy,Ltot,nx1,ny1,cdop,ppc,bk,T,q,h,alphas,eM,Gm,Lp,A,B,C,emR,hd);

%--------Initial Charge/Field Computations---------------------------------
[charge_p,charge_n]=pn_charge_v2(particles,valley,nx1,ny1,dx,dy,max_particles,cpsp);%净电子数
% phi=zeros(ny1,nx1);

number=zeros(tsteps,4);

%calculate current
p_enter=zeros(tsteps,1);
p_exit=zeros(tsteps,1);
p_real=zeros(tsteps,1);
n_enter=zeros(tsteps,1);
n_exit=zeros(tsteps,1);
n_real=zeros(tsteps,1);
current_n=zeros(tsteps,1);
current_p=zeros(tsteps,1);
net_charge=bg_charge-charge_n+charge_p;
save(save_name);
end


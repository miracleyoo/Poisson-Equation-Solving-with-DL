% function [valley, charge_n, charge_p, n_real, p_real, number] = iteration_core(valley, particles, number, fx, fy, p_icpg, n_icpg, left_pts, right_pts, scatGaAs, scatGaAs_hole, Gm, ti, nx1, ny1, xmax, ymax, qD, cpsp)
function [] = iteration_core(fx, fy, ti, save_name)

%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
load(save_name);

dx = 1e-08;
dy = 1e-08;
dt = 1e-14;

hd=3;%highly doped mear contact
max_particles=40e4;%55e4;
de=0.002;
T=300;

%--------Device Geomrety---------------------------------------------------
Ttot=0.08e-6;
Ltot=0.4e-6;

%---------Constants--------------------------------------------------------
bk=1.38066e-23;                 %Boltzmann's Constant 
q=1.60219e-19;                  %Charge of Electron
h=1.05459e-34;                  %Planck's Constant (/2pi)
emR=9.10953e-31;                %Mass of Resting Electron

%---------GaAs Specific Constants------------------------------------------
Eg=1.424;                       %Band Gap for GaAs
Egg=0;                          %Energy difference between two generate Gamma Bands
Egl=0.29;                       %Energy difference between Gamma and L Bands
emG=0.067*emR;                  %Mass of Electron in Gamma Band
emL=0.350*emR;                  %Mass of Electron in L Band
emh=0.62*emR;
eml=0.074*emR;
eM=[emG,emL,emh,eml];
alpha_G=(1/Eg)*(1-emG/emR)^2;   %Alpha for Non-Parabolicity of Gamma Band
alpha_L=(1/(Eg+Egl))*(1-emL/emR)^2;%Alpha for Non-Parabolicty of L Band
alphas=[alpha_G,alpha_L,0,0];%for holes, alpha=0(assume parabolic)

hw0=0.03536;
hwij=0.03;
hwe=hwij;

%inverse band mass parameters
A=-7.65;
B=-4.82;
C=7.7;
g100=B/A;
g111=sqrt((B/A)^2+(C/A)^2/3);

p_temp=0;
n_temp=0;
t=(ti-1)*dt;
tdt=t+dt;
for n=1:max_particles
    if valley(n,1)~=9
        ts=particles(n,4);
        t1=t;
        while ts<tdt
            tau=ts-t1;
            %Drift------------------------------
            iv=valley(n,1);
            if iv~=9
                if iv==1||iv==2
                    kx=particles(n,1);
                    ky=particles(n,2);
                    kz=particles(n,3);
                    x=particles(n,5);
                    y=particles(n,6);

                    i=min(floor(y/dy)+1,ny1);
                    j=min(floor(x/dx)+1,nx1);
                    i=max(i,1);
                    j=max(j,1);%928

                    dkx=-(q/h)*fx(i,j)*tau;%the field can be more accurate by interprolate
                    dky=-(q/h)*fy(i,j)*tau;
                    sk=kx*kx+ky*ky+kz*kz;
                    gk=(h*h/(2*eM(iv)))*sk*(1/q);

                    x=x+(h/eM(iv))*(kx+0.5*dkx)/(sqrt(1+4*alphas(iv)*gk))*tau;
                    y=y+(h/eM(iv))*(ky+0.5*dky)/(sqrt(1+4*alphas(iv)*gk))*tau;

                    particles(n,1)=kx+dkx;  %kx
                    particles(n,2)=ky+dky;  %ky
                else
                    kx=particles(n,1);
                    ky=particles(n,2);
                    kz=particles(n,3);
                    x=particles(n,5);
                    y=particles(n,6);

                    i=min(floor(y/dy)+1,ny1);
                    j=min(floor(x/dx)+1,nx1);
                    i=max(i,1);
                    j=max(j,1);

                    dkx=(q/h)*fx(i,j)*tau;
                    dky=(q/h)*fy(i,j)*tau;                                            

                    if iv==3%heavy                                    
                        kf=sqrt((kx+dkx)^2+(ky+dky)^2+kz*kz);        
                        cos_theta=kz/kf;
                        sin_theta=sqrt(1-cos_theta^2);
                        sin_phi=ky/kf/sin_theta;
                        cos_phi=(kx+dkx)/kf/sin_theta;
                        g=((B/A)^2+(C/A)^2*(sin_theta^2*cos_theta^2+sin_theta^4*cos_phi^2*sin_phi^2))^0.5;   
                        mh=emR/(abs(A)*(1-g));
                        x=x+(h/mh)*(kx+0.5*dkx)*tau;
                        y=y+(h/mh)*(ky)*tau;
                        particles(n,1)=kx+dkx;  %kx
                        particles(n,2)=ky+dky;  %ky
                    elseif iv==4
                        kf=sqrt((kx+dkx)^2+(ky+dky)^2+kz*kz);
                        cos_theta=kz/kf;
                        sin_theta=sqrt(1-cos_theta^2);
                        sin_phi=ky/kf/sin_theta;
                        cos_phi=(kx+dkx)/kf/sin_theta;
                        g=((B/A)^2+(C/A)^2*(sin_theta^2*cos_theta^2+sin_theta^4*cos_phi^2*sin_phi^2))^0.5;   
                        ml=emR/(abs(A)*(1+g));
                        x=x+(h/ml)*(kx+0.5*dkx)*tau;
                        y=y+(h/ml)*(ky)*tau;
                        particles(n,1)=kx+dkx;  %kx
                        particles(n,2)=ky+dky;  %ky
                    end
                end   
                %Boundary Condition-----the former change is incorrect, only kx or ky one has to change 921----------------
                if x < 0
                    valley(n,1)=9;
                    if iv==1||iv==2
                        p_temp=p_temp-1;%p_temp count how mant positive particles are deleted when drifting
                    else
                        p_temp=p_temp+1;
                    end
                elseif x > xmax
                    valley(n,1)=9; 
                    if iv==1||iv==2
                        n_temp=n_temp+1;%n_temp count how mant negative particles are deleted when drifting
                    else
                        n_temp=n_temp-1;
                    end
                end

                if y > ymax
                    y=ymax-(y-ymax);
                    particles(n,2)=-particles(n,2);
                elseif y < 0
                    y=-y;
                    particles(n,2)=-particles(n,2);
                end

                particles(n,5)=x;       %x
                particles(n,6)=y;       %y

                %Scatter----------------------------
                if valley(n,1)~=9
                    [particle,valley(n,1)]=pn_scat_v2(particles(n,:),valley(n,1),scatGaAs,scatGaAs_hole,de,q,h,eM,alphas,qD,hw0,A,B,C,emR,n,hwij,Egl,Egg,hwe,g100,g111);
                    particles(n,:)=particle(1,:);
                end

                t1=ts;
                ts=t1-log(rand())/Gm(iv);%can be more accurate
            else
                ts=tdt;
            end
        end

        tau=tdt-t1;
        %Drift------------------------------------
        iv=valley(n,1);
        if iv~=9                        
            if iv==1||iv==2
                kx=particles(n,1);
                ky=particles(n,2);
                kz=particles(n,3);
                x=particles(n,5);
                y=particles(n,6);

                i=min(floor(y/dy)+1,ny1);
                j=min(floor(x/dx)+1,nx1);
                i=max(i,1);
                j=max(j,1);%928

                dkx=-(q/h)*fx(i,j)*tau;%the field can be more accurate by interprolate
                dky=-(q/h)*fy(i,j)*tau;
                sk=kx*kx+ky*ky+kz*kz;
                gk=(h*h/(2*eM(iv)))*sk*(1/q);

                x=x+(h/eM(iv))*(kx+0.5*dkx)/(sqrt(1+4*alphas(iv)*gk))*tau;
                y=y+(h/eM(iv))*(ky+0.5*dky)/(sqrt(1+4*alphas(iv)*gk))*tau;

                particles(n,1)=kx+dkx;  %kx
                particles(n,2)=ky+dky;  %ky
            else
                kx=particles(n,1);
                ky=particles(n,2);
                kz=particles(n,3);
                x=particles(n,5);
                y=particles(n,6);

                i=min(floor(y/dy)+1,ny1);
                j=min(floor(x/dx)+1,nx1);
                i=max(i,1);
                j=max(j,1);

                dkx=(q/h)*fx(i,j)*tau;
                dky=(q/h)*fy(i,j)*tau;     

                if iv==3%heavy                                    
                    kf=sqrt((kx+dkx)^2+(ky+dky)^2+kz*kz);        
                    cos_theta=kz/kf;
                    sin_theta=sqrt(1-cos_theta^2);
                    sin_phi=ky/kf/sin_theta;
                    cos_phi=(kx+dkx)/kf/sin_theta;
                    g=((B/A)^2+(C/A)^2*(sin_theta^2*cos_theta^2+sin_theta^4*cos_phi^2*sin_phi^2))^0.5;   
                    mh=emR/(abs(A)*(1-g));
                    x=x+(h/mh)*(kx+0.5*dkx)*tau;
                    y=y+(h/mh)*(ky)*tau;
                    particles(n,1)=kx+dkx;  %kx
                    particles(n,2)=ky+dky;  %ky
                elseif iv==4
                    kf=sqrt((kx+dkx)^2+(ky+dky)^2+kz*kz);
                    cos_theta=kz/kf;
                    sin_theta=sqrt(1-cos_theta^2);
                    sin_phi=ky/kf/sin_theta;
                    cos_phi=(kx+dkx)/kf/sin_theta;
                    g=((B/A)^2+(C/A)^2*(sin_theta^2*cos_theta^2+sin_theta^4*cos_phi^2*sin_phi^2))^0.5;   
                    ml=emR/(abs(A)*(1+g));
                    x=x+(h/ml)*(kx+0.5*dkx)*tau;
                    y=y+(h/ml)*(ky)*tau;
                    particles(n,1)=kx+dkx;  %kx
                    particles(n,2)=ky+dky;  %ky
                end

            end   
            %Boundary Condition-----the former change is incorrect, only kx or ky one has to change 921----------------
            if x < 0
                valley(n,1)=9;
                if iv==1||iv==2
                    p_temp=p_temp-1;%p_temp count how mant positive particles are deleted when drifting
                else
                    p_temp=p_temp+1;
                end
            elseif x > xmax
                valley(n,1)=9; 
                if iv==1||iv==2
                    n_temp=n_temp+1;%n_temp count how mant negative particles are deleted when drifting
                else
                    n_temp=n_temp-1;
                end
            end

            if y > ymax
                y=ymax-(y-ymax);
                particles(n,2)=-particles(n,2);
            elseif y < 0
                y=-y;
                particles(n,2)=-particles(n,2);
            end

            particles(n,5)=x;       %x
            particles(n,6)=y;       %y 

            particles(n,4)=ts;      %ts
        end
    end
end              

%Renew--------------------
[particles,valley,p_added,n_added,number]=pn_renew_v6(particles,valley,Ttot,dx,dy,nx1,ny1,max_particles,p_icpg,n_icpg,bk,T,q,h,alphas,eM,emR,Gm,tdt,left_pts,right_pts,Ltot,A,B,C,ti,number,hd);
p_real(ti,1)=p_added-p_temp;%p_added is how many positive particles are injected in renew
n_real(ti,1)=n_added-n_temp;%n_added is how many negative particles are injected in renew
%p_real is how many positive particles are injected in ti
%n_real is how many negative particles are injected in ti

%Charge Computation-------
[charge_p,charge_n]=pn_charge_v2(particles,valley,nx1,ny1,dx,dy,max_particles,cpsp);
save(save_name);
end


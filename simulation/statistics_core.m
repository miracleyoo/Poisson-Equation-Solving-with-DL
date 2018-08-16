function [] = statistics_core(ti, save_name)
%UNTITLED4 此处显示有关此函数的摘要
%   此处显示详细说明
%net cathode current----
load(save_name);

if ti==1
    current_n(ti,1)=current_n(ti,1);
else
    current_n(ti,1)=current_n(ti-1,1)+n_real(ti,1);
end
%net anode current----
if ti==1
    current_p(ti,1)=current_p(ti,1);
else
    current_p(ti,1)=current_p(ti-1,1)+p_real(ti,1);
end

index_v1=valley(:,1)==1;
index_v1=index_v1(:,1).*(1:max_particles).';
index_v1=index_v1(index_v1~=0);
% 
index_v2=valley(:,1)==2;
index_v2=index_v2(:,1).*(1:max_particles).';
index_v2=index_v2(index_v2~=0);

index_v3=valley(:,1)==3;
index_v3=index_v3(:,1).*(1:max_particles).';
index_v3=index_v3(index_v3~=0);

index_v4=valley(:,1)==4;
index_v4=index_v4(:,1).*(1:max_particles).';
index_v4=index_v4(index_v4~=0);

particle_num(ti,:)=[length(index_v1),length(index_v2),length(index_v3),length(index_v4)];
save(save_name);
end


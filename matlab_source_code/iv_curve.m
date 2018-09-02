% function [] = iv_curve()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明

addpath ../Datasets/
addpath ../matlab_source_code/
addpath ../source/simulation_res/intermediate_file/
% ivs = containers.Map;
vs = [];
ivs = [];
ML_IV = true;

if ML_IV
    filenames = dir('../source/simulation_res/intermediate_file/');
    pic_title = "Deep Learning I-V Curve";
    suffix = "Deep Learning";
else
    filenames = dir('../Datasets/');
    pic_title = "Traditional Simulation I-V Curve";
    suffix = "Tradition Method";
end
filenames = {filenames.name};
filenames = string(filenames);
STTF = startsWith(filenames,'.');
filenames = filenames(~STTF);
MTF = contains(filenames,'.mat');
filenames = filenames(MTF);
filenum = length(filenames);

for filename = filenames
    load(filename);
    if Vp>2.0
        ti=10800;
    else
        ti=20000;
    end
    slope=draw_curve(Vp, suffix, ti, current_p, current_n);
    vs = [vs, Vp];
    ivs = [ivs, slope];
%     ivs(string(Vp))=slope;
end
[vs, id] = sort(vs);
ivs = ivs(id);
% figure
hold on;
plot(vs,ivs,'r:','LineWidth',2);
ylim([-1 200])
title(pic_title);
disp(ivs);

% end


addpath('../source/simulation_res/intermediate_file/');
addpath('../Datasets/')
filenames = dir('../source/simulation_res/intermediate_file/');
filenames = {filenames.name};
filenames = string(filenames);
STTF = startsWith(filenames,'.');
filenames = filenames(~STTF);
MTF = contains(filenames,'.mat');
filenames = filenames(MTF);
filenum = length(filenames);
slope = [];
Xs = [];

titles = ['I-V Curve ','USE DL'];
ti=20000;
for filename = filenames
    load(filename)
    Xs = [Xs, Vp];
    slope = [slope, draw_i(titles, ti, current_p, current_n)];
end
[Xs,index] = sort(Xs);
slope = slope(index);
plot(Xs,slope)

title('Final I-V curve')

function [fx, fy] = fxy_core(phi)
%UNTITLED6 此处显示有关此函数的摘要
%   此处显示详细说明
% temp_phi = phi;
% load(save_name)
% phi = temp_phi;
ny1=9;
nx1=41;
dx=1e-8;
dy=1e-8;

fx(ny1,nx1)=0;
for i=1:ny1
    for j=2:(nx1-1)
        fx(i,j)=((phi(i,j-1)-phi(i,j+1))/dx)/2;%924
        %because E=-d(phi)/dx!!
    end
end
%for left and right boundary
fx(:,1)=fx(:,2);
fx(:,end)=fx(:,end-1);

fy(ny1,nx1)=0;       
for j=1:nx1
    for i=2:(ny1-1)
        fy(i,j)=(phi(i-1,j)-phi(i+1,j))/(2*dy);        
    end
end
%for top and bottom boundary
fy(1,:)=fy(2,:);
fy(end,:)=fy(end-1,:);
% save(save_name)
end


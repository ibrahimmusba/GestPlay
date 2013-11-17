function [Ixgrad, Iygrad, Gmag, Gdir] = igradient(I)
%igradient:
% I/P: Color Image

[height, width, channels] = size(I);

hx = [-0.5 0 0.5];
hy = hx';

Ixgrad = imfilter(double(I),hx,'replicate');
Iygrad = imfilter(double(I),hy,'replicate');

Gmag = sqrt(Ixgrad.^2 + Iygrad.^2);
Gdir = atan2(double(Iygrad), double(Ixgrad));
max(max(max(Gdir)))
min(min(min(Gdir)))
if channels == 3
    [Gmag ind] = max(Gmag,[],3);
    Gdir = Gdir(:,:,1).*(ind==1) + Gdir(:,:,2).*(ind==2) + Gdir(:,:,3).*(ind==3);
end

end
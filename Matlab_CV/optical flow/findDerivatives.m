function [Ix Iy It] = findDerivatives(imCur,initialize);
% this function finds the derivative along x, y , t; 
persistent imPrev ;

if initialize ==1
    imPrev = single(imCur);
end
 
imCur = single(imCur);

siz = 3; %window size
std = 0.67; % standard deviation. value as suggested in tutorial
x2 = (-(siz-1)/2:(siz-1)/2).^2;
g = exp(-(x2)/(2*std*std));
g = g/sum(sum(g)); % generate normalized Gaussian window

Gauss   = single(g'*g); % a gaussian window
Gradx =  [-1 0 1; 
         -1 0 1;
         -1 0 1];
Gradx = single(-Gradx.*Gauss*3);
Grady = Gradx';

% tic
imCur = conv2(imCur, Gauss,'same');


Ix = conv2(imCur, Gradx,'same');
Iy = conv2(imCur, Grady,'same');
It = imCur - imPrev;
% toc
imPrev = imCur; 
end
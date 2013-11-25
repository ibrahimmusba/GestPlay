function [Vx Vy] = findMotion(Ix, Iy,It, Resolution)
% this function finds the optical flow and returns the velocity vectors 



%% gauss window generation 

std = 2; % standard deviation. value as suggested in tutorial
siz = floor(5*std) ; 
siz = siz + mod(siz,2); % make it odd
x2 = (-(siz-1)/2:(siz-1)/2).^2;
g = exp(-(x2)/(2*std*std));
g = g/sum(sum(g)); % 1-D gauss 

%% implementation of LK method 
% we know It = -A*v , where A = [Ix Iy], v= [vx vy]
% use regularized least squares solution

% compute moments: 
% it is done in this way because it is computationally efficient
 m200=resizeImage(conv2(g,g, Ix.^2 ,'same'),[Resolution Resolution]);
 m020=resizeImage(conv2(g,g, Iy.^2 ,'same'),[Resolution Resolution]);
 m110=resizeImage(conv2(g,g, Ix.*Iy,'same'),[Resolution Resolution]);
 m101=resizeImage(conv2(g,g, Ix.*It,'same'),[Resolution Resolution]);
 m011=resizeImage(conv2(g,g, Iy.*It,'same'),[Resolution Resolution]);

% here the equation is b = -S*v;
% whre b = [m101; m011] and S = [m200 m110; m110 m020]; 

RegConst = 100; % regularization constant 

% add const to the diagonal
m200 = m200 + RegConst ;
m020 = m020 + RegConst ;

% analytically compute the inverse and solve least squares problem
Vx =(-m101.*m020 + m011.*m110)./(m020.*m200 - m110.^2);
Vy =( m101.*m110 - m011.*m200)./(m020.*m200 - m110.^2);

end


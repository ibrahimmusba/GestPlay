function plotHistogram( hist, y, x)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

direction = [10:20:170]*pi/180 + pi/2;

r = 3;

[maxVal maxInd] = max(hist);

xOffset = r*cos(direction(maxInd));
yOffset = r*sin(direction(maxInd));

linesegment = [x + xOffset , y + yOffset; x - xOffset , y - yOffset ] ;

plot(linesegment(:,1),linesegment(:,2),'r','LineWidth',min(((maxVal/50)+0.01),6)  );

end


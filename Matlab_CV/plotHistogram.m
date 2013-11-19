function plotHistogram( cellMag, cellDir, x, y )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[width, height] = size(cellMag);

%Find the index of max magnitude
[valr, maxrows] = max(cellMag);
[valc, maxcol] = max(valr);
maxrow = maxrows(maxcol);

r = 3;


%plot all 
cellMagUnit = cellMag ./ valc;

% xOffsets = r.*cellMagUnit.*cos(cellDir);
% yOffsets = r.*cellMagUnit.*sin(cellDir);
% for i = 1:length(xOffsets(:))
%     linesegments = [x + xOffsets(i), y + yOffsets(i); x - xOffsets(i), y - yOffsets(i)];
%     plot(linesegments(:,1),linesegments(:,2));
% end

if cellMag(maxrow, maxcol) > 40
xOffset = r*cos(cellDir(maxrow, maxcol));
yOffset = r*sin(cellDir(maxrow, maxcol));

linesegment = [x + xOffset , y + yOffset; x - xOffset , y - yOffset ] ;

plot(linesegment(:,1),linesegment(:,2),'r','LineWidth',2);
end
end


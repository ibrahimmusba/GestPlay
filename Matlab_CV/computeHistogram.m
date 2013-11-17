function h = computeHistogram(cellMag, cellDir, numBins)

h = zeros(1,numBins);

% The bins will be 0 to 180 (unsigned)
cellDir = abs(cellDir);
binSize = pi/numBins;

%The bin to which the gradient should fall. 
%This number is decimal points. The vote will be decided on which int
%number id closer.
bin = cellDir/binSize;

%For each decimal bin value there will be two bins to which it will
%contribute. We will calculte the lower bin percentage and higher.
% For example if bin = 1.2, the contribution will be 80% to lower bin(1)
% and 20% to higher bin(2). We need to circular round off. if value is 0.8
% we need to contribute to 1 and 0, which will be rounded off to 9
lower_bin = mod(floor(bin),numBins) + 1;
upper_bin = mod(ceil(bin),numBins) + 1;

perc_lower = ceil(bin) - bin;
perc_upper = bin - floor(bin);



%Can this ve vectorized?
for i = 1:length(cellMag)
    h(upper_bin(i)) = cellMag(i)*perc_upper(i);
    h(lower_bin(i)) = cellMag(i)*perc_lower(i);
end

end
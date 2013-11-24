function [H, blockHistogram] = HoG(I)
%Input Image

shouldPlot = false;

if shouldPlot
    figure(1);
    imshow(I);
    hold on;
end

H = [];

% Values taken from the HoG paper
cellSize  = 8;
blockSize = 16;
numBins = 9;

[imHeight, imWidth, numChannels] = size(I);

if ( mod(imHeight,cellSize)~=0 || mod(imWidth,cellSize)~=0 )
    sprintf('Image size must be multiple of cellSize (8 pixels).\n');
    return;
end

numColCells = imWidth / cellSize;
numRowCells  = imHeight / cellSize;

%%
% Step1: Gamma/Color Normalization


%%
% Step2: Gradient Calculation
[Gx, Gy, Gmag, Gdir] = igradient(I);


%%
% Step3: Orientation Binning
numColBlocks = numColCells - 1;
numRowBlocks = numRowCells - 1;

cellHistogram = zeros(numRowCells, numColCells, numBins);

for i = 1:numRowCells
  
    iInd = ((i-1)*cellSize)+1;
    
    for j = 1:numColCells

        jInd = ((j-1)*cellSize)+1;
        rows = iInd : iInd + cellSize - 1;
        cols = jInd : jInd + cellSize - 1;
        
        cellMag = Gmag(rows, cols);
        cellDir = Gdir(rows, cols);
        
        cellHistogram(i,j,:) = computeHistogram(cellMag(:), cellDir(:), numBins);
        
        if shouldPlot
            plotHistogram(cellHistogram(i,j,:), rows(round(end/2)), cols(round(end/2)));
        end
    end
end

%%
% Step4: Block Normalization

blockHistogram = zeros(numRowCells-1, numColCells-1, numBins*3);

for i = 1:numRowCells-1
    for j = 1:numColCells-1
        blockHistogramSingle = cellHistogram(i:i+1, j:j+1, :);
        mag = norm(blockHistogramSingle(:));
        
        if (mag~=0)
            blockHistogramSingle = blockHistogramSingle./mag;
        end
        
        blockHistogram(i,j,:) = blockHistogramSingle(:);
        
        %This is the histogram feature vector 
        H = [H; blockHistogramSingle(:)]; %Column Vector
    end
end

end

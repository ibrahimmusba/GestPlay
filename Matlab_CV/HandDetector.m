%Hand Detector

img = imread('6.jpg');

[H, blockH] = HoG(img);

[numRowBlockFull numColBlockFull numFeatures] = size(blockH);

cellSize = 8;
imWidth = 88;
imHeight = 112;
numColCells = imWidth / cellSize;
numRowCells  = imHeight / cellSize;

numColBlocks = numColCells - 1;
numRowBlocks = numRowCells - 1;

numRowResponse = numRowBlockFull - numRowBlocks + 1;
numColResponse = numColBlockFull - numColBlocks + 1;

response = zeros(numRowResponse, numColResponse);

%Slide the window and check the response
for i = 1:numRowResponse
    for j = 1:numColResponse
        rows = i:i + numRowBlocks - 1;
        cols = j:j + numColBlocks - 1;
        HwindowTmp = blockH(rows, cols);
        Hwindow = reshapHwindowTmp
    end
end


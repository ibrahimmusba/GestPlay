%Hand Detector

img = imread('6.jpg');
img = imresize(img, 0.5);
tic
[H1, blockH] = HoG(img);
toc
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

H = [];
%Slide the window and check the response
for i = 1:numRowResponse
    disp(i)
    for j = 1:numColResponse
        rows = i:i + numRowBlocks - 1;
        cols = j:j + numColBlocks - 1;
        Hwindow = [];
        for ii = 1:length(rows)
            for jj = 1:length(cols)
                tmp = blockH(rows(ii),cols(jj),:);
                Hwindow = [Hwindow; tmp(:)];
            end
        end
        
        H = [H, Hwindow];
        if (classifySVM(Hwindow,w,b,X_support) == 1)
            response(i,j) = 1;
        end
    end
end
tic
y = classifySVM(H,w,b,X_support)
toc
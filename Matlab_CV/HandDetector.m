%Hand Detector

img = imread('43.jpg');
img = imresize(img, 0.8);
tic
[H1, blockH] = HoG(img);

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

% figure;
% imshow(img);
% hold on;
imgnum = 1;

destinationfolder ='D:\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset\Processed\random';

H = [];
%Slide the window and check the response
for i = 1:numRowResponse
%     disp(i)
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
         if (classifySVM(Hwindow,w,b,X_support) > 0)
            response(i,j) = abs(classifySVM(Hwindow,w,b,X_support));
            r = [(i).*8-7, (i+numRowBlocks).*8];
            c = [(j).*8-7, (j+numColBlocks).*8];
            imgtmp = img(r(1):r(2), c(1):c(2), :);
            imwrite(imgtmp,[destinationfolder '\' num2str(imgnum) '.jpg'],'jpg');
            imgnum = imgnum+1;
%             r = rows.*8;
%             c = cols.*8;
%             rectCorners = [r(1) r(1) r(1)+r(end) r(1)+r(end) r(1) ;
%                           c(1) c(1)+c(end)  c(1)+c(end) c(1) c(1)];
%             plot(rectCorners(2,:),rectCorners(1,:));
         end
    end
end

y = classifySVM(H,w,b,X_support)
toc

%close all;

[m indr] = max(response);
[m indc] = max(m);
indr = indr(indc);
if m~=0
    response = response./m;
end


%plot max response rectangle            
r = [(indr).*8-7, (indr+numRowBlocks).*8-7];
c = [(indc).*8-7, (indc+numColBlocks).*8-7];

figure;
imshow(img);
hold on;
rectCorners = [r(1) r(1) r(end) r(end) r(1) ;
               c(1) c(end)  c(end) c(1) c(1)];
plot(rectCorners(2,:),rectCorners(1,:));

        
[xx yy] = find(response~=0);
plot( yy.*8 + 88/2 - 7, xx.*8 + 112/2 - 7,'*');

figure;
imshow(response);
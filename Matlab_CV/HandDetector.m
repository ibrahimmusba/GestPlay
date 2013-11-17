
%Each column is a feature
% For 46x64 images, num dimensions = 1260
d = 1260;
n = 1000;

H = []
for i = i:numImages
    H = [H, HoG(I)];
end

%Train a SVM on the data




    
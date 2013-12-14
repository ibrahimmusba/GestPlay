function [X_train Y_train X_test Y_test IMG IMG_labels H_train H_test] = loadHandDataBinaryClass(folder_pos, folder_neg,wantGray, wantHoG)
%folder_pos and folder_neg are folder names which contain all the image
%files. The images should all be of the same dimension. folder_neg can be
%an array of folder names having negative images

if ~exist('wantHoG', 'var')
    wantHoG = 0;
end

X = []; %Feature Vectors
Y = []; %Output Labels
H = [];
IMG = [];
ind = 1;

n_total = 0;

%% Read all positive examples
fprintf('Reading all Positive Examples... \n')
for i = 1:length(folder_pos)
    posImageFiles = dir([folder_pos{i} '\' '*.jpg']); 
    for k = 1:length(posImageFiles)
        filename = posImageFiles(k).name;
        img = imread([folder_pos{i} '\' filename]);
        if wantHoG
            H = [H, HoG(img)];
        end
        if (size(img,3) == 3 && wantGray)
            img = rgb2gray(img);
        end
        IMG(:,:,:,ind) = img; ind = ind + 1;
        X = [X, double(img(:))];
    end
    Y = [Y, ones(1,length(posImageFiles))];
    n_total = n_total + length(posImageFiles);
    fprintf('Read %d images from positive #%d \n', length(posImageFiles), i);
end

%%
%Read all negative examples
fprintf('Reading all Negative Examples... \n')
for i = 1:length(folder_neg)
    negImageFiles = dir([folder_neg{i} '\' '*.jpg']); 
    for k = 1:length(negImageFiles)
        filename = negImageFiles(k).name;
        img = imread([folder_neg{i} '\' filename]);
        if wantHoG
            H = [H, HoG(img)];
        end
        if (size(img,3) == 3 && wantGray)
            img = rgb2gray(img);
        end
        IMG(:,:,:,ind) = img; ind = ind + 1;
        X = [X, double(img(:))];
    end
    Y = [Y, zeros(1,length(negImageFiles))];
    n_total = n_total + length(negImageFiles);
    fprintf('Read %d images from negative #%d \n', length(negImageFiles), i);
end

%%
%Shuffle training data
s = RandStream('mt19937ar','Seed',0);
randInd = randperm(s,n_total);

X = X(:,randInd);
Y = Y(:,randInd);
IMG = IMG(:,:,:,randInd);
IMG_labels = Y;
if wantHoG
    H = H(:,randInd);
end
    
%% Normalize the data
X = X./max(max(X));

%Whiten the data
% immean = mean(X,2);
% X_norm = X - immean*ones(1,n_total);
% %X = X - immean*ones(1,n_total);
% 
% cov = 1/n_total * X_norm*X_norm';
% 
% X = cov^(-1/2)*X_norm;

%% Generate Train and Test Data 
n_train = round(n_total*2/3);

X_train = X(:,1:n_train);
X_test  = X(:, n_train+1:end);
    
if wantHoG
	H_train = H(:,1:n_train);
    H_test  = H(:, n_train+1:end);
else
    H_train = [];
    H_test  = [];
end

Y_train = Y(1:n_train);
Y_test = Y(n_train+1:end);

end
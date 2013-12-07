function [X_train Y_train X_test Y_test IMG IMG_labels] = loadHandDataAllClass(folder_front, folder_right, folder_left, folder_neg)
%folder_pos and folder_neg are folder names which contain all the image
%files. The images should all be of the same dimension. folder_neg can be
%an array of folder names having negative images

X = []; %Feature Vectors
Y = []; %Output Labels
IMG = [];
ind = 1;

n_total = 0;

%% Read all front examples Label = 1
fprintf('Reading all Front Examples... \n')
for i = 1:length(folder_front)
    frontImageFiles = dir([folder_front{i} '\' '*.jpg']); 
    for k = 1:length(frontImageFiles)
        filename = frontImageFiles(k).name;
        img = imread([folder_front{i} '\' filename]);
        if (size(img,3) == 3)
            img = rgb2gray(img);
        end
        IMG(:,:,:,ind) = img; ind = ind + 1;
        X = [X, double(img(:))];
    end
    Y = [Y, ones(1,length(frontImageFiles))];
    n_total = n_total + length(frontImageFiles);
    fprintf('Read %d images from front #%d \n', length(frontImageFiles), i);
end

%%
%Read all negative examples Label = 4
fprintf('Reading all Negative Examples... \n')
for i = 1:length(folder_neg)
    negImageFiles = dir([folder_neg{i} '\' '*.jpg']); 
    for k = 1:length(negImageFiles)
        filename = negImageFiles(k).name;
        img = imread([folder_neg{i} '\' filename]);
        if size(img,3)==3
            img = rgb2gray(img);
        end
        IMG(:,:,:,ind) = img; ind = ind + 1;
        X = [X, double(img(:))];
    end
    Y = [Y, 4.*ones(1,length(negImageFiles))];
    n_total = n_total + length(negImageFiles);
    fprintf('Read %d images from negative #%d \n', length(negImageFiles), i);
end

%%
%Read all right examples Label = 2
fprintf('Reading all Right Examples... \n')
for i = 1:length(folder_right)
    rightImageFiles = dir([folder_right{i} '\' '*.jpg']); 
    for k = 1:length(rightImageFiles)
        filename = rightImageFiles(k).name;
        img = imread([folder_right{i} '\' filename]);
        if size(img,3)==3
            img = rgb2gray(img);
        end
        IMG(:,:,:,ind) = img; ind = ind + 1;
        X = [X, double(img(:))];
    end
    Y = [Y, 2.*ones(1,length(rightImageFiles))];
    n_total = n_total + length(rightImageFiles);
    fprintf('Read %d images from right #%d \n', length(rightImageFiles), i);
end
%%
%Read all left examples Label = 3
fprintf('Reading all left Examples... \n')
for i = 1:length(folder_left)
    leftImageFiles = dir([folder_left{i} '\' '*.jpg']); 
    for k = 1:length(leftImageFiles)
        filename = leftImageFiles(k).name;
        img = imread([folder_left{i} '\' filename]);
        if size(img,3)==3
            img = rgb2gray(img);
        end
        IMG(:,:,:,ind) = img; ind = ind + 1;
        X = [X, double(img(:))];
    end
    Y = [Y, 3.*ones(1,length(leftImageFiles))];
    n_total = n_total + length(leftImageFiles);
    fprintf('Read %d images from left #%d \n', length(leftImageFiles), i);
end
%%
%Shuffle training data
s = RandStream('mt19937ar','Seed',0);
randInd = randperm(s,n_total);

X = X(:,randInd);
Y = Y(:,randInd);
IMG = IMG(:,:,:,randInd);
IMG_labels = Y;
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
Y_train = Y(1:n_train);

X_test = X(:, n_train+1:end);
Y_test = Y(n_train+1:end);


end


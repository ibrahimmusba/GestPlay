function [X_train Y_train X_test Y_test] = loadHandData(folder_pos, folder_neg)
%folder_pos and folder_neg are folder names which contain all the image
%files. The images should all be of the same dimension. folder_neg can be
%an array of folder names having negative images

X = []; %Feature Vectors
Y = []; %Output Labels

n_total = 0;

%Read all positive examples
posImageFiles = dir([folder_pos '\' '*.jpg']); 
for k = 1:length(posImageFiles)
    filename = posImageFiles(k).name;
    img = imread([folder_pos '\' filename]);
    img = rgb2gray(img);
    X = [X, double(img(:))];
end
Y = [Y, ones(1,length(posImageFiles))];
n_total = n_total + length(posImageFiles);

%Read all negative examples
for i = 1:length(folder_neg)
    negImageFiles = dir([folder_neg{i} '\' '*.jpg']); 
    for k = 1:length(negImageFiles)
        filename = negImageFiles(k).name;
        img = imread([folder_neg{i} '\' filename]);
        if (size(img,3) == 3)
            img = rgb2gray(img);
        end
        X = [X, double(img(:))];
    end
    Y = [Y, zeros(1,length(negImageFiles))];
    n_total = n_total + length(negImageFiles);
end


%Shuffle training data
randInd = randperm(n_total);
X = X(:,randInd);
Y = Y(:,randInd);

%normalize the data
X = X./max(max(X));

%Whiten the data
immean = mean(X,2);
X_norm = X - immean*ones(1,n_total);
%X = X - immean*ones(1,n_total);

cov = 1/n_total * X_norm*X_norm';

X = cov^(-1/2)*X_norm;



n_train = round(n_total*2/3);

X_train = X(:,1:n_train);
Y_train = Y(1:n_train);

X_test = X(:, n_train+1:end);
Y_test = Y(n_train+1:end);


end
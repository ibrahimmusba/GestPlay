function HandDetector(folder_pos, folder_neg)

%folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\croppedResized';  % write the appropriate folder name here
% croppedHandDimension = [64 48];

posImageFiles = dir([folder_pos '\' '*.jpg']); 
negImageFiles = dir([folder_neg '\' '*.jpg']); 

H = []; %Feature Vectors
Y = []; %Output Labels

n_total = length(posImageFiles) + length(negImageFiles);

%Read all positive examples
for k = 1:length(posImageFiles)
    filename = posImageFiles(k).name;
    img = imread([folder_pos '\' filename]);
    H = [H, HoG(img)];
end
Y = [Y, ones(1,length(posImageFiles))];

%Read all negative examples
for k = 1:length(negImageFiles)
    filename = negImageFiles(k).name;
    img = imread([folder_neg '\' filename]);
    H = [H, HoG(img)];
end
Y = [Y, -1.*ones(1,length(negImageFiles))];


%Shuffle training data
randInd = randperm(n_total);

H = H(randInd);
Y = Y(randInd);


n_train = round(n_total*2/3);

H_train = H(:,1:n_train);
Y_train = Y(1:n_train);

H_test = H(:, n_train+1:end);
Y_test = Y(n_train+1:end);

svm(H_train, Y_train, H_test, Y_test)

%Train a SVM on the data


end

    
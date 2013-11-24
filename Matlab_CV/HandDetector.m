function HandDetector(folder_pos, folder_neg)

%folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\croppedResized';  % write the appropriate folder name here
% croppedHandDimension = [64 48];




H = []; %Feature Vectors
X = []; %Image Vector
Y = []; %Output Labels


n_total = 0;

%Read all positive examples
posImageFiles = dir([folder_pos '\' '*.jpg']); 
for k = 1:length(posImageFiles)
    filename = posImageFiles(k).name;
    img = imread([folder_pos '\' filename]);
    H = [H, HoG(img)];
    X = [X, img(:)];
end
Y = [Y, ones(1,length(posImageFiles))];
n_total = n_total + length(posImageFiles);

%Read all negative examples
for i = 1:length(folder_neg)
    negImageFiles = dir([folder_neg{i} '\' '*.jpg']); 
    for k = 1:length(negImageFiles)
        filename = negImageFiles(k).name;
        img = imread([folder_neg{i} '\' filename]);
        H = [H, HoG(img)];
        X = [X, img(:)];
    end
    Y = [Y, -1.*ones(1,length(negImageFiles))];
    n_total = n_total + length(negImageFiles);
end


%Shuffle training data
randInd = randperm(n_total);

H = H(:,randInd);
X = X(:,randInd);
Y = Y(:,randInd);


n_train = round(n_total*2/3);

H_train = H(:,1:n_train);
Y_train = Y(1:n_train);

H_test = H(:, n_train+1:end);
Y_test = Y(n_train+1:end);
X_test = X(:, n_train+1:end);

%Train a SVM on the data
Y_predict_test = svm(H_train, Y_train, H_test, Y_test);

misclassified = find(Y_predict_test ~= Y_test);

figure;
hold on;
for i = 1:length(misclassified)
    subplot(5,5,i);
    imshow(reshape(X_test(:,i),size(img)));
    if Y_test(i) == 1
        title('Front');
    else
        title('Negative');
    end
end

end
    
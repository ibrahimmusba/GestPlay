function [svmStruct] = HandHogTrainer(handDataSetFolder)

%folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\croppedResized';  % write the appropriate folder name here
% croppedHandDimension = [64 48];

%% Setup positive and negative folders
croppedSize = 'Cropped_144_112';
%croppedSize = 'Cropped_112_88';
%croppedSize = 'Cropped_88_64';
%croppedSize = 'Cropped_64_48';


%Positive
folder_pos{1} = [handDataSetFolder  '\Processed\front\'   croppedSize];
folder_pos{2} = [handDataSetFolder  '\Processed\front\'   'Cropped_scaled_144_112'];

%Random Patches
folder_neg{1} = [handDataSetFolder  '\Processed\random\randomPatches\'  croppedSize];
folder_neg{2} = [handDataSetFolder  '\Processed\random\randomPatches2\' croppedSize];
folder_neg{3} = [handDataSetFolder  '\Processed\negative\'              croppedSize];

%Other Hands
folder_neg{4} = [handDataSetFolder  '\Processed\left_front\' croppedSize];
folder_neg{5} = [handDataSetFolder  '\Processed\right_front\' croppedSize];
folder_neg{6} = [handDataSetFolder  '\Processed\right_back\' croppedSize];
%folder_neg{6} = [handDataSetFolder  '\Processed\front\scaled1'];

H = []; %Feature Vectors
X = []; %Image Vector
Y = []; %Output Labels


n_total = 0;

%% Read all positive examples
fprintf('Reading all Positive Examples and calculating HoG... \n')
for i = 1:length(folder_pos)
    posImageFiles = dir([folder_pos{i} '\' '*.jpg']); 
    for k = 1:length(posImageFiles)
        filename = posImageFiles(k).name;
        img = imread([folder_pos{i} '\' filename]);
        H = [H, HoG(img)];
        if (size(img,3) == 3)
            img = rgb2gray(img);
        end
        X = [X, img(:)];
    end
    Y = [Y, ones(1,length(posImageFiles))];
    n_total = n_total + length(posImageFiles);
    fprintf('Read %d images from positive #%d \n', length(posImageFiles), i);
end

%% Read all negative examples
fprintf('Reading all Negative Examples and calculating HoG... \n')
for i = 1:length(folder_neg)
    negImageFiles = dir([folder_neg{i} '\' '*.jpg']); 
    for k = 1:length(negImageFiles)
        filename = negImageFiles(k).name;
        img = imread([folder_neg{i} '\' filename]);        
        H = [H, HoG(img)];
        if (size(img,3) == 3)
            img = rgb2gray(img);
        end
        X = [X, img(:)];
    end
    Y = [Y, -1.*ones(1,length(negImageFiles))];
    n_total = n_total + length(negImageFiles);
    fprintf('Read %d images from negative #%d \n', length(negImageFiles), i);
end


%% Shuffle training data
s = RandStream('mt19937ar','Seed',0);
randInd = randperm(s,n_total);

H = H(:,randInd);
X = X(:,randInd);
Y = Y(:,randInd);


n_train = round(n_total*2/3);

H_train = H(:,1:n_train);
Y_train = Y(1:n_train);

H_test = H(:, n_train+1:end);
Y_test = Y(n_train+1:end);
X_test = X(:, n_train+1:end);

%% Train a SVM on the data
fprintf('Start SVM training... \n')
[Y_predict_test, classifySVM, kernel, w, b, X_support] = svm(H_train, Y_train, H_test, Y_test);

svmStruct.classifySVM = classifySVM;
svmStruct.w = w;
svmStruct.b = b;
svmStruct.kernel = kernel;
svmStruct.X_support = X_support;

%% Plot misclassified images

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
    
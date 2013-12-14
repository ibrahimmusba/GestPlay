function [svmStruct] = HandHogTrainer(handDataSetFolder)

%folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\croppedResized';  % write the appropriate folder name here
% croppedHandDimension = [64 48];

%% Load all data
addpath ../Matlab_common

croppedSize = 'Cropped_144_112';
%croppedSize = 'Cropped_112_88';
%croppedSize = 'Cropped_88_64';
%croppedSize = 'Cropped_64_48';

isCentered = 0; % do you want centered or uncentered
gestureName = 'front';

wantGray = 1 ; % set it 1 if you want gray images 


[ folder_pos folder_neg ] = getBinaryClassFolderNames( handDataSetFolder, ...
                                            gestureName, croppedSize );

[X_train Y_train X_test Y_test IMAGES IMAGES_labels H_train H_test] = loadHandDataBinaryClass(folder_pos, folder_neg,wantGray);

Y_train(find(Y_train == 0)) = -1;
Y_test(find(Y_test == 0)) = -1;


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
    imshow(reshape(X_test(:,misclassified(i)),size(img)));
    if Y_test(misclassified(i)) == 1
        title('Front');
    else
        title('Negative');
    end
    if i ==25 
        break;
    end
end

end
    
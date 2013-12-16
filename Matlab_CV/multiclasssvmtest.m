addpath ../Matlab_common

handDataSetFolder = 'D:\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset';

%croppedSize = 'Cropped_144_112';
%croppedSize = 'Cropped_112_88';
%croppedSize = 'Cropped_88_64';
croppedSize = 'Cropped_64_48';

wantGray = 1 ; % set it 1 if you want gray images 
wantHoG = 1;

% isCentered = 0; % do you want centered or uncentered
gestureName = cell(1,4);
gestureName{1} = 'front';
gestureName{2} = 'right';
gestureName{3} =  'left'; 
gestureName{4} = 'negative';

X_test = [];
Y_test = [];

for i = 1:4
    [ folder_pos folder_neg ] = getBinaryClassFolderNames( handDataSetFolder, ...
                                            gestureName{i}, croppedSize )

    [X_train Y_train X_test_tmp Y_test_tmp IMAGES IMAGES_labels H_train H_test] = loadHandDataBinaryClass(folder_pos, folder_neg,wantGray, wantHoG);

    %Y_train_tmp(find(Y_train_tmp == 1)) = i;
    posind = find(Y_test_tmp == 1);
   
    X_test = [X_test H_test(:,posind)];
    Y_test = [Y_test ones(1,length(posind))*i];
end


svm_val(1,:) = svmfront.classifySVM(X_test, svmfront.w, svmfront.b, svmfront.X_support);
svm_val(2,:) = svmright.classifySVM(X_test, svmright.w, svmright.b, svmright.X_support);
svm_val(3,:) = svmleft.classifySVM(X_test, svmleft.w, svmleft.b, svmleft.X_support);
svm_val(4,:) = svmneg.classifySVM(X_test, svmneg.w, svmneg.b, svmneg.X_support);

[mval Y_test_pred] = max(svm_val);

test_error = sum(Y_test ~= Y_test_pred)/length(Y_test_pred) *100;
fprintf('Test Error = %f %\n', test_error)

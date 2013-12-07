function [ patches ] = imagePatchGenerator( handDataSetFolder, numpatches )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Setup folders to take positive and negative examples from
%croppedSize = 'Cropped_144_112';
%croppedSize = 'Cropped_112_88';
%croppedSize = 'Cropped_88_64';
croppedSize = 'Cropped_64_48';

gestureName = 'front';
[ folder_pos folder_neg ] = getFolderNames( handDataSetFolder, ...
                                            gestureName, croppedSize )

[X_train Y_train X_test Y_test IMAGES IMAGES_labels] = loadHandDataBinaryClass(folder_pos, folder_neg);

%% Generate Random Patches

patchsize = [8 8];
%numpatches = 1000;

[height width channels] = size(IMAGES(:,:,:,1));


patches = zeros(patchsize(1)*patchsize(2)*channels, numpatches);

image_rand = randi([1 size(IMAGES,4)],1,numpatches);
patch_rand = zeros(2,numpatches);
patch_rand(1,:) = randi([1 height-patchsize(1)+1],1,numpatches);
patch_rand(2,:) = randi([1 width-patchsize(2)+1],1,numpatches);

for i = 1:numpatches
    rows = patch_rand(1,i):patch_rand(1,i)+patchsize(1)-1;   
    cols = patch_rand(2,i):patch_rand(2,i)+patchsize(2)-1;
    
    patch = IMAGES(rows, cols, :, image_rand(i));
    patches(:,i) = patch(:);%reshape(patch,1,patchsize(1)*patchsize(2));
    
%     imwrite(patch,[destinationfolder '\' num2str(i) '.jpg'],'jpg');
    
end


end


%% the following code implements the convolutional neural networks
% and classify the image
clear all
% close all
addpath ../utilities
addpath ../minFunc
addpath ../matFiles
handDatasetFolder = 'C:\Users\Apexit\Dropbox\courses\fall-13\fall-13\442\project\Papers_Project_442_545\Hand Database\dataset\matFiles\';
% handDatasetFolder = '\engin-labs.m.storage.umich.edu\apexit\windat.v2\Desktop\fall-13\545\Project\matFiles\';
%%======================================================================
%% Initialization
%  Here we initialize some parameters

imageHeight = 64;         % image dimension
imageWidth  = 48;
imageChannels = 3;     % number of channels (rgb, so 3)

numClasses = 4;        % select the number of classes
patchDim = 8;          % patch dimension
numPatches = 50000;    % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;           % number of hidden units 

epsilon = 0.1;	       % epsilon for ZCA whitening

poolDim = 16;          % dimension of pooling region

% put feature .mat file here
Features = 'patch8x8fromFull_on_full_image\MS4_FeaturesColor_full_image_patches_from_full_image.mat' ; %% this is colored features for 8x8

numTrainImages = 1600; % the number of train image

if (numClasses ==2)
    dataSet = 'imageSets\MS4_images_color_64_48_frontlabelled.mat' ;
else
    dataSet = 'imageSets\MS4_images_color_64_48.mat'; 
end
%%======================================================================
%% Train a sparse autoencoder (with a linear decoder) to learn 
%  features from color patches. 

    load([handDatasetFolder Features]); 
% --------------------------------------------------------------------

% Display and check to see that the features look good
W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

if (imageChannels ==1 )
    display_network((W*ZCAWhite)');
else
    displayColorNetwork( (W*ZCAWhite)');
end
%%======================================================================
%%  Convolve and pool with the dataset
%  In this step, we will convolve each of the features we learned with
%  the full large images to obtain the convolved features. We will then
%  pool the convolved features to obtain the pooled features for
%  classification.
%
%  Because the convolved features matrix is very large, we will do the
%  convolution and pooling 40 features at a time to avoid running out of
%  memory. 

stepSize = 40;
assert(mod(hiddenSize, stepSize) == 0, 'stepSize should divide hiddenSize');


     load([handDatasetFolder dataSet]); 
     trainImages   = IMAGES(:,:,:,1:numTrainImages);
    trainLabels   = IMAGES_labels(1:numTrainImages)' ; % make it a column vector
    numTestImages = size(IMAGES,4) - numTrainImages ;
    testImages    = IMAGES(:,:,:,(numTrainImages+1): end);
    testLabels    =  IMAGES_labels((numTrainImages+1): end);

    

pooledFeaturesTrain = zeros(hiddenSize, numTrainImages, ...
    floor((imageHeight - patchDim + 1) / poolDim), ...
    floor((imageWidth - patchDim + 1) / poolDim) );
pooledFeaturesTest = zeros(hiddenSize, numTestImages, ...
    floor((imageHeight - patchDim + 1) / poolDim), ...
    floor((imageWidth - patchDim + 1) / poolDim) );

tic();

for convPart = 1:(hiddenSize / stepSize)
    
    featureStart = (convPart - 1) * stepSize + 1;
    featureEnd = convPart * stepSize;
    
    fprintf('Step %d: features %d to %d\n', convPart, featureStart, featureEnd);  
    Wt = W(featureStart:featureEnd, :);
    bt = b(featureStart:featureEnd);    
    
    fprintf('Convolving and pooling train images\n');
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, ...
    trainImages, Wt, bt, ZCAWhite, meanPatch);
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTrain(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
    toc();
    clear convolvedFeaturesThis pooledFeaturesThis;
    
    fprintf('Convolving and pooling test images\n');
    convolvedFeaturesThis = cnnConvolve(patchDim, stepSize, ...
        testImages, Wt, bt, ZCAWhite, meanPatch);
    pooledFeaturesThis = cnnPool(poolDim, convolvedFeaturesThis);
    pooledFeaturesTest(featureStart:featureEnd, :, :, :) = pooledFeaturesThis;   
    toc();

    clear convolvedFeaturesThis pooledFeaturesThis;

end


% You might want to save the pooled features since convolution and pooling takes a long time
save([handDatasetFolder 'cnnPooledMS2Features_B5_L1.mat'], 'pooledFeaturesTrain', 'pooledFeaturesTest');
toc();

%%======================================================================
%% Use pooled features for classification
% use the softmax classifier

% Setup parameters for softmax
 load ([handDatasetFolder 'patch16x16fromHand_on_Centered_Hand\cnnPooledMS4FeaturesColor16x16_pool16_AllClasses.mat'])
softmaxLambda = 5e-4;

load([handDatasetFolder dataSet]); 
     trainImages   = IMAGES(:,:,:,1:numTrainImages);
    trainLabels   = IMAGES_labels(1:numTrainImages)' ; % make it a column vector
    numTestImages = size(IMAGES,4) - numTrainImages ;
    testImages    = IMAGES(:,:,:,(numTrainImages+1): end);
    testLabels    =  IMAGES_labels((numTrainImages+1): end);

% Reshape the pooledFeatures to form an input vector for softmax
softmaxX = permute(pooledFeaturesTrain, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledFeaturesTrain) / numTrainImages,...
    numTrainImages);
if (numClasses ==2)
    softmaxY = trainLabels+1;
else
    softmaxY = trainLabels;
end

options = struct;
options.maxIter = 200;
softmaxModel = softmaxTrain(numel(pooledFeaturesTrain) / numTrainImages,...
    numClasses, softmaxLambda, softmaxX, softmaxY, options);

%%======================================================================
%% STEP 5: Test classifer
%  Now you will test your trained classifer against the test images

softmaxX = permute(pooledFeaturesTest, [1 3 4 2]);
softmaxX = reshape(softmaxX, numel(pooledFeaturesTest) / numTestImages, numTestImages);
if (numClasses ==2)
    softmaxY = testLabels+1;
else
    softmaxY = testLabels;
end

[pred] = softmaxPredict(softmaxModel, softmaxX);
acc = (pred(:) == softmaxY(:));
acc = sum(acc) / size(acc, 1);
fprintf('Accuracy: %2.3f%%\n', acc * 100);

%% show missclassified images

misclassified = find(~acc);
figure;
hold on;
for i = 1:length(misclassified)
    subplot(5,5,i);
    imshow(reshape(testImages(:,misclassified(i)),[imageHeightm imageWidth,imageChannels]))
    if testLabels(i) == 1
        title('Front');
    else
        title('Negative');
    end
    if i ==25 
        break;
    end
end

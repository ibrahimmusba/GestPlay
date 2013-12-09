%% CS294A/CS294W Stacked Autoencoder Exercise

%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  sstacked autoencoder exercise. You will need to complete code in
%  stackedAECost.m
%  You will also need to have implemented sparseAutoencoderCost.m and 
%  softmaxCost.m from previous exercises. You will need the initializeParameters.m
%  loadMNISTImages.m, and loadMNISTLabels.m files from previous exercises.
%  
%  For the purpose of completing the assignment, you do not need to
%  change the code in this file. 
%
addpath ../utilities/
mnistData = false;
if(~mnistData)
    load('C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset\matFiles\patch16x16fromHand_on_Centered_Hand\cnnPooledMS4FeaturesColor16x16_pool16_AllClasses.mat')
    load('C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset\matFiles\imageSets\MS4_images_color_64_48.mat')
end
      
%%======================================================================
%% STEP 0: Here we provide the relevant parameters values that will
%  allow your sparse autoencoder to get good filters; you do not need to 
%  change the parameters below.

if(~mnistData)
    inputSize = prod(size(pooledFeaturesTrain))/size(pooledFeaturesTrain,2);
    numClasses = 4;
    hiddenSizeL1 = 500;    % Layer 1 Hidden Size
    hiddenSizeL2 = 200;    % Layer 2 Hidden Size
    sparsityParam = 0.1;   % desired average activation of the hidden units.
                           % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
    		               %  in the lecture notes). 
    lambda = 3e-3;         % weight decay parameter       
    beta = 3;              % weight of sparsity penalty term       
else
    inputSize = 28 * 28;
    numClasses = 10;
    hiddenSizeL1 = 200;    % Layer 1 Hidden Size
    hiddenSizeL2 = 200;    % Layer 2 Hidden Size
    sparsityParam = 0.1;   % desired average activation of the hidden units.
                           % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
    		               %  in the lecture notes). 
    lambda = 3e-3;         % weight decay parameter       
    beta = 3;              % weight of sparsity penalty term       
end

%%======================================================================
%% STEP 1: Load data from the MNIST database
%
%  This loads our training data from the MNIST database files.

fprintf('Loading Data');

% Load MNIST database files
if(mnistData)
    addpath ../mnist/
    trainData = loadMNISTImages('train-images.idx3-ubyte');
    trainLabels = loadMNISTLabels('train-labels.idx1-ubyte');

    trainLabels(trainLabels == 0) = 10; % Remap 0 to 10 since our labels need to start from 1
else
    % Load pooled data
    totalDataSize = length(IMAGES_labels);
    trainDataSize = size(pooledFeaturesTrain,2);
    testDataSize = totalDataSize - trainDataSize;
    
    trainData = zeros(inputSize,trainDataSize);
    trainLabels = IMAGES_labels(1:trainDataSize);
    testData = zeros(inputSize,testDataSize);
    testLabels = IMAGES_labels(trainDataSize+1:end);
    for i = 1:trainDataSize
        trainData(:,i) = reshape(pooledFeaturesTrain(:,i,:,:),inputSize,1);
    end
    for i = 1:testDataSize
        testData(:,i) = reshape(pooledFeaturesTest(:,i,:,:),inputSize,1);
    end
    addpath ../Matlab_CV    
end

%%======================================================================
%% STEP 2: Train the first sparse autoencoder
%  This trains the first sparse autoencoder on the unlabelled STL training
%  images.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

fprintf('\nPreTraining First Layer\n');

%  Randomly initialize the parameters
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the first layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL1"
%                You should store the optimal parameters in sae1OptTheta


%  Use minFunc to minimize the function
addpath ../minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[sae1OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   inputSize, hiddenSizeL1, ...
                                   lambda, sparsityParam, ...
                                   beta, trainData), ...
                              sae1Theta, options);
                          
% -------------------------------------------------------------------------



%%======================================================================
%% STEP 2: Train the second sparse autoencoder
%  This trains the second sparse autoencoder on the first autoencoder
%  featurse.
%  If you've correctly implemented sparseAutoencoderCost.m, you don't need
%  to change anything here.

fprintf('\nPreTraining Second Layer\n');

[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
                                        inputSize, trainData);

%  Randomly initialize the parameters
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the second layer sparse autoencoder, this layer has
%                an hidden size of "hiddenSizeL2" and an inputsize of
%                "hiddenSizeL1"
%
%                You should store the optimal parameters in sae2OptTheta

[sae2OptTheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   hiddenSizeL1, hiddenSizeL2, ...
                                   lambda, sparsityParam, ...
                                   beta, sae1Features), ...
                              sae2Theta, options);

% -------------------------------------------------------------------------


%%======================================================================
%% STEP 3: Train the softmax classifier
%  This trains the sparse autoencoder on the second autoencoder features.
%  If you've correctly implemented softmaxCost.m, you don't need
%  to change anything here.

fprintf('\nTraining Softmax Classifier\n');

[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
                                        hiddenSizeL1, sae1Features);

%  Randomly initialize the parameters
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL2 * numClasses, 1);


%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the softmax classifier, the classifier takes in
%                input of dimension "hiddenSizeL2" corresponding to the
%                hidden layer size of the 2nd layer.
%
%                You should store the optimal parameters in saeSoftmaxOptTheta 
%
%  NOTE: If you used softmaxTrain to complete this part of the exercise,
%        set saeSoftmaxOptTheta = softmaxModel.optTheta(:);


options.maxIter = 100;
softmaxModel = softmaxTrain(hiddenSizeL2, numClasses, lambda, ...
                            sae2Features, trainLabels, options);


saeSoftmaxOptTheta = softmaxModel.optTheta(:);

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 5: Finetune softmax model

% Implement the stackedAECost to give the combined cost of the whole model
% then run this cell.

fprintf('\nFINE TUNING\n');

% Initialize the stack using the parameters learned
stack = cell(2,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
                     hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
                     hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);

% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

%% ---------------------- YOUR CODE HERE  ---------------------------------
%  Instructions: Train the deep network, hidden size here refers to the '
%                dimension of the input to the classifier, which corresponds 
%                to "hiddenSizeL2".
%
%
        
% -save beforeFineTuning

%  Use minFunc to minimize the function
addpath ../minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';
%options.maxFunEvals = 2500;

lambda = 0.0002;
[stackedAEOptTheta, cost] = minFunc( @(p) stackedAECost(p, inputSize, hiddenSizeL2, ...
                                              numClasses, netconfig, ...
                                              lambda, trainData, trainLabels), ...
                              stackedAETheta, options);

% -------------------------------------------------------------------------



%%======================================================================
%% STEP 6: Test 
%  Instructions: You will need to complete the code in stackedAEPredict.m
%                before running this part of the code
%

% Get labelled test images
% Note that we apply the same kind of preprocessing as the training set
if(mnistData)
    testData = loadMNISTImages('t10k-images.idx3-ubyte');
    testLabels = loadMNISTLabels('t10k-labels.idx1-ubyte');

    testLabels(testLabels == 0) = 10; % Remap 0 to 10
end

[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

accBefore = mean(testLabels(:) == pred(:));
fprintf('Before Finetuning Test Accuracy: %0.3f%%\n', accBefore * 100);

[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL2, ...
                          numClasses, netconfig, testData);

accAfter = mean(testLabels(:) == pred(:));
fprintf('After Finetuning Test Accuracy: %0.3f%%\n', accAfter * 100);

% Accuracy is the proportion of correctly classified images
% The results for our implementation were:
%
% Before Finetuning Test Accuracy: 87.7%
% After Finetuning Test Accuracy:  97.6%2
%
% If your values are too low (accuracy less than 95%), you should check 
% your code for errors, and make sure you are training on the 
% entire data set of 60000 28x28 training images 
% (unless you modified the loading code, this should be the case)
% save afterFineTuning
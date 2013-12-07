addpath ../utilities
addpath ../minFunc
addpath ../matFiles
handDatasetFolder = '';
% handDatasetFolder = '\engin-labs.m.storage.umich.edu\apexit\windat.v2\Desktop\fall-13\545\Project\matFiles\';
%%  Initialization
%  Here we initialize some parameters used for the exercise.

imageChannels = 1;     % number of channels (rgb, so 3)

patchDim   = 8;          % patch dimension
numPatches = 100000;   % number of patches

visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize  = visibleSize;   % number of output units
hiddenSize  = 400;           % number of hidden units 

sparsityParam = 0.035; % desired average activation of the hidden units.
lambda = 1e-3;         % weight decay parameter       
beta = 5;              % weight of sparsity penalty term       

epsilon = 0.1;	       % epsilon for ZCA whitening

%%======================================================================
%% sparse Auto encoder (linear)
debugHiddenSize = 5;
debugvisibleSize = 8;
patches = rand([8 10]);
theta = initializeParameters(debugHiddenSize, debugvisibleSize); 

[cost, grad] = sparseAutoencoderLinearCost(theta, debugvisibleSize, debugHiddenSize, ...
                                           lambda, sparsityParam, beta, ...
                                           patches);

% Check gradients
numGrad = computeNumericalGradient( @(x) sparseAutoencoderLinearCost(x, debugvisibleSize, debugHiddenSize, ...
                                                  lambda, sparsityParam, beta, ...
                                                  patches), theta);

% Use this to visually compare the gradients side by side
disp([numGrad grad]); 

diff = norm(numGrad-grad)/norm(numGrad+grad);
% Should be small. In our implementation, these values are usually less than 1e-9.
disp(diff); 

assert(diff < 1e-9, 'Difference too large. Check your gradient computation again');



%%======================================================================
%% Learn features on small patches
%  In this step, we will use your sparse autoencoder (which now uses a 
%  linear decoder) to learn features on small patches sampled from related
%  images.

%% Load patches
%  In this step, we load 100k patches sampled from the hand dataset and
%  visualize them. Note that these patches have been scaled to [0,1]

% load([handDatasetFolder 'stlSampledPatches.mat']);


%TODO
load([handDatasetFolder 'MS4_patches_gray_100k.mat']);

if (imageChannels ==1 )
    display_network(patches(:,1:100));
else
    displayColorNetwork(patches(:, 1:100));
end


%% Apply preprocessing
%  we preprocess the sampled patches, in particular, ZCA whitening them. 
% 
% Subtract mean patch (hence zeroing the mean of the patches)
meanPatch = mean(patches, 2);  
patches = bsxfun(@minus, patches, meanPatch);

% Apply ZCA whitening
sigma = patches * patches' / numPatches;
[u, s, v] = svd(sigma);
ZCAWhite = u * diag(1 ./ sqrt(diag(s) + epsilon)) * u';
patches = ZCAWhite * patches;
figure
if (imageChannels ==1 )
    display_network(patches(:,1:100));
else
    displayColorNetwork(patches(:, 1:100));
end
%% Learn features
%  use sparse autoencoder (with linear decoder) to learn
%  features on the preprocessed patches. 

theta = initializeParameters(hiddenSize, visibleSize);

% Use minFunc to minimize the function
addpath minFunc/

options = struct;
options.Method = 'lbfgs'; 
options.maxIter = 400;
options.display = 'on';

[optTheta, cost] = minFunc( @(p) sparseAutoencoderLinearCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

% Save the learned features and the preprocessing matrices for use in 
% the later exercise on convolution and pooling
fprintf('Saving learned features and preprocessing matrices...\n');                          
save([handDatasetFolder 'MS2Features_B5_L1.mat'], 'optTheta', 'ZCAWhite', 'meanPatch');
fprintf('Saved\n');

%% STEP 2d: Visualize learned features

W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

if (imageChannels ==1 )
    display_network((W*ZCAWhite)');
else
    displayColorNetwork( (W*ZCAWhite)');
end
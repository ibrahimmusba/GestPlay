function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = size(data, 2);
n_l = numel(stack) + 2;
n = numel(stack);
% Compute forward pass
a = cell(1,n_l);
z = cell(1,n_l);

a{1} = data;

for d = 1:numel(stack)
    z{d+1} = stack{d}.w*a{d} + repmat(stack{d}.b,1,M);
    a{d+1} = 1./(1+exp(-z{d+1}));
end
% d = numel(stack);

thetaTimesX = softmaxTheta * a{n+1};
thetaTimesX = bsxfun(@minus, thetaTimesX, max(thetaTimesX, [], 1));
expThetaTimesX = exp(thetaTimesX);


[m pred] = max(expThetaTimesX);





% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

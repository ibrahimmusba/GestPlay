function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% 
%Cost
thetaTimesX = theta * data;
thetaTimesX = bsxfun(@minus, thetaTimesX, max(thetaTimesX, [], 1));
expThetaTimesX = exp(thetaTimesX);

den = ones(numClasses,1)*sum(expThetaTimesX);

cost = -1/numCases * sum(sum(groundTruth.*log( expThetaTimesX./den ))) + (lambda/2*sum(sum(theta.^2)));


%Gradient
thetagrad = (-1/numCases * data*(groundTruth-(expThetaTimesX./den))')'  + (lambda*theta);

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end


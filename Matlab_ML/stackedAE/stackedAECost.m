function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
M = size(data, 2);
groundTruth = full(sparse(labels, 1:M, 1));


%% 
n_l = numel(stack) + 1;
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

den = ones(numClasses,1)*sum(expThetaTimesX);

p = expThetaTimesX./den; %probability or of output(p(y|x)) or hypothesis 

softmaxThetaGrad = (-1/hiddenSize * a{n+1}*(groundTruth-p)')'  + (lambda*softmaxTheta);

delta = cell(1,numel(stack)+1); % Range from 2:n_layers


delta{n_l} = -1*(softmaxTheta'*(groundTruth - p)).*a{n_l}.*(1-a{n_l});
for l = n_l-1:-1:2
    delta{l} = a{l}.*(1-a{l}).*(stack{l}.w'*delta{l+1});%+ beta.*((-rho./rho_hat)+((1-rho)./(1-rho_hat)))*ones(1,m)
end

for l = 1:numel(stack)
    stackgrad{l}.w = ((1./M)*delta{l+1}* a{l}'); + lambda*stack{l}.w;
    stackgrad{l}.b = 1./M*sum(delta{l+1},2);
end

cost = -1/M * sum(sum(groundTruth.*log(p) )) + (lambda/2) * sum(sum(softmaxTheta.^2)); 
% W1grad = ((1./m)*gamma2* a1') + lambda*W1;
% W2grad = ((1./m)*gamma3* a2') + lambda*W2;
% 
% b1grad = 1./m*sum(gamma2,2);
% b2grad = 1./m*sum(gamma3,2);

% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end

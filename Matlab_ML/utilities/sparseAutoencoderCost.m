function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% 

m = size(data,2);

% Compute forward pass
a1 = data;

z2 = W1*a1 + repmat(b1,1,m);
a2 = 1./(1+exp(-z2));

z3 = W2*a2 + repmat(b2,1,m);
a3 = 1./(1+exp(-z3));

% Backpropagate to compute gradients
rho = sparsityParam;
rho_hat = 1./m*(sum(a2,2));

gamma3 = -1*(data - a3).*a3.*(1-a3);
gamma2 = a2.*(1-a2).*(W2'*gamma3 + beta.*((-rho./rho_hat)+((1-rho)./(1-rho_hat)))*ones(1,m));

W1grad = ((1./m)*gamma2* a1') + lambda*W1;
W2grad = ((1./m)*gamma3* a2') + lambda*W2;

b1grad = 1./m*sum(gamma2,2);
b2grad = 1./m*sum(gamma3,2);

%Cost
%cost1 = 1/(2*size(data,2))*trace((data - a3)'*(data - a3));
cost1 = 1/(2*size(data,2))*sum(sum((data - a3).^2));
cost2 = (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
cost3 = beta.*sum(rho.*(log(rho./rho_hat))+(1-rho).*(log((1-rho)./(1-rho_hat))));
cost =  cost1 + cost2 + cost3;


%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end


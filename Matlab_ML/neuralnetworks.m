%BackPropogation Algorithm for training neural networks

%% Load MNIST data
[X_train Y_train X_test Y_test] = loadMNISTData;
d = size(X_train,1);
%% Load Raw Hand Data
%dataset_folder = 'D:\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset'; %Change this based on your computer
dataset_folder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset';
folder_pos = [dataset_folder '\Processed\front\croppedResized'];
folder_neg{1} = [dataset_folder '\Processed\random\randomPatches'];
folder_neg{2} = [dataset_folder '\Processed\random\randomPatches2'];
folder_neg{3} = [dataset_folder '\Processed\negative\croppedResized'];
%folder_neg{3} = 'D:\Dropbox\CLASS STUFF\Project_442_545\Hand Database\Apexit\down\croppedResized';
[X_train Y_train X_test Y_test] = loadHandData(folder_pos, folder_neg);

d = size(X_train,1);

%% Define and Initialise the neural network
n_ip = d;                       %Number of input features
n_op = 1;                       %Number of op layers
n_l = 4;                        %Number of layers including ip and op layer
n_nodes = [n_ip 300 300 n_op];       %Number of nodes per layer

alpha = 10.0;            %Learning rate (or) gradient descent step size
lambda = 0.0001;           %Regularization constant

% Sigmoid and its differencial
sig = @(k) (1./(1+exp(-k)));
d_sig = @(k) (sig(k).*(1-sig(k)));

%initiailse the weights randomly (TODO: find best initialization)
W = cell(1,n_l-1);
b = cell(1,n_l-1);
for i = 1:n_l-1
    W{i} = (rand(n_nodes(i+1),n_nodes(i)) - 0.5 ) / 90;    %small random value 
    b{i} = zeros(n_nodes(i+1),1);
end
 
%% This does not work

iter = 0;
while(1) %Loop till convergence
    %Gradient Descent
    iter = iter+1;
    %Initialise tmp gradients
    Wgrad = cell(1,n_l-1);
    bgrad = cell(1,n_l-1);
    for l = 1:n_l-1
        Wgrad{l} = zeros(size(W{l}));
        bgrad{l} = zeros(size(b{l}));
    end
    
    theta = convertWbToTheta(W,b);
    
    [cost grad] = evaluateNeuralNetwork(theta, X_train, Y_train, n_l, ...
                                                n_nodes, lambda);
    [Wgrad bgrad] = convertThetaToWb(grad, n_nodes);
    %Code to check gradients
%     numgrad = computeNumericalGradients(@(x) evaluateNeuralNetwork(x, ip, op, n_l, n_nodes, lambda), theta);
%     grad = convertWbToTheta(Wgrad,bgrad);
%     
%     disp([numgrad  grad]);
%     disp(norm(numgrad-grad));

    %Update the wieghts 
    for l = 1:n_l-1
        W{l} = W{l} - alpha*(Wgrad{l}); % + lambda*W{l}
        b{l} = b{l} - alpha*(bgrad{l});
    end
    
    %Check the training error in each step and see if it decreases 
    Y_predict = zeros(1,length(Y_test));
    [a1, z1] = doForwardPass(W,b,X_test,n_l);
    Y_predict = (a1{n_l}>0.5);
    test_error = sum(Y_test ~= Y_predict)/length(Y_test)*100

    %Need to implement stopping criterion
    if(iter == 400)
        break;
    end
end

%% Gradient Descent
%  Randomly initialize the parameters
theta = convertWbToTheta(W,b);


%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 70;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';

% 
% [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
%                                    visibleSize, hiddenSize, ...
%                                    lambda, sparsityParam, ...
%                                    beta, patches), ...
%                               theta, options);

[opttheta, cost] = minFunc( @(p) evaluateNeuralNetwork(p, X_train, Y_train, ...
                                            n_l, n_nodes, lambda), ...
                              theta, options)
                          
%%======================================================================

%%
[W_new b_new] = convertThetaToWb(opttheta, n_nodes);

Y_predict_test = zeros(1,length(Y_test));
Y_predict_train = zeros(1,length(Y_train));

[a1, z1] = doForwardPass(W_new,b_new,X_test,n_l);
Y_predict_test = (a1{n_l}>0.5);

[a2, z2] = doForwardPass(W_new,b_new,X_train,n_l);
Y_predict_train = (a2{n_l}>0.5);



test_error  = sum(Y_test ~= Y_predict_test)/length(Y_test)*100;
train_error = sum(Y_train ~= Y_predict_train)/length(Y_train)*100;

sprintf('Number of training samples = %d', length(Y_train))
sprintf('Number of test samples = %d', length(Y_test))

sprintf('Test Error = %f %', test_error)
sprintf('Training Error = %f %', train_error)


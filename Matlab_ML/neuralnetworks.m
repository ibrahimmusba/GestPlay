%BackPropogation Algorithm for training neural networks

addpath RBM

%% Load MNIST data
[X_train Y_train X_test Y_test] = loadMNISTData;
d = size(X_train,1);
%% Load Raw Hand Data
addpath ../Matlab_CV
%dataset_folder = 'D:\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset'; %Change this based on your computer
handDataSetFolder = 'C:\Users\imusba\Dropbox\CLASS STUFF\Project_442_545\Hand Database\dataset';

% croppedSize = 'Cropped_144_112';
%croppedSize = 'Cropped_112_88';
%croppedSize = 'Cropped_88_64';
croppedSize = 'Cropped_64_48';

folder_pos{1} = [handDataSetFolder  '\Processed\front\'   croppedSize];
folder_pos{2} = [handDataSetFolder  '\Processed\front\'   'Cropped_scaled_64_48'];

%Random Patches
folder_neg{1} = [handDataSetFolder  '\Processed\random\randomPatches\'  croppedSize];
folder_neg{2} = [handDataSetFolder  '\Processed\random\randomPatches2\' croppedSize];
folder_neg{3} = [handDataSetFolder  '\Processed\negative\'              croppedSize];

%Other Hands
folder_neg{4} = [handDataSetFolder  '\Processed\left_front\' croppedSize];
folder_neg{5} = [handDataSetFolder  '\Processed\right_front\' croppedSize];
folder_neg{6} = [handDataSetFolder  '\Processed\right_back\' croppedSize];%folder_neg{3} = 'D:\Dropbox\CLASS STUFF\Project_442_545\Hand Database\Apexit\down\croppedResized';

%[X_train Y_train X_test Y_test] = loadHandData(folder_pos, folder_neg);
[X_train Y_train X_test Y_test] = loadHandHoGData(folder_pos, folder_neg);
Y_train = -1.*Y_train;
Y_test = -1.*Y_test;
d = size(X_train,1);

%% Define and Initialise the neural network
n_ip = d;                       %Number of input features
n_op = 1;                       %Number of op layers
n_nodes = [n_ip 300 300 n_op];  %Number of nodes per layer
n_l = length(n_nodes);          %Number of layers including ip and op layer


alpha = 10.0;            %Learning rate (or) gradient descent step size
lambda = 0.0002;           %Regularization constant

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
 
%% RBM 
% initialization 
epochs = 2000;
learning_rate =0.1;
in = X_train';
for i=1:n_l-1
 
    [W{i} b{i} d h] = trainRBM(in, W{i}, epochs, learning_rate);
    in = h;
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
options.maxIter = 180;	  % Maximum number of iterations of L-BFGS to run 

options.display = 'on';

% 
% [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
%                                    visibleSize, hiddenSize, ...
%                                    lambda, sparsityParam, ...
%                                    beta, patches), ...
%                               theta, options);

[opttheta, cost] = minFunc( @(p) evaluateNeuralNetwork(p, X_train, Y_train, ...
                                            n_l, n_nodes, lambda), ...
                              theta, options);
                          
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


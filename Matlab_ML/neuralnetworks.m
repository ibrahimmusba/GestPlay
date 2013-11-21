%BackPropogation Algorithm for training neural networks

%% Train with MNIST Data
%Ready the data
load mnist_49_3000
[d,n] = size(x);

m = 2000;
X_train = x(:,1:2000);
Y_train = y(1:2000);


X_test = x(:,2001:3000);
Y_test = y(2001:3000);

%Inputs and Outputs
ip = X_train;
op = Y_train;

%% Train with sin data
t1(1,:) = 0:0.01:1;
t1(2,:) = 0:0.01:1;
t1(3,:) = 0:0.01:1;
t1(4,:) = 0:0.01:1;

% Train data
x1(1,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) + 0.2);
x1(2,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) - 0.2);
x1(3,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) + 0.6);
x1(4,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) - 0.6);

X_train = [x1(1,:), x1(2,:), x1(3,:), x1(4,:);
            t1(1,:), t1(2,:), t1(3,:), t1(4,:)];
Y_train = [ones(1,length(t1(1,:))*2) zeros(1,length(t1(1,:))*2)];

%Test Data
x1(1,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) + 0.2);
x1(2,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) - 0.2);
x1(3,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) + 0.6);
x1(4,:) = sin(2*pi*t1(1,:)) + (0.1*randn(1,length(t1)) - 0.6);

X_test = [x1(1,:), x1(2,:), x1(3,:), x1(4,:);
            t1(1,:), t1(2,:), t1(3,:), t1(4,:)];
Y_test = [ones(1,length(t1(1,:))*2) zeros(1,length(t1(1,:))*2)];

%Visualize the data
figure(1); hold on;
plot(X_train(2,1:202),X_train(1,1:202),'*b');
plot(X_train(2,203:end),X_train(1,203:end),'*r');

[d m] = size(X_train);

%Normalize the data
X_train = X_train./4 + 0.5;
X_test = X_test./4 + 0.5;

ip = X_train;
op = Y_train;


%%
%Define the neural network
n_ip = d;                       %Number of input features
n_op = 1;                       %Number of op layers
n_l = 3;                        %Number of layers including ip and op layer
n_nodes = [n_ip 30 n_op];       %Number of nodes per layer

alpha = 1.0;            %Learning rate (or) gradient descent step size
lambda = 0.0;           %Regularization constant

% Sigmoid and its differencial
sig = @(k) (1./(1+exp(-k)));
d_sig = @(k) (sig(k).*(1-sig(k)));

%% Initialise the neural network

%initiailse the weights
W = cell(1,n_l-1);
b = cell(1,n_l-1);
for i = 1:n_l-1
    W{i} = (rand(n_nodes(i+1),n_nodes(i)) - 0.5 ) / 50;    %small random value 
    b{i} = (zeros(n_nodes(i+1),1) - 0.5 ) / 50;
end

gamma = cell(1,n_l); % Range from 2:n_l
 


iter = 0;
while(1) %Loop till convergence
    %Gradient Descent
    iter = iter+1;
    %Initialise tmp gradients
    d_W = cell(1,n_l-1);
    d_b = cell(1,n_l-1);
    for l = 1:n_l-1
        d_W{l} = zeros(size(W{l}));
        d_b{l} = zeros(size(b{l}));
    end
    
    [cost Wgrad bgrad] = evaluateNeuralNetwork(W, b, ip, op, n_l);
    
    
    %Update the wieghts 
    for l = 1:n_l-1
        W{l} = W{l} - alpha*(Wgrad{l}); % + lambda*W{l}
        b{l} = b{l} - alpha*(bgrad{l});
    end
    
    %Check the training error in each step and see if it decreases 
    Y_predict = zeros(1,length(Y_test));
    for i = 1:length(Y_test)
        [a1, z1] = doForwardPass(W,b,X_test(:,i),n_l);
        Y_predict(i) = ((sign(a1{n_l} - 0.5)/2)) + 0.5;
    end
    training_error = sum(Y_test ~= Y_predict)/length(Y_test)*100

    %Need to implement stopping criterion
    if(iter == 400)
        break;
    end
end

%% Gradient Descent
%  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);


%  Use minFunc to minimize the function
addpath minFunc/
options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

%%======================================================================

%%

Y_predict = zeros(1,length(Y_test));
for i = 1:404
    [a, z] = doForwardPass(W,b,X_test(:,i),n_l);
    Y_predict(i) = sign(a{n_l});
end

training_error = sum(Y_test ~= Y_predict)/404*100


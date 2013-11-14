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

ip = X_train;
op = Y_train;

%%
%Define the neural network
n_ip = d;                       %Number of input features
n_op = 1;                       %Number of op layers
n_l = 5;                        %Number of layers including ip and op layer
n_nodes = [n_ip 5 5 5 n_op];    %Number of nodes per layer

alpha = 0.1;            %Learning rate (or) gradient descent step size
lambda = 0.0;           %Regularization constant

NN{1} = n_ip;
NN{2} = n_op;
NN{3} = n_l;
NN{4} = n_nodes;

%% Initialise the neural network

%initiailse the weights
W = cell(1,n_l-1);
b = cell(1,n_l-1);
for i = 1:n_l-1
    W{i} = (rand(n_nodes(i+1),n_nodes(i)) - 0.5 ) / 50;    %small random value 
    b{i} = (zeros(n_nodes(i+1),1) - 0.5 ) / 50;
end

% Sigmoid and its differencial
sig = @(k) (1./(1+exp(-k)));
d_sig = @(k) (sig(k).*(1-sig(k)));


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
    
    % Do a forward pass for each data points and backpropogate to find
    % gradients
    for i = 1:m
        [a, z] = forward_pass(W,b,ip(:,i),NN);
        [d_W_tmp d_b_tmp] = backpropagate(W, b, ip(:,i), op(i), NN, a, z, sig, d_sig);
        for l = 1:n_l-1
            d_W{l} = d_W{l} + d_W_tmp{l};
            d_b{l} = d_b{l} + d_b_tmp{l};
        end
    end
    
    %Update the wieghts 
    for l = 1:n_l-1
        W{l} = W{l} - alpha*(d_W{l}./m + lambda*W{l});
        b{l} = b{l} - alpha*(d_b{l}./m);
    end
    
    %Check the training error in each step and see if it decreases 
    Y_predict = zeros(1,length(Y_test));
    for i = 1:length(Y_test)
        [a1, z1] = forward_pass(W,b,X_test(:,i),NN);
        Y_predict(i) = ((sign(a1{n_l} - 0.5)/2)) + 0.5;
    end
    training_error = sum(Y_test ~= Y_predict)/length(Y_test)*100

    %Need to implement stopping criterion
    if(iter == 200)
        break;
    end
end

%%

Y_predict = zeros(1,length(Y_test));
for i = 1:404
    [a, z] = forward_pass(W,b,X_test(:,i),NN);
    Y_predict(i) = sign(a{n_l});
end

training_error = sum(Y_test ~= Y_predict)/404*100


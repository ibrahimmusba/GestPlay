function [X_train Y_train X_test Y_test] = loadMNISTData()

%% Train with MNIST Data
%Ready the data
load mnist_49_3000
[d,n] = size(x);

y = (y+1)./2;
m = 2000;
X_train = x(:,1:2000);
Y_train = y(1:2000);


X_test = x(:,2001:3000);
Y_test = y(2001:3000);

end
function [X_train Y_train X_test Y_test] = loadSinData()

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
Y_train = Y_train;

end
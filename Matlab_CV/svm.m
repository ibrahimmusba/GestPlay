%SVM code from ML HW2, need to modify and combine with HoG code

load mnist_49_3000
[d,n] = size(x);

X_train = x(:,1:2000);
Y_train = y(1:2000);

X_test = x(:,2001:3000);
Y_test = y(2001:3000);


% USAGE: [alpha,bias] = smo(K, y, C, tol)
%  
%   INPUT:
%  
%     K: n x n kernel matrix
%     y: 1 x n vector of labels, -1 or 1
%     C: a regularization parameter such that 0 <= alpha_i <= C/n
%     tol: tolerance for terminating criterion
%  
%   OUTPUT:
%  
%     alpha: 1 x n lagrange multiplier coefficient
%     bias: scalar bias (offset) term

%holdout training
X_train_ho = x(:,1:1000);
Y_train_ho = y(1:1000);

X_test_ho = x(:,1001:2000);
Y_test_ho = y(1001:2000);

p = [1 2];
idx = 2;

%%
%Kernel Matrix
A = X_train_ho;
K = (A'*A + 1).^p(idx);

C = logspace(log10(1),log10(2000),20);
for i = 1:length(C)
    i
    [alpha_ho(i,:), b_ho(i)] = smo(K, Y_train_ho, C(i), 0.001);
end

%%
error_ho = zeros(1,length(C));

for i = 1:length(C)
    K_ho = (X_train_ho' * X_test_ho + 1).^p(idx);
    Y_ans_ho = sign( (alpha_ho(i,:).*Y_train_ho) *K_ho + b_ho(i) );
    
    error_ho(i) = sum(Y_test_ho ~= Y_ans_ho)/1000;
end

[error_ho*100 ; C]

%%
if idx == 1
    C = 279.2102;
else
    C = 2;
end

%Kernel Matrix
A = X_train;
K_train = (A'*A + 1).^p(idx);

[alpha, b] = smo(K_train, Y_train, C, 0.0001);

K_test = (X_train' * X_test + 1).^p(idx);
Y_ans = sign( (alpha.*Y_train) *K_test + b );
%num_support_vec = sum(abs((alpha.*Y_train) *K_test + b) <= 1)
sum(alpha > 0)
error = sum(Y_test ~= Y_ans)/1000

training_error = sum(sign( (alpha.*Y_train) *K_train + b ) ~= Y_train)
%%
%Figure to plot 4x5 image 
clear sort ind
support_vec_bool = (alpha > 0);
%eta = support_vec_bool .* abs(Y_train.*(alpha.*Y_train) *K_train + b);
v = (abs((alpha.*Y_train) *K_train + b));

[sortd, ind] = sort(v,'ascend');%,'descend')

for i = 1:20
    subplot(4,5,i);
    imshow(reshape(X_train(:,ind(i)),[sqrt(d),sqrt(d)])');
    if Y_train(ind(i)) == 0
        title('4');
    else
        title('9');
    end
    %title( (sprintf('A=%d P=%d', label_test, label_predict)));
    %title( (sprintf('%d ', label_test)));
end
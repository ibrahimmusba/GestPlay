function [Y_predict] = svm(X_train, Y_train, X_test, Y_test)
%I/P 
% load mnist_49_3000

%Parameters
p = 2;

[d,n] = size(X_train);

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

%Use half of the Train vectors for holdout C estimation
%holdout training
n_ho = round(n/2);
X_train_ho = X_train(:,1:n_ho);
Y_train_ho = Y_train(1:n_ho);

X_test_ho = X_train(:,n_ho+1:end);
Y_test_ho = Y_train(n_ho+1:end);


%%
%Kernel Matrix
A = X_train_ho;
K = (A'*A + 1).^p;

C_ho = logspace(log10(0.1),log10(1000),15);
for i = 1:length(C_ho)
    i
    [alpha_ho(i,:), b_ho(i)] = smo(K, Y_train_ho, C_ho(i), 0.001);
end

%%
error_ho = zeros(1,length(C_ho));

for i = 1:length(C_ho)
    K_ho = (X_train_ho' * X_test_ho + 1).^p;
    Y_ans_ho = sign( (alpha_ho(i,:).*Y_train_ho) *K_ho + b_ho(i) );
    
    error_ho(i) = sum(Y_test_ho ~= Y_ans_ho)/length(Y_ans_ho);
end

[error_ho*100 ; C_ho]

[min_err min_ind] = min(error_ho);

C = C_ho(min_ind)




%% After C is chosen run svm on all data

%Kernel Matrix
A = X_train;
K_train = (A'*A + 1).^p;

[alpha, b] = smo(K_train, Y_train, C, 0.0001);

K_test = (X_train' * X_test + 1).^p;
Y_predict = sign( (alpha.*Y_train) *K_test + b );
%num_support_vec = sum(abs((alpha.*Y_train) *K_test + b) <= 1)
sum(alpha > 0);%Num support vectors

sprintf('Number of training samples = %d %', length(Y_train))
sprintf('Number of test samples = %d %', length(Y_test))

test_error = sum(Y_test ~= Y_predict)/length(Y_predict) *100;
sprintf('Test Error = %f %', test_error)

training_error = sum(sign( (alpha.*Y_train) *K_train + b ) ~= Y_train) / length(Y_train);
sprintf('Training Error = %f %', training_error)
%%
%Figure to plot 4x5 image 
% clear sort ind
% support_vec_bool = (alpha > 0);
% %eta = support_vec_bool .* abs(Y_train.*(alpha.*Y_train) *K_train + b);
% v = (abs((alpha.*Y_train) *K_train + b));
% 
% [sortd, ind] = sort(v,'ascend');%,'descend')
% 
% for i = 1:20
%     subplot(4,5,i);
%     imshow(reshape(X_train(:,ind(i)),[sqrt(d),sqrt(d)])');
%     if Y_train(ind(i)) == 0
%         title('4');
%     else
%         title('9');
%     end
%     %title( (sprintf('A=%d P=%d', label_test, label_predict)));
%     %title( (sprintf('%d ', label_test)));
% end


end
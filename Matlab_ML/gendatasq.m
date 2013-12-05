% This script generates a dataset of 1000 samples
% an NxN image with a MxM black spot. Need to train neural network
% to recognize the black spot
%-------------------------------------------------------------------------

function [X_train, Y_train, X_test, Y_test] = gendatasq(N, M, Nsamptrain, Nsamptest)
%N = 8;
%M = 3;          %Note M must be odd

%Nsamp = 50;

C = round((M+1)/2 + (N-M).*rand(Nsamptrain,2));    %centroids randomly picked

X_train = zeros(N*N,Nsamptrain);
Y_train = zeros(2*N,Nsamptrain);

for i=1:Nsamptrain
    W_im = ones(N); 
    W_im(C(i,1)-((M-1)/2):C(i,1)+((M-1)/2), C(i,2)-((M-1)/2):C(i,2)+((M-1)/2)) = zeros(M);
    w_im_vec = reshape(W_im,N*N,1);
    X_train(:,i) = w_im_vec;
    Y_train(C(i,1),i)=1;
    Y_train(N+C(i,2),i)=1;
end

C = round((M+1)/2 + (N-M).*rand(Nsamptest,2));    %centroids randomly picked

X_test = zeros(N*N,Nsamptest);
Y_test = zeros(2*N,Nsamptest);

for i=1:Nsamptest
    W_im = ones(N); 
    W_im(C(i,1)-((M-1)/2):C(i,1)+((M-1)/2), C(i,2)-((M-1)/2):C(i,2)+((M-1)/2)) = zeros(M);
    w_im_vec = reshape(W_im,N*N,1);
    X_test(:,i) = w_im_vec;
    Y_test(C(i,1),i)=1;
    Y_test(N+C(i,2),i)=1;
end


% figure(1);
% imagesc(W_im);
% colormap('gray');
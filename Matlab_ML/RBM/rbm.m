% this function implements a restricted Boltzmann machine with one 
% visible layer and one hidden layer
% reference: http://blog.echen.me/2011/07/18/introduction-to-restricted-boltzmann-machines/

numVisible = 6;
numHidden = 2; 
learning_rate =0.1;
epochs = 2000;
data = [1,1,1,0,0,0;
        1,0,1,0,0,0;
        1,1,1,0,0,0;
        0,0,1,1,1,0;
        0,0,1,1,0,0;
        0,0,1,1,1,0];

trainSize = size(data,1);    
    
  

% randomize the weights
W = 0.1*randn(numVisible, numHidden)';
% W = [zeros(numVisible,1), W]; % insert zeros for bias
% W = [zeros(1, numHidden+1); W];


%% train the model
[Weights, b  d, h]= trainRBM(data, W, epochs, learning_rate);
% d is the predicted binary input

%% test the model
testData= [1 0 1 0 0 0 ; % expected h = [1 0]
    1 1 1 1 0 0 ; % expected h = [1 0]
    0 0 0 0 1 1 ; % expected h = [0 1]
    1 0 1 1 1 0]; % expected h = [0 1]

[h label] = run_visible( testData,Weights,b,numVisible, numHidden)
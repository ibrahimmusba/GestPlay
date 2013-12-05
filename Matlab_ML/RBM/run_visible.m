function [h label]= run_visible( data,weights,b, numVisible, numHidden)
%     
%     Assuming the RBM has been trained (so that weights for the network have been learned),
%     run the network on a set of visible units, to get a sample of the hidden units.
%     
%     Parameters
%     ----------
%     data: A matrix where each row consists of the states of the visible units.
%     
%     Returns
%     -------
%     hidden_states: A matrix where each row consists of the hidden units activated from the visible
%     units in the data matrix passed in.

    logistic = @(x) ((1+exp(-x)).^-1);  
    num_examples = size(data,1);
    
%     Create a matrix, where each row is to be the hidden units (plus a bias unit)
%     sampled from a training example.
    hidden_states = ones(num_examples, numHidden + 1);
    
%     Insert bias units of 1 into the first column of data.
    data = [ones(num_examples,1) data];

%     Calculate the activations of the hidden units.
    weights = [b' ; weights'] ; % append b as the first row
    hidden_activations = data*weights;
%     Calculate the probabilities of turning the hidden units on.
    hidden_probs = logistic(hidden_activations);
%     Turn the hidden units on with their specified probabilities.
    hidden_states = hidden_probs > 0.5;
%     Always fix the bias unit to 1.
%      hidden_states(:,1) = 1; % comment it for now
  
%     Ignore the bias units and return state or probability.
     h = hidden_probs;%(:,2:end);
    [m,label] = max(h,[],2);

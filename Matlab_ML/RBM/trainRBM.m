function [weights, b,d, h]= trainRBM(data, wts, epochs, learning_rate)

% data = each ROW is a feature vector
% wts = weight matrix of dimension numHidden X numVisible  


[numVisible numHidden] = size(wts');

weights= [zeros(numVisible,1), wts']; % insert zeros for bias
weights = [zeros(1, numHidden+1); weights];

logistic = @(x) ((1+exp(-x)).^-1); 
num_examples = size(data,1);
data = [ones(num_examples,1), data]; % insert the first column of ones for bias 
for i=1:epochs
    % postive CD phase, the reality phase
    pos_hidden_activation   = data*weights;
    pos_hidden_probs        = logistic(pos_hidden_activation);
    pos_hidden_states = pos_hidden_probs > 0.5;
     
    pos_associations = data'*pos_hidden_probs ;
    
    
    % Reconstruct the visible units and sample again from the hidden units.
    % (This is the "negative CD phase", aka the daydreaming phase.) 
    neg_visible_activations = pos_hidden_states*weights';
    neg_visible_probs = logistic(neg_visible_activations);
    neg_visible_probs(:,1) = 1 ;% Fix the bias unit.
    neg_hidden_activations = neg_visible_probs*weights;
    neg_hidden_probs = logistic(neg_hidden_activations);
    % Note, again, that we're using the activation *probabilities* when computing associations, not the states 
    % themselves.
    neg_associations = neg_visible_probs'*neg_hidden_probs;

    % Update weights.
    weights = weights + learning_rate * ((pos_associations - neg_associations) / num_examples);

    error = sum(sum((data - neg_visible_probs).^2))/(size(data,1)*size(data,2));
    [i error]

end
b= weights(1,2:end)';
weights = weights(2:end, 2:end)';
d = neg_visible_probs(:,2:end)>0.5;
h = pos_hidden_states(:,2:end)>0.5; 
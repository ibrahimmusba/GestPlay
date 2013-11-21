function [cost Wgrad bgrad] = evaluateNeuralNetwork(W, b, ip, op, n_l)

%Vectorized Implementation of above code
    [a, z] = doForwardPass(W,b,ip,n_l);
    a{n_l}
    %rho = sparsityParam;
    %rho_hat = 1./m*(sum(a2,2));
    [Wgrad bgrad] = backpropagate(W, b, ip, op, n_l, a, z);
    
    cost1 = 1/(2*size(ip,2))*trace((op - a{n_l})'*(op - a{n_l}));
    %cost2 = (lambda/2)*(sum(sum(W1.^2)) + sum(sum(W2.^2)));
    %cost3 = beta.*sum(rho.*(log(rho./rho_hat))+(1-rho).*(log((1-rho)./(1-rho_hat))));
    cost =  cost1; % + cost2 + cost3;
end
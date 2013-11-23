function [cost grad] = evaluateNeuralNetwork(theta, ip, op, n_l, n_nodes, lambda)

[W b] = convertThetaToWb(theta, n_nodes);

[a, z] = doForwardPass(W,b,ip,n_l);

%rho = sparsityParam;
%rho_hat = 1./m*(sum(a2,2));
[Wgrad bgrad] = backpropagate(W, b, ip, op, n_l, a, z, lambda);

grad = convertWbToTheta(Wgrad, bgrad);

cost1 = 1/(2*size(ip,2))*trace((op - a{n_l})'*(op - a{n_l}));
cost2 = 0;
for l = 1:n_l-1
    cost2 = cost2 + (lambda/2)*(sum(sum(W{l}.^2)));
end
%cost3 = beta.*sum(rho.*(log(rho./rho_hat))+(1-rho).*(log((1-rho)./(1-rho_hat))));
cost =  cost1 + cost2;% + cost3;


end
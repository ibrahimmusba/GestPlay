function [Wgrad bgrad] = backpropagate(W, b, ip, op, n_l, a, z, lambda)


%Compute all delta
%delta = d(error)/d(z)
m = size(ip,2);
    
delta = cell(1,n_l); % Range from 2:n_l

% TODO: Not generalized for multiple outputs
delta{n_l} = -1*(op - a{n_l}).*a{n_l}.*(1-a{n_l});
for l = n_l-1:-1:2
    %delta{l} = d_sig(z{l}).*(W{l}'*delta{l+1});
    delta{l} = a{l}.*(1-a{l}).*(W{l}'*delta{l+1});
    %beta.*((-rho./rho_hat)+((1-rho)./(1-rho_hat)))*ones(1,m)
end


%Compute partial derivatives
for l = 1:n_l-1
    Wgrad{l} = ((1./m)*delta{l+1}*a{l}') + lambda*W{l};
    bgrad{l} = 1./m*sum(delta{l+1},2); 
end


end
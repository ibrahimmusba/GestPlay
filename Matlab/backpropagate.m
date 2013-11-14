function [d_W d_b] = backpropagate(W, b, ip, op, NN, a, z, sig, d_sig)

%n_ip = NN{1};                  
%n_op = NN{2};                  
n_l = NN{3};              
%n_nodes = NN{4};

%Compute all gamma
%gamma = d(error)/d(z)
gamma{n_l} = -1*(op - a{n_l}).*a{n_l}.*(1-a{n_l});
for l = n_l-1:-1:2
    %gamma{l} = d_sig(z{l}).*(W{l}'*gamma{l+1});
    gamma{l} = a{l}.*(1-a{l}).*(W{l}'*gamma{l+1});
end


%Compute partial derivatives
for l = 1:n_l-1
    d_W{l} = gamma{l+1}*a{l}';
    d_b{l} = gamma{l+1};
end


end
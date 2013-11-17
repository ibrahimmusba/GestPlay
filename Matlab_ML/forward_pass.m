function [a, z] = forward_pass(W,b,ip,NN)

n_ip = NN{1};                  
n_op = NN{2};                  
n_l = NN{3};              
n_nodes = NN{4};

a{1} = ip;
m = size(ip,2);

for i = 1:n_l-1
    z{i+1} = W{i}*a{i} + repmat(b{i},1,m);
    a{i+1} = 1./(1+exp(-z{i+1}));
end


end
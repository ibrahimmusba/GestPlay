function [a, z] = doForwardPass(W,b,ip,n_l)

a{1} = ip;
m = size(ip,2);

for i = 1:n_l-1
    z{i+1} = W{i}*a{i} + repmat(b{i},1,m);
    a{i+1} = 1./(1+exp(-z{i+1}));
end


end
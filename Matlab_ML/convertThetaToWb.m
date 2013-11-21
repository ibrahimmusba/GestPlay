function [W b] = convertThetaToWb(theta, n_nodes)

n_l = length(n_nodes);

W = cell(1,n_l-1);
b = cell(1,n_l-1);

ind = 1;
for i = 1:n_l-1
    rows = n_nodes(i+1);
    cols = n_nodes(i);
    W{i} = reshape(theta(ind:ind-1+rows*cols), rows, cols);
    ind = ind + rows*cols;
end

for i = 1:n_l-1
   b{i} = theta(ind:ind-1+n_nodes(i+1));
   ind = ind + n_nodes(i+1);
end



end
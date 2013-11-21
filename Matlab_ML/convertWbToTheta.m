function [theta] = convertWbToTheta(W,b)

theta = [];

n_l = size(W,2) + 1;

for i = 1:n_l-1
   theta = [theta;W{i}(:)];
end

for i = 1:n_l-1
   theta = [theta;b{i}(:)];
end



end
function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

epsilon = 1e-4;
for i = 1:length(theta)
    
    theta_tmp = theta;
    theta_tmp(i) = theta_tmp(i)+epsilon; 
    f_plus = J(theta_tmp);
    
    theta_tmp = theta;
    theta_tmp(i) = theta_tmp(i)-epsilon; 
    f_minus = J(theta_tmp);
    
    numgrad(i) = (f_plus - f_minus)/(2*epsilon);
end

%% ---------------------------------------------------------------
end

% function [d_W d_b] = computeNumericalGradients( W, b, ip, op, NN, a, z, sig, d_sig )
% n_l = NN{3};
% 
% [a, ~] = forward_pass(W,b,ip,NN);
% f = 1/2*(a{n_l} - op)^2;
% 
% for l = 1:n_l-1
%         d_W{l} = zeros(size(W{l}));
%         d_b{l} = zeros(size(b{l}));
% end
%     
% epsilon = 0.001;
% 
% for l = 1:n_l-1
%     for i = 1:size(W{l},1)
%         %Compute all W gradients
%         for j = 1:size(W{l},2)
%             
%             W_tmp = W;
%             W_tmp{l}(i,j) = W_tmp{l}(i,j) + epsilon;
%             [a, ~] = forward_pass(W_tmp,b,ip,NN);
%             fW_plus = 1/2*(a{n_l} - op)^2;
%             
%             W_tmp = W;
%             W_tmp{l}(i,j) = W_tmp{l}(i,j) - epsilon;
%             [a, ~] = forward_pass(W_tmp,b,ip,NN);
%             fW_minus = 1/2*(a{n_l} - op)^2;
%             
%             d_W{l}(i,j) = (fW_plus - fW_minus)/(2*epsilon);
%         end
%         
%         %Compute all b gradients
%         b_tmp = b;
%         b_tmp{l}(i) = b_tmp{l}(i) + epsilon;
%         [a, ~] = forward_pass(W,b_tmp,ip,NN);
%         fb_plus = 1/2*(a{n_l} - op)^2;
% 
%         b_tmp = b;
%         b_tmp{l}(i) = b_tmp{l}(i) - epsilon;
%         [a, ~] = forward_pass(W,b_tmp,ip,NN);
%         fb_minus = 1/2*(a{n_l} - op)^2;
%         
%         d_b{l}(i) = (fb_plus - fb_minus)/(2*epsilon);
%     end
% end
% 
% 
% 
% end


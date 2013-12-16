function numgrad = computeNumericalGradient(J, theta)
% numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% 

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

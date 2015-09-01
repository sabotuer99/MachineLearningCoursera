function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

	% psuedocode from Gradient Descent lecture
	% temp0 = theta0 - alpha * J(theta0, theta1)
	% temp1 = theta1 - alpha * J(theta0, theta1)
	% theta0 = temp0
	% theta1 = temp1
	%
	% from Gradient Descent for Linear Regression video (~2:30 mark)
	% theta0 partial derivative = 1/m E (h0(x) - y)
	% theta1 partial derivative = 1/m E (h0(x) - y) * x
	%
	% h0(x) - y can be computed as in the computeCost function ((X - theta) - y)
	% the equation for theta1 requires elementwise multiplication with x, which is
	% the second column of X, which requires the X(:,2) Octave syntax
	
	temp0 = theta(1) - alpha * sum((X * theta) - y)/m;
	temp1 = theta(2) - alpha * sum(((X * theta) - y).*X(:,2))/m;
	theta(1) = temp0;
	theta(2) = temp1;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end

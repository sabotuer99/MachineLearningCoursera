function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%size(X)
%size(y)
%size(theta)

%x = X(:,2:end);
%m = length(theta);

%J = 1/m * sum(-y .* log(sigmoid(x)) - (1 - y) .* log(1 - sigmoid(x)));
%grad = 1/m * sum(sigmoid(x) .- y) .* x;
% had no idea how to proceed, Googled for some help... I was kind close
% http://dudarev.com/wiki/ml-class-logistic-regression.html

%x = X * theta;
%J = 1/m * sum(-y' .* log(sigmoid(x)) - (1 - y') .* log(1 - sigmoid(x)));
%grad = 1/m * X' * (sigmoid(x) - y);

%https://github.com/schneems/Octave/blob/master/mlclass-ex2/mlclass-ex2/costFunction.m
%J = 1/m * (-y'* log(sigmoid(x) - y) - (1 - y)' * log(1 - (sigmoid(x) - y)));

h = sigmoid(X*theta);
J = (1/m)*(-y'* log(h) - (1 - y)'* log(1-h));
grad = (1/m)*X'*(h - y);

% =============================================================

end

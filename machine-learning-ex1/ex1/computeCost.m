function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% J(0) = 1/2m E(h0(x) - y)^2
% matrix multiplication, X * theta, to calculate h0(x), as in Matrix Vector Multiplication lecture (11:30 mark)
% subtract actual values (X * theta) - y to calculate prediction errors
% elementwise power ((X * theta) - y).^2 to calculate squared errors
% sum elements of resulting vector sum(((X * theta) - y).^2)
% divide by 2m (same as multiplying by 1/(2m) ... assign to J:


J = sum(((X * theta) - y).^2)/(2*m)


% http://math.jacobs-university.de/oliver/teaching/iub/resources/octave/octave-intro/octave-intro.html
% https://github.com/schneems/Octave/blob/master/mlclass-ex1/computeCost.m
% https://www.gnu.org/software/octave/doc/interpreter/Sums-and-Products.html
% =========================================================================

end

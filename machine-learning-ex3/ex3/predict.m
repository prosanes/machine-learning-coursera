function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

AllA3 = zeros(m, num_labels);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for example = 1:m
  a1 = [1 X(example, :)]';
  a2 = [1; sigmoid(Theta1 * a1)];
  a3 = sigmoid(Theta2 * a2);
  AllA3(example, :) = a3;
end

% FIXED VERSION. Not tested. Cheated looking at predict.m from ex4
% A1 = [ones(m,1) X]';
% A2_tmp = Theta1 * A1;
% A2 = sigmoid(A2_tmp);
% A3_tmp = [ones(1,size(A2, 2)); A2];
% A3 = sigmoid(Theta2 * A2);
% AllA3 = A3';

[_, p] = max(AllA3, [], 2);




% =========================================================================


end

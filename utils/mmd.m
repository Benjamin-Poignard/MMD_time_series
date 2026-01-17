function MMD_est = mmd(Y,X,sigma)

% MMD loss function

% inputs: - Y: T x p matrix containing the observed time series
%         - X: N x p matrix containing the simulated time series
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: - MMD_est: MMD loss function

m=size(X,1); n = size(Y,1);
L = rbf_dot(Y,Y,sigma);
K = rbf_dot(X,X,sigma);
KL = rbf_dot(X,Y,sigma);

MMD_est = (sum(sum(K))/m^2)+(sum(sum(L))/n^2)-2*(sum(sum(KL))/(m*n));
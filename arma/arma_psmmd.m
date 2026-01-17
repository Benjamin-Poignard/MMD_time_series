function loss = arma_psmmd(param,data,error,p_lag,q_lag,TT,p,sigma)

% Compute MMD loss function for ARMA-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: 1 x TT vector for innovation u_t
%         - p_lag: AR order
%         - q_lag: MA order
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

y = generate_arma_T(param,error,p_lag,q_lag,TT);

Y_sim = zeros(TT-p,p+1); Y_sim(:,1) = y(p+1:end);
for i=1:p
    Y_sim(:,i+1) = y(p-i+1:end-i);
end
loss = mmd(data,Y_sim,sigma);
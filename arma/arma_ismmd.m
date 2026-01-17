function loss = arma_ismmd(param,data,error,p_lag,q_lag,TT,p,N,sigma)

% Compute MMD loss function for ARMA-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: TT x N matrix for innovation u_t
%         - p_lag: AR order
%         - q_lag: MA order
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

y = generate_arma(param,error,p_lag,q_lag,TT,N);

Y_sim = zeros(N,p+1);
for oo=1:N
    Y_sim(oo,:) = flip(y(TT-p:end,oo)');
end
loss = mmd(data,Y_sim,sigma);
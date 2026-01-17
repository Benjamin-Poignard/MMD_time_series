function loss = NLts_ismmd(param,data,error,TT,p,N,sigma)

% Compute MMD loss function for non-linear MA-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: innovation u_t
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

y = zeros(TT,N); 
for t = 2:TT
    y(t,:) = error(t,:) + param*error(t-1,:).^2;
end
Y_sim = zeros(N,p+1);
for oo=1:N
    Y_sim(oo,:) = flip(y(TT-p:end,oo)');
end
loss = mmd(data,Y_sim,sigma);
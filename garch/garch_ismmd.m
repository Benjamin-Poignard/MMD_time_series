function loss = garch_ismmd(param,data,error,TT,p,N,sigma)

% Compute MMD loss function for GARCH-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: innovation u_t
%         - TT: T0 length as in ISMMD
%         - p: lag to construct \tilde{y}_i: x^{(i)}_{TT},...,x^{(i)}_{TT-p}
%         - N: length of the simulated vector \tilde{y}_i
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

a = param(1); b = param(2); c = param(3);

y = zeros(TT,N); var_cond = zeros(TT,N);
var_cond(1,:) = a/(1-b-c)*ones(1,N); y(1,:) = sqrt(var_cond(1,:)).*error(1,:);
for t = 2:TT
    var_cond(t,:) = a + b*var_cond(t-1,:) + c*y(t-1,:).^2;
    y(t,:) = sqrt(var_cond(t,:)).*error(t,:);
end
Y_sim = zeros(N,p+1);
for oo=1:N
    Y_sim(oo,:) = flip(y(TT-p:end,oo)');
end
loss = mmd(data,Y_sim,sigma);
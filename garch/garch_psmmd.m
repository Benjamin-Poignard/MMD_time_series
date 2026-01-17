function loss = garch_psmmd(param,data,error,TT,p,sigma)

% Compute MMD loss function for SV-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error: innovation u_t
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

a = param(1); b = param(2); c = param(3);

y = zeros(TT,1); var_cond = zeros(TT,1);
var_cond(1) = a/(1-b-c); y(1) = sqrt(var_cond(1)).*error(1);
for t = 2:TT
    var_cond(t) = a + b*var_cond(t-1) + c*y(t-1)^2;
    y(t) = sqrt(var_cond(t)).*error(t);
end
Y_sim = zeros(TT-p,p+1); Y_sim(:,1) = y(p+1:end);
for i=1:p
    Y_sim(:,i+1) = y(p-i+1:end-i);
end
loss = mmd(data,Y_sim,sigma);
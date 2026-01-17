function loss = sv_psmmd(param,data,error1,error2,TT,p,sigma)

% Compute MMD loss function for SV-based time series

% inputs: - param: vector of parameters
%         - data: T x 1 vector containing the time series
%         - error1: innovation v_t
%         - error2: innovation \eta_t  ==> u_t = (v_t,\eta_t)'
%         - TT: length of simulated data for PSMMD
%         - p: lag to construct \tilde{y}_i: x_{i},...,x_{i-p}, i=1,...,TT
%         - sigma: Gaussian kernel hyperparameter set by median heuristic

% output: loss: MMD loss function

b = param(1); sigma_e = param(2); sigma_y = param(3);
y = zeros(TT,1); var_sto = zeros(TT,1); var_sto(1) = var(data(:,1));
for t = 2:TT
    var_sto(t) = b*var_sto(t-1) + sigma_e*error1(t);
    y(t) = sigma_y.*exp(var_sto(t)/2).*error2(t);
end
Y_sim = zeros(TT-p,p+1); Y_sim(:,1) = y(p+1:end);
for i=1:p
    Y_sim(:,i+1) = y(p-i+1:end-i);
end
loss = mmd(data,Y_sim,sigma);
% The following code allows to replicate the simulation experiment for
% SV model, GARCH model, ARMA model, non-linear MA model and Ricker model
% and for one batch. To replicate the results averaged over 100 batches for
% each model, please check the folder 'sim'

% param_ISMMD1: ISMMD estimator, obtained by gradient_sv_ismmd when N = 1000
% and T = 100

% param_ISMMD2: ISMMD estimator, obtained by gradient_sv_ismmd when N = 2000
% and T = 100

% param_PSMMD1: PSMMD estimator, obtained by gradient_sv_psmmd when T = 1000

% param_PSMMD2: PSMMD estimator, obtained by gradient_sv_psmmd when T = 2000

% The code for rbf_dot.m, mmd.m and median_heuristic.m were downloaded from
% the website: https://www.gatsby.ucl.ac.uk/~gretton/mmd/mmd.htm
% "A Kernel Two-Sample Test" by Arthur Gretton, Karsten Borgwardt, 
% Malte Rasch, Bernhard Schoelkopf, Alex Smola
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Simulation experiments for SV model %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SV model:
addpath(genpath(pwd))
clear
clc

% max number of lags
max_lag = 16;

param_ISMMD1 = zeros(3,max_lag); param_PSMMD1 = zeros(3,max_lag);
param_ISMMD2 = zeros(3,max_lag); param_PSMMD2 = zeros(3,max_lag);

% True parameters:
% case 1: phi = 0.8; sigma_e = 0.05; sigma_x = 0.15;
% case 2: phi = 0.9; sigma_e = 0.1; sigma_x = 0.2;
phi = 0.8; sigma_e = 0.05; sigma_x = 0.15;
% phi = 0.9; sigma_e = 0.1; sigma_x = 0.2;

param_true = [phi,sigma_e,sigma_x];
% (True) distribution of the innovations
dist = 'Gaussian'; % dist = 'Student';

% Date generation
T = 300; % T = 1000 (sample size)
Y_obs = simulate_sv(param_true,T,dist);

% p = 0 lag
p=0; Y_p = zeros(T-p,p+1);
Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_sv_psmmd(Y_p,1000,p,sigma);
param_PSMMD2(:,1) = gradient_sv_psmmd(Y_p,2000,p,sigma);

% p = 1,...,15 lag
parfor p = 1:max_lag-1
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,p+1) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,p+1) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,p+1) = gradient_sv_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,p+1) = gradient_sv_psmmd(Y_p,2000,p,sigma);
end

% Estimation via particle filter
optimoptions.MaxRLPIter = 5000;
optimoptions.MaxFunEvals = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';
startcoeff = [0.75+(0.95-0.75)*rand(1),0.01+(0.15-0.01)*rand(1),0.1+(0.15)*rand(1)]';
[parameters_sv,~,~,~,~,~]=fmincon(@(x)sv_particle_filter_loglik(x,Y_obs),startcoeff,[],[],[],[],[],[],@(x)constr_sv(x),optimoptions);
param_MLE = parameters_sv;

% Compute the l2-error
error_ISMMD1 = zeros(max_lag,1); error_PSMMD1 = zeros(max_lag,1);
error_ISMMD2 = zeros(max_lag,1); error_PSMMD2 = zeros(max_lag,1);
param_true = [phi,sigma_e,sigma_x]';

for k = 1:max_lag
    error_ISMMD1(k) = norm(param_ISMMD1(:,k)-param_true);
    error_PSMMD1(k) = norm(param_PSMMD1(:,k)-param_true);
    error_ISMMD2(k) = norm(param_ISMMD2(:,k)-param_true);
    error_PSMMD2(k) = norm(param_PSMMD2(:,k)-param_true);
end
error_MLE = norm(param_MLE-param_true);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Simulation experiments for GARCH model %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GARCH model
addpath(genpath(pwd))
clear
clc

% max number of lags
max_lag = 16;

param_ISMMD1 = zeros(3,max_lag); param_PSMMD1 = zeros(3,max_lag);
param_ISMMD2 = zeros(3,max_lag); param_PSMMD2 = zeros(3,max_lag);

% True parameters:
% case 1: omega = 0.05; beta = 0.85; alpha = 0.1;
% case 2: omega = 0.05; beta = 0.92; alpha = 0.05;
omega = 0.05; beta = 0.85; alpha = 0.1;
param_true = [omega,beta,alpha];

% (True) distribution of the innovations
dist = 'Gaussian'; % dist = 'Student';

% GARCH model simulation
T = 300; % T = 1000 (sample size)
Y_obs = simulate_garch(param_true,T,dist);

% p = 0 lag
p=0; Y_p = zeros(T-p,p+1); Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_garch_psmmd(Y_p,1000,p,sigma);
param_PSMMD2(:,1) = gradient_garch_psmmd(Y_p,2000,p,sigma);

% p = 1,...,15 lag
parfor p = 1:max_lag-1
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,p+1) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,p+1) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,p+1) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,p+1) = gradient_garch_psmmd(Y_p,2000,p,sigma);
end

% Estimation by Gaussian QMLE
EstMdl = estimate(garch(1,1),Y_obs);
param_MLE = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];

% Compute the l2-error
error_ISMMD1 = zeros(max_lag,1); error_PSMMD1 = zeros(max_lag,1);
error_ISMMD2 = zeros(max_lag,1); error_PSMMD2 = zeros(max_lag,1);
param_true = [omega,beta,alpha]';

for k = 1:max_lag
    error_ISMMD1(k) = norm(param_ISMMD1(:,k)-param_true);
    error_PSMMD1(k) = norm(param_PSMMD1(:,k)-param_true);
    error_ISMMD2(k) = norm(param_ISMMD2(:,k)-param_true);
    error_PSMMD2(k) = norm(param_PSMMD2(:,k)-param_true);
end
error_MLE = norm(param_MLE-param_true);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Simulation experiments for ARMA model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARMA model
addpath(genpath(pwd))
clear
clc

% max number of lags
max_lag = 16;
param_ISMMD1 = zeros(3,max_lag); param_PSMMD1 = zeros(3,max_lag);
param_ISMMD2 = zeros(3,max_lag); param_PSMMD2 = zeros(3,max_lag);

% AR and MA orders:
p_lag = 1; q_lag = 1;

% True coefficients
% case 1: Phi = {0.8}; Psi = {0.15}; Sigma_true = 0.05;
% case 2: Phi = {0.9}; Psi = {0.08}; Sigma_true = 0.03;
Phi = {0.8}; Psi = {0.15}; Sigma_true = 0.05;

% Distribution of the true innovations
dist = 'Gaussian'; % dist = 'Student';

% ARMA model simulation
T = 300; % T = 1000 (sample size)
Y_obs = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);

% p = 0 lag
p=0; Y_p = zeros(T-p,p+1); Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
param_PSMMD2(:,1) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);

% p = 1,...,15 lag
parfor p = 1:max_lag-1
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,p+1) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
    param_ISMMD2(:,p+1) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
    param_PSMMD1(:,p+1) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
    param_PSMMD2(:,p+1) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);
end

% Estimation by Gaussian QML
model = arima(1,0,1); model.Constant = 0; EstMdl = estimate(model,Y_obs);
param_MLE = [EstMdl.AR{1};EstMdl.MA{1};EstMdl.Variance];

% Compute the l2-error
error_ISMMD1 = zeros(max_lag,1); error_PSMMD1 = zeros(max_lag,1);
error_ISMMD2 = zeros(max_lag,1); error_PSMMD2 = zeros(max_lag,1);
param_true = [Phi{1} Psi{1} Sigma_true]';

for k = 1:max_lag
    error_ISMMD1(k) = norm(param_ISMMD1(:,k)-param_true);
    error_PSMMD1(k) = norm(param_PSMMD1(:,k)-param_true);
    error_ISMMD2(k) = norm(param_ISMMD2(:,k)-param_true);
    error_PSMMD2(k) = norm(param_PSMMD2(:,k)-param_true);
end
error_MLE = norm(param_MLE-param_true);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Simulation experiments for non-linear MA model %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Non-linear model
addpath(genpath(pwd))
clear
clc

% max number of lags
max_lag = 41;
param_ISMMD1 = zeros(max_lag,1); param_ISMMD2 = zeros(max_lag,1);
param_PSMMD1 = zeros(max_lag,1); param_PSMMD2 = zeros(max_lag,1);

% True coefficients:
% case 1: param_PSMMD1rue = 0.7;
% case 2: param_PSMMD1rue = 0.9;
param_true = 0.9;

% Distribution of the true innovations
dist = 'Gaussian'; % dist = 'Student';

% Ricker model simulation
T = 300; % T = 1000 (sample size)
Y_obs = simulate_NLts(param_true,T,dist);

% p = 0 lag
p=0; Y_p = zeros(T-p,p+1); Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(1) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
param_ISMMD2(1) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
param_PSMMD1(1) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
param_PSMMD2(1) = gradient_NLts_psmmd(Y_p,2000,p,sigma);

% p = 1,...,40 lag
parfor p = 1:max_lag-1
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(p+1) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(p+1) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(p+1) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(p+1) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
end

% Estimation by method of moment: sample average
param_mom = mean(Y_obs);

% Compute the l2-error
error_ISMMD1 = zeros(max_lag,1); error_PSMMD1 = zeros(max_lag,1);
error_ISMMD2 = zeros(max_lag,1); error_PSMMD2 = zeros(max_lag,1);

for k = 1:max_lag
    error_ISMMD1(k) = norm(param_ISMMD1(k)-param_true);
    error_PSMMD1(k) = norm(param_PSMMD1(k)-param_true);
    error_ISMMD2(k) = norm(param_ISMMD2(k)-param_true);
    error_PSMMD2(k) = norm(param_PSMMD2(k)-param_true);
end
error_mom = norm(param_mom-param_true);

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Simulation experiments for Ricker model %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ricker model
addpath(genpath(pwd))
clear
clc

% max number of lags
max_lag = 16;
param_ISMMD1 = zeros(3,max_lag); param_ISMMD2 = zeros(3,max_lag);
param_PSMMD1 = zeros(3,max_lag); param_PSMMD2 = zeros(3,max_lag);

% True coefficients:
% case 1: logr = log(5); sigma_e = 0.03; phi = 5;
% case 2: logr = log(7); sigma_e = 0.05; phi = 7;
logr = log(7); sigma_e = 0.05; phi = 7;
param_true = [logr,sigma_e,phi];

% Distribution of the true innovations
dist = 'Gaussian'; % dist = 'Student';

% Ricker model simulation
T = 300; % T = 1000 (sample size)
Y_obs = simulate_ricker(param_true,T,dist);

% p = 0 lag
p=0; Y_p = zeros(T-p,p+1); Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
param_PSMMD2(:,1) = gradient_ricker_psmmd(Y_p,2000,p,sigma);

% p = 1,...,15 lag
parfor p = 1:max_lag-1
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,p+1) = gradient_ricker_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,p+1) = gradient_ricker_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,p+1) = gradient_ricker_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,p+1) = gradient_ricker_psmmd(Y_p,2000,p,sigma);
end

% Estimation by synthetic likelihood (SL method)
lb = [log(1),0.01,1]; ub = [log(10),0.15,15];
Rsim = 1000;
opts = optimoptions('patternsearch','Display','iter', ...
    'MaxFunctionEvaluations',50000,'MaxIterations',10000, ...
    'UseCompletePoll',true,'PollMethod','GSSPositiveBasis2N');
startcoeff = [log(4+(9-4)*rand(1));log(0.01+(0.1-0.01)*rand(1));log(4+(9-4)*rand(1))];
[theta_SL, fval] = patternsearch(@(x)neglog_synthlik_ricker(x,Y_obs,Rsim), startcoeff, [], [], [], [], lb, ub, [], opts);
param_SL = theta_SL';

% Compute the l2-error
error_ISMMD1 = zeros(max_lag,1); error_PSMMD1 = zeros(max_lag,1);
error_ISMMD2 = zeros(max_lag,1); error_PSMMD2 = zeros(max_lag,1);
param_true = [Phi{1} Psi{1} Sigma_true]';

for k = 1:max_lag
    error_ISMMD1(k) = norm(param_ISMMD1(:,k)-param_true);
    error_PSMMD1(k) = norm(param_PSMMD1(:,k)-param_true);
    error_ISMMD2(k) = norm(param_ISMMD2(:,k)-param_true);
    error_PSMMD2(k) = norm(param_PSMMD2(:,k)-param_true);
end
error_MLE = norm(param_MLE-param_true);
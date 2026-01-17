% The following code allows to replicate the simulation experiment for
% the selection of an optimal lag p based on MMD criterion for non-linear
% model, ARMA model and GARCH model

% To replicate results for the MMD-based information criterion, the
% following data should be used: Y_NLts_valid.mat, Y_arma_valid.mat,
% Y_garch_valid.mat
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% Simulation experiments for non-linear MA model %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Non-linear MA model // Student case
addpath(genpath(pwd))
clear
clc
rng(10,'twister');
max_lag = 41;
param_ISMMD1 = zeros(1,max_lag); param_ISMMD2 = zeros(1,max_lag);
param_PSMMD1 = zeros(1,max_lag); param_PSMMD2 = zeros(1,max_lag);

% param_PSMMD1rue = 0.9;
% dist = 'Student';
% Non linear model simulation
%T = 1000; %Y_NLts = simulate_NLts(param_PSMMD1rue,T,dist);
load('Y_NLts_valid.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In sample estimation (training sample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_0 = round(0.75*length(Y_NLts)); Y_obs = Y_NLts(1:T_0);
T_in = length(Y_obs);
p=0;
Y_p = zeros(T_in-p,p+1);
Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
param_PSMMD2(:,1) = gradient_NLts_psmmd(Y_p,2000,p,sigma);

parfor p = 1:max_lag-1
    Y_p = zeros(T_in-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,p+1) = gradient_NLts_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,p+1) = gradient_NLts_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,p+1) = gradient_NLts_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,p+1) = gradient_NLts_psmmd(Y_p,2000,p,sigma);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validation sample for MMD computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y_obs = Y_NLts(1:T_0); Y_valid = Y_NLts(T_0+1:end);
MMD_crit = zeros(max_lag,4);
p = 0; Y_p = zeros(T_in-p,p+1);
Y_p(:,1) = Y_obs(p+1:end); Y_p_valid = Y_valid(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i); Y_p_valid(:,i+1) = Y_valid(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
N = 10000; TT = 100;
MMD_crit(1,1) = NLts_ismmd(param_ISMMD1(:,1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
N = 10000; TT = 100;
MMD_crit(1,2) = NLts_ismmd(param_ISMMD2(:,1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
TT = 10000;
MMD_crit(1,3) = NLts_psmmd(param_PSMMD1(:,1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);
TT = 10000;
MMD_crit(1,4) = NLts_psmmd(param_PSMMD2(:,1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);

for p=1:max_lag-1
    Y_p = zeros(T_in-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end); Y_p_valid = Y_valid(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i); Y_p_valid(:,i+1) = Y_valid(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    N = 10000; TT = 100;
    MMD_crit(p+1,1) = NLts_ismmd(param_ISMMD1(:,p+1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
    N = 10000; TT = 100;
    MMD_crit(p+1,2) = NLts_ismmd(param_ISMMD2(:,p+1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
    TT = 10000;
    MMD_crit(p+1,3) = NLts_psmmd(param_PSMMD1(:,p+1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);
    TT = 10000;
    MMD_crit(p+1,4) = NLts_psmmd(param_PSMMD2(:,p+1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);
end

figure
hold on; plot(10*MMD_crit(:,1),'r','LineWidth',1.5); plot(10*MMD_crit(:,2),'r','LineStyle','-.','LineWidth',1.5); ...
    plot(10*MMD_crit(:,3),'b','LineWidth',1.5); plot(10*MMD_crit(:,4),'b','LineStyle','-.','LineWidth',1.5);

ylim([0 1.5])
yticks(0:0.1:1.5);
yticklabels({'0','0.10','0.20','0.30','0.40','0.50','0.60','0.70','0.80','0.80','1.00','1.10','1.20','1.30','1.40','1.50'})

xticks([1 6 11 16 21 26 31 36 41]);
xticklabels({'0','5','10','15','20','25','30','35','40'});

legend({'\theta^{(1)}_{N,T},N=1000','\theta^{(1)}_{N,T},N=2000','\theta^{(2)}_{N,T},N=1000','\theta^{(2)}_{N,T},N=2000','\theta^{mle}'},'Location','best','FontSize',12)
ylabel('MMD information (x 10)')
xlabel('Number of lags p')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Simulation experiments for ARMA model %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ARMA model // Student case
addpath(genpath(pwd))
clear
clc
rng(5,'twister');
max_lag = 41;
param_ISMMD1 = zeros(3,max_lag); param_ISMMD2 = zeros(3,max_lag);
param_PSMMD1 = zeros(3,max_lag); param_PSMMD2 = zeros(3,max_lag);

% ARMA p and q orders
p_lag = 1; q_lag = 1;

% AR coefficients
Phi = {0.8};
% MA coefficients
Psi = {0.15};
% Covariance matrix for the Gaussian white noise
Sigma_true = 0.05;
%dist = 'Student';

% ARMA model
%T = 1000; Y_arma = simulate_varma(T,p_lag,q_lag,Phi,Psi,Sigma_true,dist);
load('Y_arma_valid.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In sample estimation (training sample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_0 = round(0.75*length(Y_arma)); Y_obs = Y_arma(1:T_0);
T_in = length(Y_obs);
p=0;
Y_p = zeros(T_in-p,p+1);
Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_arma_ismmd(Y_p,p_lag,q_lag,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_arma_psmmd(Y_p,p_lag,q_lag,1000,p,sigma);
param_PSMMD2(:,1) = gradient_arma_psmmd(Y_p,p_lag,q_lag,2000,p,sigma);

parfor p = 1:max_lag-1
    Y_p = zeros(T_in-p,p+1);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validation sample for MMD computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y_obs = Y_arma(1:T_0); Y_valid = Y_arma(T_0+1:end);
MMD_crit = zeros(max_lag,4);
p = 0;
Y_p = Y_obs(p+1:end); Y_p_valid = Y_valid(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i); Y_p_valid(:,i+1) = Y_valid(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
N = 10000; TT = 100;
MMD_crit(1,1) = arma_ismmd(param_ISMMD1(:,1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),p_lag,q_lag,TT,p,N,sigma);
N = 10000; TT = 100;
MMD_crit(1,2) = arma_ismmd(param_ISMMD2(:,1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),p_lag,q_lag,TT,p,N,sigma);
TT = 10000;
MMD_crit(1,3) = arma_psmmd(param_PSMMD1(:,1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),p_lag,q_lag,TT,p,sigma);
TT = 10000;
MMD_crit(1,4) = arma_psmmd(param_PSMMD2(:,1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),p_lag,q_lag,TT,p,sigma);

for p=1:max_lag-1
    Y_p = zeros(T_in-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end); Y_p_valid = Y_valid(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i); Y_p_valid(:,i+1) = Y_valid(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    N = 10000; TT = 100;
    MMD_crit(p+1,1) = arma_ismmd(param_ISMMD1(:,p+1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),p_lag,q_lag,TT,p,N,sigma);
    N = 10000; TT = 100;
    MMD_crit(p+1,2) = arma_ismmd(param_ISMMD2(:,p+1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),p_lag,q_lag,TT,p,N,sigma);
    TT = 10000;
    MMD_crit(p+1,3) = arma_psmmd(param_PSMMD1(:,p+1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),p_lag,q_lag,TT,p,sigma);
    TT = 10000;
    MMD_crit(p+1,4) = arma_psmmd(param_PSMMD2(:,p+1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),p_lag,q_lag,TT,p,sigma);
end

figure
hold on; plot(10*MMD_crit(:,1),'r','LineWidth',1.5); plot(10*MMD_crit(:,2),'r','LineStyle','-.','LineWidth',1.5); ...
    plot(10*MMD_crit(:,3),'b','LineWidth',1.5); plot(10*MMD_crit(:,4),'b','LineStyle','-.','LineWidth',1.5);

ylim([0 0.5])
yticks(0:0.05:0.5);
yticklabels({'0','0.05','0.10','0.15','0.20','0.25','0.30','0.35','0.40','0.45','0.50'})

xticks([1 6 11 16 21 26 31 36 41]);
xticklabels({'0','5','10','15','20','25','30','35','40'});

ylabel('MMD information (x 10)')
xlabel('Number of lags p')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%% Simulation experiments for GARCH model %%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GARCH model // Student case
addpath(genpath(pwd))
clear
clc
rng(8,'twister');
max_lag = 41;
param_ISMMD1 = zeros(3,max_lag); param_ISMMD2 = zeros(3,max_lag);
param_PSMMD1 = zeros(3,max_lag); param_PSMMD2 = zeros(3,max_lag);

a = 0.05; b = 0.92; c = 0.05; param_true = [a,b,c]';
%dist = 'Student';

% Non linear model simulation
%T = 1000; Y_garch = simulate_garch(param_PSMMD1rue,T,dist);
load('Y_garch_valid.mat')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% In sample estimation (training sample)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T_0 = round(0.75*length(Y_garch)); Y_obs = Y_garch(1:T_0);
T_in = length(Y_obs);
p=0;
Y_p = zeros(T_in-p,p+1); Y_p(:,1) = Y_obs(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
param_ISMMD1(:,1) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
param_ISMMD2(:,1) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
param_PSMMD1(:,1) = gradient_garch_psmmd(Y_p,1000,p,sigma);
param_PSMMD2(:,1) = gradient_garch_psmmd(Y_p,2000,p,sigma);

parfor p = 1:max_lag-1
    Y_p = zeros(T_in-p,p+1);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validation sample for MMD computation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Y_obs = Y_garch(1:T_0); Y_valid = Y_garch(T_0+1:end);
MMD_crit = zeros(max_lag,4);
p = 0;
Y_p = Y_obs(p+1:end); Y_p_valid = Y_valid(p+1:end);
for i=1:p
    Y_p(:,i+1) = Y_obs(p-i+1:end-i); Y_p_valid(:,i+1) = Y_valid(p-i+1:end-i);
end
sigma = median_heuristic(Y_p);
N = 10000; TT = 100;
MMD_crit(1,1) = garch_ismmd(param_ISMMD1(:,1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
MMD_crit(1,2) = garch_ismmd(param_ISMMD2(:,1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
TT = 10000;
MMD_crit(1,3) = garch_psmmd(param_PSMMD1(:,1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);
MMD_crit(1,4) = garch_psmmd(param_PSMMD2(:,1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);

for p=1:max_lag-1
    Y_p = zeros(T_in-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end); Y_p_valid = Y_valid(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i); Y_p_valid(:,i+1) = Y_valid(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    N = 10000; TT = 100;
    MMD_crit(p+1,1) = garch_ismmd(param_ISMMD1(:,p+1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
    N = 10000; TT = 100;
    MMD_crit(p+1,2) = garch_ismmd(param_ISMMD2(:,p+1),Y_p_valid,mvnrnd(zeros(N,1),eye(N),TT),TT,p,N,sigma);
    TT = 10000;
    MMD_crit(p+1,3) = garch_psmmd(param_PSMMD1(:,p+1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);
    TT = 10000;
    MMD_crit(p+1,4) = garch_psmmd(param_PSMMD2(:,p+1),Y_p_valid,mvnrnd(zeros(TT,1),eye(TT)),TT,p,sigma);
end

figure
hold on; plot(10*MMD_crit(:,1),'r','LineWidth',1.5); plot(10*MMD_crit(:,2),'r','LineStyle','-.','LineWidth',1.5); ...
    plot(10*MMD_crit(:,3),'b','LineWidth',1.5); plot(10*MMD_crit(:,4),'b','LineStyle','-.','LineWidth',1.5);

ylim([0 0.2])
yticks(0:0.05:0.2);
yticklabels({'0','0.05','0.10','0.15','0.20'})

xticks([1 6 11 16 21 26 31 36 41]);
xticklabels({'0','5','10','15','20','25','30','35','40'});

ylabel('MMD information (x 10)')
xlabel('Number of lags p')
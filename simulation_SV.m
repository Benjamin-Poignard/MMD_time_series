%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% Simulation experiments for SV model %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% True DGP:  phi = 0.8, sigma_e = 0.05, sigma_x = 0.15, T = 300
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
rng(1,'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_PSMMD1 = zeros(3,max_lag,Sim);
param_ISMMD2 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);

param_MLE = zeros(3,Sim);
phi = 0.8; sigma_e = 0.05; sigma_x = 0.15;
param_true = [phi,sigma_e,sigma_x];
dist = 'Gaussian';
% dist = 'Student';

optimoptions.MaxRLPIter = 5000;
optimoptions.MaxFunEvals = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

for oo=1:Sim
    T = 300;
    Y_obs = simulate_sv(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [0.75+(0.95-0.75)*rand(1),0.01+(0.15-0.01)*rand(1),0.1+(0.15)*rand(1)]';
    [parameters_sv,~,~,~,~,~]=fmincon(@(x)sv_particle_filter_loglik(x,Y_obs),startcoeff,[],[],[],[],[],[],@(x)constr_sv(x),optimoptions);
    param_MLE(:,oo) = parameters_sv;
end

%% True DGP:  phi = 0.8, sigma_e = 0.05, sigma_x = 0.15, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
rng(2, 'twister' );

Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_PSMMD1 = zeros(3,max_lag,Sim);
param_ISMMD2 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);
phi = 0.8; sigma_e = 0.05; sigma_x = 0.15;
param_true = [phi,sigma_e,sigma_x];
dist = 'Gaussian';
% dist = 'Student';

optimoptions.MaxRLPIter = 5000;
optimoptions.MaxFunEvals = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

for oo=1:Sim
    T = 1000;
    Y_obs = simulate_sv(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [0.75+(0.95-0.75)*rand(1),0.01+(0.15-0.01)*rand(1),0.1+(0.15)*rand(1)]';
    [parameters_sv,~,~,~,~,~]=fmincon(@(x)sv_particle_filter_loglik(x,Y_obs),startcoeff,[],[],[],[],[],[],@(x)constr_sv(x),optimoptions);
    param_MLE(:,oo) = parameters_sv;
end

%% True DGP:  phi = 0.9, sigma_e = 0.1, sigma_x = 0.2, T = 300
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
rng(3, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_PSMMD1 = zeros(3,max_lag,Sim);
param_ISMMD2 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);

param_MLE = zeros(3,Sim);
phi = 0.9; sigma_e = 0.1; sigma_x = 0.2;
param_true = [phi,sigma_e,sigma_x];
dist = 'Gaussian';
% dist = 'Student';

optimoptions.MaxRLPIter = 5000;
optimoptions.MaxFunEvals = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

for oo=1:Sim
    T = 300;
    Y_obs = simulate_sv(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [0.75+(0.95-0.75)*rand(1),0.01+(0.15-0.01)*rand(1),0.1+(0.15)*rand(1)]';
    [parameters_sv,~,~,~,~,~]=fmincon(@(x)sv_particle_filter_loglik(x,Y_obs),startcoeff,[],[],[],[],[],[],@(x)constr_sv(x),optimoptions);
    param_MLE(:,oo) = parameters_sv;
end

%% True DGP:  phi = 0.9, sigma_e = 0.1, sigma_x = 0.2, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian' or 'Student'
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_PSMMD1 = zeros(3,max_lag,Sim);
param_ISMMD2 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);

param_MLE = zeros(3,Sim);
phi = 0.9; sigma_e = 0.1; sigma_x = 0.2;
param_true = [phi,sigma_e,sigma_x];
dist = 'Gaussian';
% dist = 'Student';

optimoptions.MaxRLPIter = 5000;
optimoptions.MaxFunEvals = 5000;
optimoptions.Algorithm = 'sqp';
optimoptions.TolCon = 1e-09;
optimoptions.TolRLPFun = 1e-09;
optimoptions.MaxSQPIter = 10000;
optimoptions.Diagnostics = 'on';
optimoptions.Jacobian = 'off';
optimoptions.Display = 'iter';

for oo=1:Sim
    T = 1000;
    Y_obs = simulate_sv(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_sv_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_sv_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_sv_psmmd(Y_p,2000,p,sigma);
    end
    
    startcoeff = [0.75+(0.95-0.75)*rand(1),0.01+(0.15-0.01)*rand(1),0.1+(0.15)*rand(1)]';
    [parameters_sv,~,~,~,~,~]=fmincon(@(x)sv_particle_filter_loglik(x,Y_obs),startcoeff,[],[],[],[],[],[],@(x)constr_sv(x),optimoptions);
    param_MLE(:,oo) = parameters_sv;
end

%% \|\|_2-error, apply mean for averaging over the 100 batches
dist_NT = zeros(Sim,max_lag); dist_T = zeros(Sim,max_lag);
dist_NT2 = zeros(Sim,max_lag); dist_T2 = zeros(Sim,max_lag);
param_true = [phi,sigma_e,sigma_x]';
dist_MLE = zeros(Sim,1);
for oo=1:Sim
    for k = 1:max_lag
        dist_NT(oo,k) = norm(param_ISMMD1(:,k,oo)-param_true);
        dist_T(oo,k) = norm(param_PSMMD1(:,k,oo)-param_true);
        dist_NT2(oo,k) = norm(param_ISMMD2(:,k,oo)-param_true);
        dist_T2(oo,k) = norm(param_PSMMD2(:,k,oo)-param_true);
    end
    dist_MLE(oo) = norm(param_MLE(:,oo)-param_true);
end
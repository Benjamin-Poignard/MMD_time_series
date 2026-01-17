%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%% Simulation experiments for GARCH model %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% True DGP:  omega = 0.05, beta = 0.85, alpha = 0.1, T = 300
% Select the distribution of the innovation: dist = 'Student'
addpath(genpath(pwd))
clear
clc
rng(1, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.85; alpha = 0.1;
param_true = [omega,beta,alpha];
dist = 'Student';

for oo= 1:Sim
    % GARCH model simulation
    T = 300;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.85, alpha = 0.1, T = 300
% Select the distribution of the innovation: dist = 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(1, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.85; alpha = 0.1;
param_true = [omega,beta,alpha];
dist = 'Gaussian';

for oo= 1:Sim
    % GARCH model simulation
    T = 300;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.85, alpha = 0.1, T = 1000
% Select the distribution of the innovation: dist = 'Student'
addpath(genpath(pwd))
clear
clc
rng(2, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.85; alpha = 0.1;
param_true = [omega,beta,alpha];
dist = 'Student';

for oo=1:Sim
    % GARCH model simulation
    T = 1000;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.85, alpha = 0.1, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(2, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.85; alpha = 0.1;
param_true = [omega,beta,alpha];
dist = 'Gaussian';

for oo=1:Sim
    % GARCH model simulation
    T = 1000;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.92, alpha = 0.05, T = 300
% Select the distribution of the innovation: dist = 'Student'
addpath(genpath(pwd))
clear
clc
rng(3, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.92; alpha = 0.05;
param_true = [omega,beta,alpha];
dist = 'Student';

for oo=1:Sim
    % GARCH model simulation
    T = 300;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.92, alpha = 0.05, T = 300
% Select the distribution of the innovation: dist = 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(3, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.92; alpha = 0.05;
param_true = [omega,beta,alpha];
dist = 'Gaussian';

for oo=1:Sim
    % GARCH model simulation
    T = 300;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.92, alpha = 0.05, T = 1000
% Select the distribution of the innovation: dist = 'Student'
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.92; alpha = 0.05;
param_true = [omega,beta,alpha];
dist = 'Student';

for oo=1:Sim
    % GARCH model simulation
    T = 1000;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% True DGP:  omega = 0.05, beta = 0.92, alpha = 0.05, T = 1000
% Select the distribution of the innovation: dist = 'Gaussian'
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 16;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.92; alpha = 0.05;
param_true = [omega,beta,alpha];
dist = 'Gaussian';

for oo=1:Sim
    % GARCH model simulation
    T = 1000;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

%% \|\|_2-error, apply mean for averaging over the 100 batches
dist_NT = zeros(Sim,max_lag); dist_T = zeros(Sim,max_lag);
dist_NT2 = zeros(Sim,max_lag); dist_T2 = zeros(Sim,max_lag);
param_true = [omega,beta,alpha]';
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

%% True DGP:  omega = 0.05, beta = 0.92, alpha = 0.05, T = 1000
% Select the distribution of the innovation: dist = 'Student'
addpath(genpath(pwd))
clear
clc
rng(4, 'twister' );
Sim = 100; max_lag = 41;
param_ISMMD1 = zeros(3,max_lag,Sim); param_ISMMD2 = zeros(3,max_lag,Sim);
param_PSMMD1 = zeros(3,max_lag,Sim); param_PSMMD2 = zeros(3,max_lag,Sim);
param_MLE = zeros(3,Sim);

omega = 0.05; beta = 0.92; alpha = 0.05;
param_true = [omega,beta,alpha];
dist = 'Student';

for oo=1:Sim
    % GARCH model simulation
    T = 1000;
    Y_obs = simulate_garch(param_true,T,dist);
    
    p=0;
    Y_p = zeros(T-p,p+1);
    Y_p(:,1) = Y_obs(p+1:end);
    for i=1:p
        Y_p(:,i+1) = Y_obs(p-i+1:end-i);
    end
    sigma = median_heuristic(Y_p);
    param_ISMMD1(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
    param_ISMMD2(:,1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
    param_PSMMD1(:,1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
    param_PSMMD2(:,1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    
    parfor p = 1:max_lag-1
        Y_p = zeros(T-p,p+1);
        Y_p(:,1) = Y_obs(p+1:end);
        for i=1:p
            Y_p(:,i+1) = Y_obs(p-i+1:end-i);
        end
        sigma = median_heuristic(Y_p);
        param_ISMMD1(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,1000,sigma);
        param_ISMMD2(:,p+1,oo) = gradient_garch_ismmd(Y_p,100,p,2000,sigma);
        param_PSMMD1(:,p+1,oo) = gradient_garch_psmmd(Y_p,1000,p,sigma);
        param_PSMMD2(:,p+1,oo) = gradient_garch_psmmd(Y_p,2000,p,sigma);
    end
    
    EstMdl = estimate(garch(1,1),Y_obs);
    param_MLE(:,oo) = [EstMdl.Constant;EstMdl.GARCH{1};EstMdl.ARCH{1}];
end

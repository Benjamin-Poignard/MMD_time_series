function y = simulate_garch(param,T,dist)

% GARCH model DGP

% inputs: - param: parameter of GARCH(1,1)
%         - T: number of time points
%         - dist: type of distribution: 'Gaussian' or 'Student'

% output: GARCH based generated data

a = param(1); b = param(2); c = param(3); gamma = 3;
T = T+100;
y = zeros(T,1); var = zeros(T,1);
y(1) = a/(1-b-c);
for t = 2:T
    var(t) = a + b*var(t-1) + c*y(t-1)^2;
    switch dist
        case 'Gaussian'
            y(t) = mvnrnd(zeros(1),var(t));
        case 'Student'
            y(t) = sqrt(var(t)) * (sqrt(gamma-2)*trnd(gamma)/sqrt(gamma));
    end
end
y = y(101:end);
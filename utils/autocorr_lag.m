function r = autocorr_lag(x,k)

% Summary statistics for SL method

x = x - mean(x);
n = length(x);
if k >= n
    r = 0;
else
    r = sum(x(1:n-k).*x(k+1:n)) / sum(x.^2);
end
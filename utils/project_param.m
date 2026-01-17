function param_proj = project_param(param)
% Apply parameter constraints for GARCH
param(1) = min(max(param(1), eps), 0.1);
param(2) = min(max(param(2),0.6),0.99);
param(3) = min(max(param(3),eps), 0.3);

if param(2) + param(3) > 0.99
    scale = 0.99 / (param(2) + param(3));
    param(2) = param(2) * scale;
    param(3) = param(3) * scale;
end

param_proj = param;
end
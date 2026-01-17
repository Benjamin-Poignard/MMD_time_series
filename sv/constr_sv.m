function [c,ceq] = constr_sv(x)

% SV model parameter constraint

b = x(1); sigma_e = x(2); sigma_y = x(3);
c = [];

c = [ c ; b - 0.95 ];
c = [ c ; 0.7 - b ];

c = [ c ; 0.01 - sigma_e ];
c = [ c ; sigma_e - 0.3 ];

c = [ c ; 0.01 - sigma_y ];
c = [ c ; sigma_y - 0.3 ];
ceq = [];
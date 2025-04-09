clear;
close all;

init_unlocbox();

verbose = 2;

g = 10;
N = 5000;
K = g * 10;
p = 2;
R = max(ceil(4/p), ceil(log(N)/p));
fprintf('The compression ratio is %g\n', N/(R*K));

A = randn(R * K, N);

x = zeros(N, 1);
I2 = randperm(N/g)*g;
I = zeros(size(x));

for i=0:N/g-1
	I(i*g+1:(i+1)*g) = I2(i+1) * ones(g,1) - (1:g)' + 1;
end

x(I(1:K)) = randn(K, 1);
x = x / norm(x);

g_d = 1:N;
g_t = g * ones(1, N/g);

y = A * x;

operatorA = @(x) A * x;
operatorAt = @(x) A' * x;
epsilon2 = 1e-5;
param_proj.epsilon = epsilon2;
param_proj.A = operatorA;
param_proj.At = operatorAt;
param_proj.y = y;
param_proj.tight = 0;
param_proj.nu = norm(A)^2;
param_proj.verbose = verbose - 1;
f2.prox = @(x,T) proj_b2(x, T, param_proj);
f2.eval = @(x) norm(A*x - y)^2;

param_linf1.verbose = verbose - 1;
param_linf1.g_d = g_d;
param_linf1.g_t = g_t;
f1.prox = @(x, T) prox_linf1(x, T, param_linf1);
f1.eval = @(x) norm_linf1(x, g_d, g_t);

param_solver.verbose = verbose;
param_solver.maxit = 300;
param_solver.tol = 1e-4;
param_solver.gamma = 1e-2;

sol = douglas_rachford(zeros(N,1), f1, f2, param_solver);

plot(1:N, x, 'o', 1:N, sol, 'xr')
legend('Original signal', 'Reconstructed signal')

close_unlocbox();

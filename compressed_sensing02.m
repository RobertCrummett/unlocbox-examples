clear;
close all;

init_unlocbox();

verbose = 2;

N = 5000;
K = 100;
R = max(4, ceil(log(N)));
fprintf('The compression ratio is %g\n', N/(R*K));

% Measurement matrix
A = randn(R * K, N);

% Create a K sparse signal
x = zeros(N, 1);
I = randperm(N);
x(I(1:K)) = randn(K,1);
x = x / norm(x);

% Measurements
y = A * x;

operatorA = @(x) A * x;
operatorAt = @(x) A' * x;
epsilon2 = 1e-7;
param_proj.epsilon = epsilon2;
param_proj.A = operatorA;
param_proj.At = operatorAt;
param_proj.y = y;
param_proj.tight = 0;
param_proj.nu = norm(A)^2;
param_proj.verbose = verbose - 1;
f2.prox = @(x,T) proj_b2(x, T, param_proj);
f2.eval = @(x) eps;

param_l1.verbose = verbose - 1;
param_l1.tight = 1;
f1.prox = @(x,T) prox_l1(x, T, param_l1);
f1.eval = @(x) norm(x, 1);

param_solver.verbose = verbose;
param_solver.maxit = 300;
param_solver.tol = 1e-4;
param_solver.gamma = 1e-2;

sol = solvep(zeros(N,1), {f1, f2}, param_solver);

plot(1:N, x, 'o', 1:N, sol, 'xr')
legend('Original signal', 'Reconstructed signal')

close_unlocbox();

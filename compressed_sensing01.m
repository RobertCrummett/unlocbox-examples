clear;
close all;

addpath('unlocbox\')
init_unlocbox();

verbose = 2;

tau = 1;      % regularization parameter

N = 5000;     % size of the signal
K = 100;      % sparsity level
R = max(4, ceil(log(N)));

fprintf('The compression ratio is %g\n', N/(R*K));

% Measurement matrix
A = randn(R * K, N);

% Create sparse signal
x = zeros(N, 1);
I = randperm(N);
x(I(1:K)) = randn(K, 1);
x = x / norm(x);

% Measurements
y = A * x;

% Define proximal operators
f2.grad = @(x) 2*A'*(A*x-y);
f2.eval = @(x) norm(A*x-y)^2;
f2.beta = 2 * norm(A)^2;

param_l1.verbose = verbose - 1;
param_l1.tight = 1;

f1.prox = @(x,T) prox_l1(x, T*tau, param_l1);
f1.eval = @(x) tau*norm(x,1);

% Solve the problem
param_sovler.verbose = verbose;
param_solver.matit = 300;
param_solver.tol = 1e-4;
param_solver.method = 'FISTA';

sol = solvep(zeros(N,1), {f1, f2}, param_solver);

plot(1:N, x, 'o', 1:N, sol, 'xr')
legend('Original signal', 'Reconstucted signal')

close_unlocbox();
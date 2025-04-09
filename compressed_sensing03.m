clear;
close all;

init_unlocbox();

verbose = 2;

g = 10;		% number of elements per group
N = 5000;	% signal size
K = g * 10;	% sparsity level
p = 4;		% gain with respect to the traditional compression ratio
R = max(ceil(4/p),ceil(log(N)/p));
fprintf('The compression ratio is %g\n', N/(R*K));

% Measurements
A = randn(R * K, N);

% Create K sparse signal
x = zeros(N, 1);
I2 = randperm(N/g)*g;
I = zeros(size(x));

for i=0:N/g-1
	I(i*g+1:(i+1)*g) = I2(i+1) * ones(g,1) - (1:g)' + 1;
end

x(I(1:K)) = randn(K, 1);
x = x / norm(x);

% Create groups
g_d = 1:N;
g_t = g * ones(1, N/g);

% Measurements
y = A * x;

% Define the proximal operator of f2
operatorA = @(x) A * x;
operatorAt = @(x) A' * x;
epsilon2 = 1e-5;
param_proj.epsilon = epsilon2;
param_proj.A = operatorA;
param_proj.At = operatorAt;
param_proj.y = y;
param_proj.tight = 1;
param_proj.nu = norm(A)^2;
param_proj.verbose = verbose - 1;
f2.prox = @(x, T) proj_b2(x, T, param_proj);
f2.eval = @(x) norm(A*x - y)^2;

% Define the function f1
param_l21.verbose = verbose - 1;
param_l21.g_d = g_d;
param_l21.g_t = g_t;
f1.prox = @(x, T) prox_l21(x, T, param_l21);
f1.eval = @(x) norm_l21(x, g_d, g_t);

param_solver.verbose = verbose;
param_solver.maxit = 300;
param_solver.tol = 1e-4;
param_solver.gamma = 1e-2;

sol = solvep(zeros(N,1), {f1, f2}, param_solver);

figure;
plot(1:N, x, 'o', 1:N, sol, 'xr');
legend('Original signal', 'Reconstructed signal');

close_unlocbox();

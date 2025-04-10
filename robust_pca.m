clear;
close all;

addpath('unlocbox/');
init_unlocbox();

verbose = 2;

N = 100;		% column number
M = 200;		% row number
k = 10;			% rank
p = 0.1;		% probability for the sparse noise
sigma = 10;		% noise level
tau = 10;		% regularization parameter

% Create the data
X = randn(100, k);
Y = rand(k,M);

Z = X * Y;		% low rank data
N = sigma * sprand(N,M,p);
Znoisy = Z + N;		% measurements

paraml1.verbose = verbose - 1;
paraml1.y = Znoisy;
f_f.prox = @(x,T) prox_l1(x,T,paraml1);
f_f.eval = @(x) sum(abs(x(:)));

% 2) nuclear norm f_n (X) = tau || X ||_*
paramnuclear.verbose = verbose - 1;
f_n.prox = @(x,T) prox_nuclearnorm(x,T*tau,paramnuclear);
f_n.eval = @(x) tau*norm_nuclear(x);

% Solve the problem
paramsolver.verbose = verbose;
paramsolver.gamma = 0.5;

Zsol = solvep(Znoisy, {f_f, f_n}, paramsolver);
N = Znoisy - Zsol;

% Errors and plots
err_sol = norm(Zsol-Z, 'fro')/norm(Z,'fro');
err_in = norm(Znoisy-Z, 'fro')/norm(Z,'fro');

figure(1)
subplot(221)
imagesc(Z)
title('Original low rank data')
subplot(222)
imagesc(Znoisy)
title('Noisy data')
subplot(223)
imagesc(Zsol)
title('Recovery')
subplot(224)
imagesc(N)
title('Sparse noise')

close_unlocbox();

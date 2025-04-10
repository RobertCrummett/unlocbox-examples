clear;
close all;

addpath('unlocbox/', 'rwt/bin');
init_unlocbox();

verbose = 2;

im_original = barbara();

sigma = 0.9;
b = im_original + sigma^2*rand(size(im_original));

tau1 = 0.2;
tau2 = 0.2;

% TV norm
param_tv.verbose = verbose - 1;
param_tv.maxit = 100;

g_tv.prox = @(x,T) prox_tv(x, T*tau1, param_tv);
g_tv.eval=  @(x) tau1 * norm_tv(x);

g_tv.x0 = b;
g_tv.L = @(x) x;
g_tv.Lt = @(x) x;

% Wavelets
L = 8;
A2 = @(x) mdwt(x, 'db1', L);
A2t = @(x) midwt(x, 'db1', L);

param_l1.verbose = verbose - 1;
g_l1.prox = @(x, T) prox_l1(x, T*tau2, param_l1);
g_l1.eval = @(x) tau2 * norm(reshape(A2(x), [], 1), 1);

g_l1.L = A2;
g_l1.Lt = A2t;
g_l1.x0 = b;

% L2 Norm
paraml2.verbose = verbose - 1;
paraml2.y = b;
g_l2.prox = @(x,T) prox_l2(x, T, paraml2);
g_l2.eval = @(x) norm(x, 2);
g_l2.x0 = b;
g_l2.L = @(x) x;
g_l2.Lt = @(x) x;

% Solving the problem
F = {g_l1, g_tv, g_l2};
param_solver.Qinv = @(x) 1/3*x;
param_solver.maxit = 30;
param_solver.verbose = verbose;

sol = sdmm(F, param_solver);

imagesc_gray(im_original, 1, 'Original image');
imagesc_gray(b, 2, 'Depleted image');
imagesc_gray(sol, 3, 'Reconstructed image');

close_unlocbox();
clear;
close all;

addpath('unlocbox\')
init_unlocbox();

verbose = 2;

im_original = barbara();

A = rand(size(im_original));
A = A > 0.05;

b = A .* im_original;

operatorA = @(x) A .* x;
epsilon2 = 0;
param_proj.epsilon = epsilon2;
param_proj.A = operatorA;
param_proj.At = operatorA;
param_proj.y = b;
param_proj.verbose = verbose - 1;
f2.prox = @(x,T) proj_b2(x, T, param_proj);
f2.eval = @(x) eps;

f2.prox = @(x,T) (x - A.*x) + A.*b;

param_tv.verbose = verbose - 1;
param_tv.maxit = 50;

f1.prox = @(x,T) prox_tv(x, T, param_tv);
f1.eval = @(x) norm_tv(x);

paramsolver.verbose = verbose;
paramsolver.maxit = 100;
paramsolver.tol = 1e-6;
paramsolver.gamma = 0.1;

fig = figure(100);
paramsolver.do_sol = @(x) plot_image(x, fig);

sol = douglas_rachford(b, f1, f2, paramsolver);

close(100);

imagesc_gray(im_original, 1, 'Original image');
imagesc_gray(b, 2, 'Depleted image');
imagesc_gray(sol, 3, 'Reconstructed image');

close_unlocbox();
% GPU crashes my system.
% So this actually just uses the CPU.
clear;
close all;

addpath("unlocbox\")
init_unlocbox()

verbose = 1;

% Original image
im_original = barbara();

% Create the problem
M = rand(size(im_original)) < 0.3;
A = @(x) M .* x;
At = A;

% Depleted image
b = A(im_original);

tau = 0.1;

param_tv.verbose = verbose - 1;
param_tv.maxit = 50;
param_tv.tol = 1e-5;

f.prox = @(x,T) prox_tv(x, T * tau, param_tv);
f.eval = @(x) tau * norm_tv(x);

% set parameters for the simulation
param_solver.verbose = verbose;
param_solver.maxit = 20;
param_solver.tol = 1e-5;
param_solver.nu = 1;

[sol, infos] = rlr(b, f, A, At, param_solver);
fprintf('Computation time with the CPU: %g\n', infos.time);

imagesc_gray(im_original, 1, 'Original image');
imagesc_gray(sol, 4, ...
    strcat('Reconstructed image with the CPU -- time:',num2str(infos.time)));

close_unlocbox()
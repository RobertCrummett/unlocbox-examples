clear;
close all;

addpath('unlocbox/')
init_unlocbox();

verbose = 2;

% Infill problem

input_snr = 30;     % noise level
im = cameraman();
imagesc_gray(im, 1, 'Original image');

% Mask
mask = rand(size(im)) < 0.33;
ind = find(mask == 1);
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(im));

% Masking operator
A = @(x) Ma * x(:);                         % selects 33% of the values in x
At = @(x) reshape(Ma'*x(:), size(im));      % adjoint operator + reshape image

% Inpainting
y = A(im);
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + randn(size(y))* sigma_noise;

imagesc_gray(At(y), 2, 'Measured image');

% Parameter
param.verbose = 2;
param.gamma = 0.1;      % step size
param.tol = 1e-4;       % stopping criteria for the tvdn problem
param.maxit = 200;      % max number of iterations for the tvdn solution
param.maxit_tv = 100;   % max number of iterations for the proximal TV operator
param.nu_b2 = 1;        % bound on the norm of operator A
param.tol_b2 = 1e-4;    % toleratnce for the projection onto the L2 ball
param.tight_b2 = 0;     % indicate if A is a tight frame (1) or not (0)
param.maxit_b2 = 500;

epsilon = sqrt(chi2inv(0.99, numel(ind))) * sigma_noise; % tolerance on noise
sol = solve_tvdn(y, epsilon, A, At, param);

imagesc_gray(sol, 3, 'Reconstructed image');

% Reconstruct from 33% Fourier measurements
A = @(x) Ma*reshape(fft2(x)/sqrt(numel(im)), numel(x), 1);
At = @(x) ifft2(reshape(Ma'*x(:), size(im))*sqrt(numel(im)));

y = A(im);
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y))) * sigma_noise/sqrt(2);

imagesc_gray(real(At(y)), 4, 'Measured image', 121);

epsilon = sqrt(chi2inv(0.99, 2*numel(ind))/2)*sigma_noise;

sol = solve_tvdn(y, epsilon, A, At, param);

imagesc_gray(real(sol), 4, 'Reconstructed image', 122);

close_unlocbox();
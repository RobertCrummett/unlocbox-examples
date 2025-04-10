clear;
close all;

addpath('unlocbox/');
init_unlocbox();

N = 64;
input_snr = 30;     % noise level (on the measurements)

% Create image with some spikes
im = zeros(N);
ind = randperm(N^2);
im(ind(1:100)) = 1;

figure(1);
subplot(221)
imagesc(im);
axis image
axis off
colormap gray
title('Original image')
drawnow;

% Create mask
mask = rand(size(im)) < 0.095;
ind = find(mask == 1);
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(im));

% Reconstruct from a few Fourier coefficients
A = @(x) Ma * reshape(fft2(x)/sqrt(numel(im)), numel(x), 1);
At = @(x) ifft2(reshape(Ma'*x(:), size(im)) *sqrt(numel(im)));

Psit = @(x) x;
Psi = Psit;

y = A(im);

% Add Gaussian iid noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y))) * sigma_noise / sqrt(2);

% Display the downsampled image
figure(1);
subplot(222)
imagesc(real(At(y)));
axis image
axis off
colormap gray
title('Measured image');
drawnow;

% Tolerance on noise
epsilon = sqrt(chi2inv(0.99, 2*numel(ind))/2)*sigma_noise;

param.verbose = 1; % Print log or not
param.gamma = 1e-1; % Converge parameter
param.tol = 1e-4; % Stopping criterion for the BPDN problem
param.maxit = 300; % Max. number of iterations for the BPDN problem 
param.nu_b2 = 1; % Bound on the norm of the operator A
param.tol_b2 = 1e-4; % Tolerance for the projection onto the L2-ball
param.tight_b2 = 1; % Indicate if A is a tight frame (1) or not (0)
param.tight_l1 = 1; % Indicate if Psit is a tight frame (1) or not (0)
param.pos_l1 = 1;

sol = solve_bpdn(y, epsilon, A, At, Psi, Psit, param);

figure(1);
subplot(223);
imagesc(real(sol));
axis image
axis off
colormap gray
title(['First estimate - ' num2str(snr(im, real(sol))) 'dB']);
drawnow;

param.weights = 1./(abs(sol)+1e-5);
sol = solve_bpdn(y, epsilon, A, At, Psi, Psit, param);

figure(1);
subplot(224);
imagesc(real(sol));
axis image
axis off
colormap gray
title(['Second estimate - ' num2str(snr(im, real(sol))) 'dB']);
drawnow;

close_unlocbox();
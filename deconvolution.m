clear;
close all;

init_unlocbox();

verbose = 2;
clim = [0 1];

im_original = cameraman();

sigma = 0.1;
[x, y] = meshgrid(linspace(-1, 1, length(im_original)));
r = x.^2 + y.^2;
G = exp(-r/(2*sigma^2));

A = @(x) real(ifft2(fftshift(G).*(fft2(x))));
At = @(x) real(ifft2(conj(fftshift(G)).*(fft2(x))));

b = A(im_original);

tau = 0.0003;

L = 8;
W = @(x) fwt2(x, 'db1', L);
Wt = @(x) ifwt2(x, 'db1', L);

param_l1.verbose = verbose - 1;
param_l1.tight = 1;
param_l1.At = Wt;
param_l1.A = W;

f.prox = @(x, T) prox_l1(x, T*tau, param_l1);
f.eval = @(x) tau * sum(sum(abs(W(x))));

param_proj.maxit = 10;
param_proj.epsilon = 0;
param_proj.tight = 0;
param_proj.nu = 2;
param_proj.A = A;
param_proj.At = At;
param_proj.y = b;
param_proj.verbose = verbose - 1;
f2.eval = @(x) norm(A(x) - b).^2;
f2.prox = @(x,T) proj_b2(x, T, param_proj);

param_solver.verbose = verbose;
param_solver.maxit = 300;
param_solver.tol = 1e-8;

fig = figure(100);
param_solver.do_sol = @(x) plot_image(x, fig);

sol = solvep(b, {f, f2}, param_solver);

close(fig);

imagesc_gray(im_original, 1, 'Original image', 111, clim)
imagesc_gray(b, 2, 'Depleted image', 111, clim)
imagesc_gray(sol, 3, 'Reconstructed image', 111, clim)

close_unlocbox();
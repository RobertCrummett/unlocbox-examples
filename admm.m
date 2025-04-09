clear;
close all;

addpath("unlocbox\", "ltfat\");
init_unlocbox();
ltfatstart();

verbose = 2;

tau = 50;
sigma = 0.1;
p = 50;

% Original
im_original = barbara();

% Depleted image
mask = rand(size(im_original)) > p / 100;
z = mask .* im_original + sigma * rand(size(im_original));

% Defining the wavelet operator
L = @(x) fwt2(x, 'db8', 6);
Lt = @(x) ifwt2(x, 'db8', 6);

f1.proxL = @(x, T) (1+tau*T*mask).^(-1) .* (Lt(x)+tau*T*mask.*z);
f1.eval = @(x) tau * norm(mask .* x - z)^2;

param_l1.verbose = verbose - 1;
f2.prox = @(x, T) prox_l1(x, T, param_l1);
f2.eval = @(x) norm(L(x), 1);
f2.L = L;
f2.Lt = Lt;
f2.norm_L = 1;

paramsolver.verbose = verbose;
paramsolver.maxit = 100;
paramsolver.tol = 1e-3;
paramsolver.gamma = 1;
paramsolver.debug_mode = 1;

fig = figure(100);
paramsolver.do_sol=@(x) plot_image(x, fig);

sol = admm(z, f1, f2, paramsolver);

imagesc_gray(im_original, 1, 'Original image');
imagesc_gray(z, 2, 'Depleted image');
imagesc_gray(sol, 3, 'Reconstructed image');

close_unlocbox();
clear;
close all;

addpath('unlocbox\', 'ltfat\');
init_unlocbox();
ltfatstart;

sigma = 0.1;
missing_ratio = 0.4;

lambda = 0.05;
tau = 0.1;
verbose = 2;

img = barbara();

[nx, ny] = size(img);
A = rand(nx, ny) > missing_ratio;
noisy_img = img + sigma * randn(nx, ny);
b = A .* noisy_img;

ffid.grad = @(x) 2 * A .* (A.*x - b);
ffid.eval = @(x) norm(A(:).*x(:)-b(:))^2;
ffid.beta = 2;

param_tv.verbose = verbose - 1;
param_tv.maxit = 50;

ftv.prox = @(x, T) prox_tv(x, T*lambda, param_tv);
ftv.eval = @(x) lambda * norm_tv(x);

Nlevel = 5;
W = @(x) fwt2(x, 'db8', Nlevel);
Wt = @(x) ifwt2(x, 'db8', Nlevel);
paraml1.verbose = verbose - 1;
fw.prox = @(x,T) prox_l1(x, tau*T,paraml1);
fw.eval = @(x) tau*norm(W(x),1);
fw.L = W;
fw.Lt = Wt;
fw.norm_L = 1;

param_solver.verbose = verbose;
param_sovler.maxit = 30;
param_solver.tol = 1e-6;
param_solver.debug_mode = 1;

fig = figure(100);
param_solver.do_sol = @(x) plot_image(x, fig);

sol = fb_based_primal_dual(b,ffid,ftv,fw,param_solver);
close(100);

imagesc_gray(img, 1, '(a) Original image',221,[0 1]);
imagesc_gray(noisy_img, 1, '(b) Noisy image',222,[0 1]);
imagesc_gray(b, 1, '(c) Measurements',223,[0 1]);
imagesc_gray(sol, 1, '(d) Solution of optimization',224,[0 1]); 

close_unlocbox();
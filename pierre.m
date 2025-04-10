clear;
close all;

addpath('unlocbox/');
init_unlocbox();

verbose = 1;

mu1 = 1e-3;
mu2 = 1e-3;

N = 1024;
K = 50;

M = rand(N, 1);
M = M > 0.5;
M = M*2-1;

% Create operator Psi
Psi = @(x) 1/sqrt(N)*fft(x);
Psit = @(x) sqrt(N)*ifft(x);
nu_Psi = 1;

% Create operator Phi
Phi = @(x) M.*Psi(x);
Phit = @(x) Psit(M.*x);
nu_Phi = 1;

% Create a K sparse signal
sigb = zeros(N, 1);
I = randperm(N);
sigb(I(1:K)) = randn(K, 1);
sigb = sigb/norm(sigb);

% Create another K sparse signal
sigc = zeros(N, 1);
I = randperm(N);
sigc(I(1:K)) = randn(K, 1);
sigc = sigc/norm(sigc);

% Creation of the measurements
s = Psi(sigc) + Phi(sigb);

% Define mask functions
Mc = @(x) x(1:N);
Mb = @(x) x(N+1:end);
Cc = @(x,c) [c; Mb(x)];
Cb = @(x,b) [Mc(x); b];

% Functional mu1*||c||_1
param_l1_1.verbose = verbose - 1;
f1.prox = @(x, T) Cc(x, prox_l1(Mc(x), T*mu1, param_l1_1));
f1.eval = @(x)    mu1*norm(Mc(x),1);

% Functional mu2*||c||_1
param_l1_2.verbose = verbose - 1;
f2.prox = @(x, T) Cb(x, prox_l1(Mb(x), T*mu2, param_l1_2));
f2.eval = @(x)    mu2*norm(Mb(x),1);

% Define the gradient of ||s - Psi c - Phi b||_2^2
g.grad = @(x) [Psit( Psi(Mc(x)) + Phi(Mb(x)) - s ) ; ...
               Phit( Psi(Mc(x)) + Phi(Mb(x)) - s ) ];
g.eval = @(x)  norm(s-(Psi(Mc(x))+Phi(Mb(x))))^2;
g.beta = (nu_Phi + nu_Psi);

param_solver.maxit = 1000;
param_solver.verbose = verbose;

x0 = eps*ones(2*N,1);
sol = solvep(x0, {f1,f2,g}, param_solver);

figure(1)
subplot(211)
t=1:N;
plot(t,abs(sigb),'xb',t,abs(Mb(sol)),'or');
legend('Original b', 'Recovered b');
subplot(212)
plot(t,abs(sigc),'xb',t,abs(Mc(sol)),'or');
legend('Original c', 'Recovered c');

close_unlocbox();

clear;
close all;

addpath('unlocbox\', 'ltfat\')
init_unlocbox();
ltfatstart();

verbose = 2;
writefile = 0;

[sound_original, Fs] = gspi();

length_sig = length(sound_original);
sound_part = sound_original(1:length_sig);

if writefile
    wavesave(sound_part, Fs, 'original.wav');
end

tmax = 0.08;
tmin = -0.3;
Mask = 1-(sound_part>tmax) - (sound_part<tmin);

sound_depleted = sound_part;
sound_depleted(sound_part>tmax) = tmax;
sound_depleted(sound_part<tmin) = tmin;
if writefile
    wavsave(sound_depleted,Fs,'depleted.wav');
end

tau = 1e-2;
a = 64;
M = 256;
F = frametight(frame('dgtreal','gauss',a,M));

GB = M/a;

Psi = @(x) frana(F,x);
Psit = @(x) frsyn(F,x);

f2.prox = @(x,T) Psi(Psit(x) .* (1 - Mask) + Mask.*sound_depleted);
f2.eval = @(x) eps;

% setting the function f1 (l1 norm of the Gabor transform)
% set parameters
param_l21.verbose = verbose - 1;

% The groups are made like this
%
%   For a 2 by 8 spectrogram 
%   11112222        21111222        22111122        22211112       
%   33334444        43333444        44333344        44433334

% Create the group (improved code)
% -------------------------------------------- %
lg = 4; % length of the group;

xin = Psi(sound_depleted);
xin_im = framecoef2native(F, xin);

[K, L] = size(xin_im);  % More robust way to get dimensions

% Initialize group indices matrix
total_elements = K * L;
indice = 1:total_elements;
indice_mat = reshape(indice, L, K).';  % Transpose to match original K rows, L cols

% Calculate number of complete groups we can make
num_complete_groups_per_row = floor(L / lg);
sgd = num_complete_groups_per_row * lg * K;

% Initialize group matrix
g_d = zeros(lg, sgd);

for ii = 1:lg
    for jj = 1:num_complete_groups_per_row
        for ll = 1:K
            % Calculate column indices with circular shift
            cols = mod((0:lg-1) + (jj-1)*lg + (ii-1), L) + 1;
            
            % Get the linear indices for this group
            group_indices = indice_mat(ll, cols);
            
            % Store in the output matrix
            start_col = (jj-1)*lg + (ll-1)*num_complete_groups_per_row*lg + 1;
            end_col = start_col + lg - 1;
            g_d(ii, start_col:end_col) = group_indices;
        end
    end
end

g_t = lg * ones(lg, sgd/lg);  % Group weights
% -------------------------------------------- %

param_l21.g_t = g_t;
param_l21.g_d = g_d;
param_121.maxit = 5;

f1.prox = @(x,T) prox_l21(x, T*tau, param_l21);
f1.eval = @(x) tau*norm_l21(x, g_d, g_t);

param.verbose = verbose;
param.maxit = 100;
param.tol = 1e-5;
param.method = 'FISTA';

param.to_ts = @(x) log_decreasing_ts(x, 10, 0.1, 80);
sol = Psit(solvep(Psi(sound_part), {f1, f2}, param));

snr_in = snr(sound_part,sound_depleted);
snr_fin = snr(sound_part,sol);

fprintf('The SNR of the initial signal is %g dB \n',snr_in);
fprintf('The SNR of the recovered (FB) signal is %g dB \n',snr_fin);

if writefile
    wavsave(sol, Fs, 'restored.wav');
end
dr=90;

figure(1);
plotframe(F,Psi(sound_part),Fs,dr);
title('Gabor transform of the original sound');

figure(2);
plotframe(F,Psi(sound_depleted),Fs,dr);
title('Gabor transform of the depleted sound');

figure(3);
plotframe(F,Psi(sol),Fs,dr);
title('Gabor transform of the reconstructed sound');

close_unlocbox();
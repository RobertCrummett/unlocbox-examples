clear;
close all;
addpath('unlocbox\')

N = 64;
k = 5;
d = 10;

algorithm = 'DR';

A = @(x) idct(x);
At = @(x) dct(x);

if k>N
    error('Sparsity cannot be greater than signal length/number of atoms')
end

support = randperm(N);
support = support(1:k);

x_support = 1 + 3*rand([k 1]);
x_support = x_support .* (((rand([k 1])>.5)*2)-1); % random signs
x = zeros(N, 1);
x(support) = x_support;

y = A(x);

fig_coef = figure;
h_coef_orig = bar(x);

hold on;
title('Original coefficients of sparse signal')

min_y = min(min(y));
max_y = max(max(y));
range = max_y - min_y;

quant_step = range / (d-1);
dec_bounds = (min_y + quant_step/2) : quant_step : (max_y-quant_step/2);
quant_levels = min_y : quant_step : max_y;

[index, y_quant] = quantiz(y, dec_bounds, quant_levels);
y_quant = y_quant';

lower_dec_bounds = y_quant - (quant_step/2);
upper_dec_bounds = y_quant + (quant_step/2);

min_quant_level = quant_levels(1);
max_quant_level = quant_levels(end);
upper_dec_bounds(upper_dec_bounds > max_quant_level) = max_quant_level;
lower_dec_bounds(lower_dec_bounds < min_quant_level) = min_quant_level;

grey = 0.6 * ones(1,3);
lightgrey = 0.8 * ones(1,3);
black = [0,0,0];
blue = [0.251, 0.584, 0.808];
orange = [0.851, 0.325, 0.098];
green = [0 1 0];

fig_time = figure;
h_orig = plot(y, '.-', 'Color', blue);
hold on;
h_quant = plot(y_quant, '.-', 'Color', orange);
title('Original and quantized signal')

h_signal_constr = plot(upper_dec_bounds, 'Color', lightgrey);
plot(lower_dec_bounds, 'Color', lightgrey);

for j=1:d
    yPos = quant_levels(j);
    h_q_lev = plot(ones(1,N) * yPos, 'Color', grey);
end

for j=1:(d-1)
    yPos = dec_bounds(j);
    h_dec_b = plot(ones(1,N) * yPos, ':', 'Color', grey);
end

axis tight;

uistack(h_orig, 'top');
uistack(h_quant, 'top');
h_legend_time = legend([h_orig, h_quant, h_q_lev, h_dec_b, h_signal_constr], 'original', 'quantized', 'quantiz. levels', 'decision bounds', 'signal constraints');

fig_quant_noise = figure;
quant_noise = y - y_quant;
h_quant_noise = plot(quant_noise, 'Color', blue);
hold on;
title('Quantization noise');

switch algorithm
    case 'LP' % ie, Linear Programming. doubles the number of variables
        f = (ones([1 2*N]));

        b = [-lower_dec_bounds; upper_dec_bounds];
        Amatrix = A(eye(N));
        A_ = [Amatrix -Amatrix; -Amatrix Amatrix];
        lb = zeros(2*N, 1);

        w = linprog(f,A_,b,[],[],lb);

        uv = reshape(w,N,2);
        u = uv(:,1);
        v = uv(:,2);
        x_reconstructed = v - u;
        y_dequant = A(x_reconstructed);

        sol = x_reconstructed;
    case 'DR' % Douglas-Rachford

        % UNLocBoX version
        init_unlocbox();

        param.lower_lim = lower_dec_bounds;
        param.upper_lim = upper_dec_bounds;

        indi_thr = 1e-4;
        f1.eval = @(x) norm(x,1);
        f1.prox = @(x,T) sign(x).*max(abs(x)-T, 0);

        f2.eval = @(x) 1 / (1 - (any((A(x(:))-param.upper_lim)>indi_thr)) || any((A(x(:))-param.lower_lim) < -indi_thr)) - 1;
        f2.prox = @(x,T) At(proj_box(A(x),[],param));

        paramsolver.verbose = 5;
        paramsolver.maxit = 300;
        paramsolver.tol = 1e-6;
        paramsolver.lambda = 1;
        paramsolver.gamma = 1e-2;

        [sol, info] = douglas_rachford(At(y_quant), f1, f2, paramsolver);
        info

        sol = f2.prox(sol,[]);
        y_dequant = A(sol);
        
        close_unlocbox();

        % Manual version
        DR_y = A(y_quant);
        DR_x_old = DR_y;

        relat_change_coefs = 1;
        relat_change_obj = 1;
        cnt = 1;
        obj_eval = [];

        while relat_change_obj > paramsolver.tol
            DR_x = f2.prox(DR_y,[]);
            obj_eval = [obj_eval, f1.eval(DR_x) + f2.eval(DR_x)];
            DR_y = DR_y + paramsolver.lambda*(f1.prox(2*DR_x-DR_y, paramsolver.gamma)-DR_x);
            if cnt > 1
                relat_change_coefs = norm(DR_x-DR_x_old) / norm(DR_x_old);
                relat_change_obj = norm(obj_eval(end) - obj_eval(end-1)) / norm(obj_eval(end-1));
                if paramsolver.verbose > 1
                    fprintf('  relative change in coefficients: %e \n', relat_change_coefs);
                    fprintf('  relative change in objective fun: %e \n', relat_change_obj);
                    fprintf('\n');
                end
            end
            DR_x_old = DR_x;
            cnt = cnt + 1;
        end

        DR_x = f2.prox(DR_y);
        y_dequant = A(DR_x);

        disp(['Finished after ' num2str(cnt) ' iterations.'])

        figure;
        plot([y_dequant A(sol)])
        norm(y_dequant - A(sol))
        title('UNLOCBOX vs. manual solution')

        figure
        plot(obj_eval)
        title('Objective function value (after projeciton into contraints in each iteration)')
end

figure(fig_time)
h_dequant = plot(y_dequant, '.-', 'Color', green);
uistack(h_dequant,'top');
delete(h_legend_time)
h_legend_time = legend([h_orig, h_quant, h_dequant, h_q_lev, h_dec_b, h_signal_constr ], 'original', 'quantized', 'dequantized', 'quantiz. levels', 'decision bounds', 'signal constraints');
title('Original, quantized and dequantized signals')

%quantization and reconstruction errors
figure(fig_quant_noise)
h_dequant_error = plot(y-y_dequant, 'Color', green);
title('Quantization error and error of reconstruction (i.e. original - reconstr.)');
axis tight
legend([h_quant_noise h_dequant_error], 'Quantizat. error', 'Error of reconstr.')

%coefficients of reconstructed signal
figure(fig_coef)
hold on
h_coefs_dequant = bar(sol,'FaceColor',green);
title('Coefficients of original and reconstructed signals');
legend([h_coefs_orig h_coefs_dequant], 'Coefs of orig. signal', 'Coefs of dequant. signal')
axis tight


figure(fig_time)
axis tight
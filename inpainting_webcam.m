clear;
close all;

addpath('unlocbox\')
init_unlocbox();

scale = 0.5;
imgcolor = 1;
p = 0.8;

verbose = 2;
maxit = 30;

clear('cam');
camList = webcamlist;

if numel(camList) > 1
    fprintf('Please choose a webcam:\n');
    for ii = 1:numel(camList)
        fprintf([' ',num2str(ii),') ',camList{ii},'\n'])
    end
    fprintf('Camera number: ');
    prompt = 1;
    numCam = str2num(input('','s'));
else
    numCam = 1;
end

cam = webcam(numCam);

preview(cam);

fprintf('Push a buttom to acquire image...')
pause;

rgbImage = snapshot(cam);
fprintf('  Done!\n');

rgbImage = imresize(rgbImage, scale);
if imgcolor
    im_original = double(rgbImage)/256;
else
    grayImage = double(rgb2gray(rgbImage))/256;
    im_original = grayImage;
end

closePreview(cam);
clear('cam');

A = rand(size(im_original,1),size(im_original,2));
A = A > p;

if imgcolor
    A = cat(3,A,A,A);
end

b = A.*im_original;

imagesc_gray(im_original, 1, 'Original image')
imagesc_gray(b, 2, 'Depleted image')

operatorA = @(x) A.*x;
operatorAt = @(x) A.*x;
epsilon2 = 0;
param_proj.epsilon = epsilon2;
param_proj.A = operatorA;
param_proj.At = operatorAt;
param_proj.y = b;
param_proj.verbose = verbose - 1;
f2.prox = @(x,T) proj_b2(x, T, param_proj);
f2.eval = @(x) eps;

f2.prox = @(x,T) (x - A.*x) + A.*b;

param_tv.verbose = verbose - 1;
param_tv.maxit = maxit;

f1.prox = @(x, T) prox_tv(x, T, param_tv);
f1.eval = @(x) sum(norm_tv(x));

paramsolver.verbose = verbose;
paramsolver.maxit = maxit;
paramsolver.tol = 1e-6;
paramsolver.gamma = 0.1;

fig = figure(100);
paramsolver.do_sol = @(x) plot_image(x, fig, 1);

sol = douglas_rachford(b, f1, f2, paramsolver);

close(100);

sol(sol<0) = 0;
sol(sol>1) = 1;
imagesc_gray(im_original, 1, 'Original image');
imagesc_gray(b, 2, 'Depleted image');
imagesc_gray(sol, 3, 'Reconstructed image');

close_unlocbox();
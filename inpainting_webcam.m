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

close_unlocbox();
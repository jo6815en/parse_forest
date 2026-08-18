im = imread('testim.jpg');

% manually set R in this case to something reasonable ...
% 17 rotation around x-direction + some random small rot to test function

R =[0.9629   -0.2559   -0.0861;...
    0.2603    0.9645    0.0445;...
    0.0716   -0.0652    0.9953];

f = 20000;
[imout,H] = rectification_image_from_rot_f(im,R,f);

imshow([im imout]);

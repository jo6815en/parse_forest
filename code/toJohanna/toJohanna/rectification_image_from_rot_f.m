function [imout,H] = rectification_image_from_rot_f(im,R,f)

[M,N,~] = size(im); 

K = [f 0 N/2;0 f M/2;0 0 1]; % calibration matrix, assume principal point in middle of image

r2 = R(:,2); 
v2 = null(r2'); % two orthogonal vectors in plane perpendicular to r2
rr3 = v2*([0 0 1]*v2)'; % project the direction [0 0 1] on this plane (is this ok?)
rr1 = cross(r2,rr3); % find the first row of R0 from r2 and rr3
R0 = [rr1 r2 rr3]';
H = K*R0/K; % apply calibration to form homography 3x3
tform = projtform2d(H); % make matlab tranformation

xc = [N/2;M/2;1]; % center of original image 
xcp = H*xc; % projection of center in new image
xlim = [xcp(1)-N/2 xcp(1)+N/2]; % make output image the same size as input with center projected to center
ylim = [xcp(2)-M/2 xcp(2)+M/2];
outview = imref2d([M N],xlim,ylim); % make matlab coordinate output system 

imout = imwarp(im,tform,OutputView = outview); % warp image with transform with correct output size (ok?)


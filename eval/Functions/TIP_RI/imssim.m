% imssim evaluates the ssim between images
%
% ssim = imssim(X, Y, peak, b)
%
% Output parameters:
%  ssim: ssim between images X and Y (per channel)
%
% Input parameters:
%  X: image whose dimensions should be same to that of Y
%  Y: image whose dimensions should be same to that of X
%  peak (optional): peak value (default: 255)
%  b (optional): border size to be neglected for evaluation
%
% Example:
%  X = imread('X.png');
% Y = imread('Y.png');
% ssim = imssim(X, Y);
% fprintf('%g\n', ssim);
%
% Version: 202501
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Miscellaneous tools for image processing                 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ssim_val = imssim(X, Y, peak, b)

if nargin < 3
    peak = 255;
end

if nargin < 4
    b = 0;
end

if b > 0
    X = X(b + 1 : end - b, b + 1 : end - b, :);
    Y = Y(b + 1 : end - b, b + 1 : end - b, :);
end

X = double(X);
Y = double(Y);

% Constants for SSIM
C1 = (0.01 * peak)^2;
C2 = (0.03 * peak)^2;

% Gaussian window (11x11, sigma = 1.5)
h = fspecial('gaussian', 11, 1.5);

for i = 1 : size(X, 3)
    x = X(:, :, i);
    y = Y(:, :, i);
    
    mu_x = imfilter(x, h, 'replicate');
    mu_y = imfilter(y, h, 'replicate');
    
    mu_x2 = mu_x.*mu_x;
    mu_y2 = mu_y.*mu_y;
    mu_xy = mu_x.*mu_y;
    
    sigma_x2 = imfilter(x.*x, h, 'replicate') - mu_x2;
    sigma_y2 = imfilter(y.*y, h, 'replicate') - mu_y2;
    sigma_xy = imfilter(x.*y, h, 'replicate') - mu_xy;
    
    num = (2 * mu_xy + C1).*(2 * sigma_xy + C2);
    den = (mu_x2 + mu_y2 + C1).*(sigma_x2 + sigma_y2 + C2);
    
    ssim_map = num./ den;
    ssim_val(i) = mean(ssim_map(:));
end

end

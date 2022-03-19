clear;
clc;

% Nominal calibrated camera model parameters from Fig. 3
fx = 2359.40946;
fy = 2359.61091;
cx = 1370.05852;
cy = 1059.63818;
k1 = -0.06652;
k2 = +0.06534;
k3 = -0.07555;
p1 = +0.00065;
p2 = -0.00419;

% Estimated standard deviations
% Multiply these by 1.96 to get the half-width of the 95% confidence interval
std_fx = 0.84200;
std_fy = 0.76171;
std_cx = 1.25225;
std_cy = 0.98041;
std_k1 = 0.00109;
std_k2 = 0.00624;
std_k3 = 0.01126;
std_p1 = 0.00011;
std_p2 = 0.00014;

% Image width and height
W = 2816;
H = 2112;

% This point will project approximately to the lower-right corner (u=W, v=H)
% when using the nominal calibrated camera values
X = [0.64063963 ; 0.46381152 ; 1.0];
uv = project(X, fx, fy, cx, cy, k1, k2, k3, p1, p2);
fprintf('%.02f\n%.02f\n', uv(1), uv(2)); % Check that this is close to (W, H)

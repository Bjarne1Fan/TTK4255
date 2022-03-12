clc;
clear;

% Generate some random 3D points between [-5,+5] in the world frame.
% The coordinates are rounded to integers to make it easier to read.
num_points = 3;
rng(1);
X = round(-5 + 10*rand(4,num_points));
X(4,:) = 1;

% Make up some projection matrices.
P1 = [ 0.9211, 0.0000,-0.3894, 0.0000 ;
       0.0000, 1.0000, 0.0000, 0.0000 ;
       0.3894, 0.0000, 0.9211, 6.0000 ];
P2 = [ 0.9211, 0.0000, 0.3894, 0.0000 ;
       0.0000, 1.0000, 0.0000, 0.0000 ;
      -0.3894, 0.0000, 0.9211, 6.0000 ];

% Perspective projection.
x1 = P1*X;
x2 = P2*X;
x1 = x1./x1(3,:);
x2 = x2./x2(3,:);

X_hat = triangulate_many(x1, x2, P1, P2);

if size(X_hat,1) ~= 4
    fprintf('Triangulation is NOT GOOD. The coordinates should be homogeneous 4-vectors.');
elseif size(X_hat,2) ~= size(X,2)
    fprintf('Triangulation is NOT GOOD. Does not return the same number of points.');
else
    fprintf('True vs. estimated 3D coordinates\n');
    fprintf('---------------------------------\n');
    for i=1:num_points
        fprintf('True: %4.1f %4.1f %4.1f %4.1f\n', X(1,i), X(2,i), X(3,i), X(4,i));
        fprintf('Est.: %4.1f %4.1f %4.1f %4.1f\n', X_hat(1,i), X_hat(2,i), X_hat(3,i), X_hat(4,i));
    end
    if any(vecnorm(X - X_hat) > 1e-10)
        fprintf('Triangulation is NOT GOOD. Estimated points do not match the true points.');
    else
        fprintf('Triangulation is GOOD.');
    end
end

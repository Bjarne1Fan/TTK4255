clc;
clear;

K = load('../data/K.txt');
I1 = im2double(imread('../data/image1.jpg'));
I2 = im2double(imread('../data/image2.jpg'));
matches = load('../data/matches.txt');
% matches = load('../data/task4matches.txt'); % Part 4

uv1 = [matches(:,1:2)' ; ones(1, size(matches, 1))];
uv2 = [matches(:,3:4)' ; ones(1, size(matches, 1))];

%
% Task 2: Estimate E
%
% E = ...

%
% Task 3: Triangulate 3D points
%
% X = ...

%
% Uncomment in Task 2
%
% rng(4); % Leave as commented out to get a random selection each time.
% draw_correspondences(I1, I2, uv1, uv2, F_from_E(E, K));

%
% Uncomment in Task 3
%
% draw_point_cloud(X, I1, uv1, [-1,+1], [-1,+1], [1,3]);
